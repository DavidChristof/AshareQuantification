"""FastAPI 展示层：把训练好的模型和纸面账户包装成 HTTP 接口。

用法：
    python api/main.py
    # 浏览器打开 http://127.0.0.1:8001/docs 查看接口文档
    # 前端看板: frontend/index.html

接口：
    GET /                      基本信息
    GET /api/stocks            股票列表
    GET /api/predict/{symbol}  某只股票最近 N 天的预测概率与信号
    GET /api/backtest/{symbol} 某只股票的回测绩效
    GET /api/dashboard         总览：股票池最新信号 + 账户摘要
    GET /api/account           纸面账户详情
    GET /api/positions         纸面持仓
    GET /api/trades            纸面成交记录
    GET /api/equity            纸面净值历史
"""
from __future__ import annotations

import sys
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                                    # noqa: E402
from fastapi import FastAPI, HTTPException                             # noqa: E402
from fastapi.middleware.cors import CORSMiddleware                     # noqa: E402
from pydantic import BaseModel                                         # noqa: E402

from quant.advisor.advisor import TradeAdvisor                         # noqa: E402
from quant.advisor.weights import learn_weights                        # noqa: E402
from quant.backtest.engine import BacktestEngine                       # noqa: E402
from quant.backtest.metrics import summarize                           # noqa: E402
from quant.config import load_config                                   # noqa: E402
from quant.data.loader import load_all                                 # noqa: E402
from quant.models.cross_model import CrossSectionalPredictor          # noqa: E402
from quant.models.predict import ModelPredictor                        # noqa: E402
from quant.realtime.indices import IndexQuoter                         # noqa: E402
from quant.realtime.manager import QuoteManager                        # noqa: E402
from quant.realtime.minute_manager import MinuteManager                # noqa: E402
from quant.realtime.minute_store import MinuteStore                    # noqa: E402
from quant.timing.engine import TimingEngine                           # noqa: E402
from quant.timing.regime import MarketRegime                           # noqa: E402
from quant.timing.selector import explain as timing_explain            # noqa: E402
from quant.timing.selector import select_weights                       # noqa: E402
from quant.trading.paper import PaperBroker                            # noqa: E402

logger = logging.getLogger(__name__)

cfg = load_config()
app = FastAPI(title="A股量化预测服务", version="0.2.0")

# 允许前端（file:// 直接打开 / 本地静态页）跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 本地开发放开，生产环境应收紧
    allow_methods=["*"],
    allow_headers=["*"],
)

# 股票名称映射（优先用股票池生成的名称表，缺失时回退到内置常用名）
STOCK_NAMES = {
    **{"600519": "贵州茅台", "000001": "平安银行", "000858": "五粮液",
       "300750": "宁德时代", "601318": "中国平安"},
    **cfg.get("data", {}).get("universe_names", {}),
}


def _name(symbol: str) -> str:
    return STOCK_NAMES.get(symbol, symbol)


# 启动时加载数据与模型
DATA = load_all(cfg)

# 每日选股结果（启动时从磁盘恢复，避免重启丢失）
SELECTION_RESULT: dict | None = None
_selection_path = cfg.resolve("results") / "daily_selection.json"
if _selection_path.exists():
    try:
        SELECTION_RESULT = json.loads(_selection_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        SELECTION_RESULT = None
# 模型：优先加载横截面增强 v2（results/model_v2），缺省回退旧模型
# v2 = 相对标签(跑赢全池中位数) + 截面/Alpha101 特征 + LSTM/Transformer/GBM 集成
_v2_dir = cfg.resolve((cfg.get("model_v2") or {}).get("dir", "results/model_v2"))
if (_v2_dir / "meta.json").exists():
    try:
        PREDICTOR = CrossSectionalPredictor(_v2_dir)
        MODEL_LABEL = "v2(relative, " + ",".join(PREDICTOR.members) + ")"
        logger.info("已加载横截面增强模型 v2（成员 %s, 阈值 %.2f）",
                    PREDICTOR.members, PREDICTOR.threshold)
    except Exception as exc:  # noqa: BLE001
        logger.error("v2 模型加载失败，回退旧模型: %s", exc)
        PREDICTOR = None
        MODEL_LABEL = "none(v2加载失败)"
else:
    CHECKPOINT = cfg.resolve("results") / f"{cfg['model']['type']}_model.pt"
    PREDICTOR = ModelPredictor(CHECKPOINT) if CHECKPOINT.exists() else None
    MODEL_LABEL = cfg["model"]["type"]
    if PREDICTOR is not None:
        logger.info("已加载旧模型 %s（绝对涨跌标签，建议运行 18_train_ensemble 升级 v2）",
                    MODEL_LABEL)

# 纸面账户（与 05_update.py 共用同一个库，保证数据一致）
BROKER = PaperBroker(
    cfg.resolve("paper/paper_account.db"),
    initial_capital=cfg["backtest"]["initial_capital"],
    commission=cfg["backtest"]["commission"],
    slippage=cfg["backtest"]["slippage"],
    stamp_tax=cfg["backtest"].get("stamp_tax", 0.0005),
)

# 手动模拟盘（独立账户，10 万初始资金，用于模拟炒股练手）
MANUAL_BROKER = PaperBroker(
    cfg.resolve(cfg["manual"]["db_path"]),
    initial_capital=cfg["manual"]["initial_capital"],
    commission=cfg["backtest"]["commission"],
    slippage=cfg["backtest"]["slippage"],
    stamp_tax=cfg["backtest"].get("stamp_tax", 0.0005),
    lot_size=int(cfg["manual"].get("lot_size", 100)),
)

# 买卖决策辅助引擎
ADVISOR = TradeAdvisor(threshold=cfg["backtest"]["threshold"])

# 择时引擎与市场状态检测
TIMING_ENGINE = TimingEngine(
    threshold=cfg["backtest"]["threshold"],
    sell_line=cfg["backtest"]["threshold"] - 0.10,
)
REGIME_DETECTOR = MarketRegime()

# 实时行情（阶段一：盘中实时看盘）
_rt_cfg = cfg.get("realtime", {})
QUOTE_MANAGER = QuoteManager(
    list(DATA.keys()),
    interval=_rt_cfg.get("interval_seconds", 10),
)
if _rt_cfg.get("enabled", True):
    QUOTE_MANAGER.start()

# 分钟K线（阶段二：分钟级数据）
_min_cfg = cfg.get("minute", {})
MINUTE_STORE = MinuteStore(cfg.resolve(_min_cfg.get("db_path", "data/minute.db")))
MINUTE_MANAGER = MinuteManager(
    MINUTE_STORE,
    list(DATA.keys()),
    scale=_min_cfg.get("scale", 5),
    interval=_min_cfg.get("refresh_interval", 60),
    datalen=_min_cfg.get("datalen", 1023),
    concurrency=_min_cfg.get("concurrency", 4),
)
if _min_cfg.get("enabled", True):
    MINUTE_MANAGER.start()

# 分钟级模型（阶段四：盘中分钟信号，训练后加载）
_MINUTE_MODEL = cfg.resolve("results") / "minute_model.pt"
MINUTE_PREDICTOR = ModelPredictor(_MINUTE_MODEL) if _MINUTE_MODEL.exists() else None
if MINUTE_PREDICTOR:
    logger.info("分钟级模型已加载 (window=%d, horizon=%d)",
                MINUTE_PREDICTOR.window, MINUTE_PREDICTOR.horizon)


def _signal_tables() -> dict[str, pd.DataFrame]:
    """对每只股票生成最新信号表（含日期/收盘/概率/信号）。"""
    if PREDICTOR is None:
        return {}
    if hasattr(PREDICTOR, "make_signals_all"):
        # v2 横截面模型：截面/Alpha 特征需要全池，一次批量预测
        return PREDICTOR.make_signals_all(DATA)
    return {
        symbol: PREDICTOR.make_signal(DATA[symbol], threshold=cfg["backtest"]["threshold"])
        for symbol in DATA
    }


# 启动时预计算，避免每次请求重复推理
SIGNALS = _signal_tables()

# ============ 后台自动刷新 ============
_update_lock = threading.Lock()
_last_updated = None          # 上次自动更新的时间


def _run_auto_update():
    """执行一次完整自动更新：拉数据 → 重算信号 → 自动纸面调仓。"""
    global DATA, SIGNALS, _last_updated
    if PREDICTOR is None:
        return
    with _update_lock:
        try:
            from quant.trading.updater import (  # noqa: PLC0415
                latest_trade_date, rebalance_auto, rebuild_signals, refresh_market_data,
            )
            logger.info("[auto] 开始自动更新 ...")
            refresh_market_data(cfg)
            data, signals = rebuild_signals(cfg, PREDICTOR)
            date = latest_trade_date(data)
            ar = cfg.get("auto_refresh", {})
            if ar.get("auto_rebalance", True):
                rebalance_auto(cfg, BROKER, signals, date, data)
            DATA, SIGNALS = data, signals
            _last_updated = datetime.now()
            logger.info("[auto] 自动更新完成，最新交易日 %s", date)
            # 每日选股（收盘后随自动刷新一起跑，后台线程）
            if cfg.get("selection", {}).get("auto", True):
                threading.Thread(target=_run_daily_selection, daemon=True).start()
        except Exception as exc:  # noqa: BLE001
            logger.error("[auto] 自动更新失败: %s", exc, exc_info=True)


def _scheduler():
    """后台调度循环：每个交易日收盘后自动更新一次。"""
    ar = cfg.get("auto_refresh", {})
    interval = ar.get("check_interval_minutes", 30) * 60
    update_time = ar.get("update_time", "15:30")
    enabled = ar.get("enabled", True)
    logger.info("[auto] 定时刷新已启动：每 %d 分钟检查一次，工作日 %s 后更新",
                interval // 60, update_time)
    while True:
        time.sleep(interval)
        if not enabled:
            continue
        now = datetime.now()
        if now.weekday() >= 5:                     # 周末休市
            continue
        target_h, target_m = map(int, update_time.split(":"))
        if (now.hour, now.minute) < (target_h, target_m):   # 还没到收盘后
            continue
        if _last_updated is not None and _last_updated.date() == now.date():
            continue                               # 今天已更新过
        _run_auto_update()


# 启动后台调度线程（daemon，随主进程退出）
threading.Thread(target=_scheduler, daemon=True).start()


@app.get("/")
def root():
    return {
        "service": "A股量化预测",
        "stocks": list(DATA.keys()),
        "model_loaded": PREDICTOR is not None,
        "model": MODEL_LABEL,
        "window": PREDICTOR.window if PREDICTOR else None,
        "horizon": PREDICTOR.horizon if PREDICTOR else None,
        "paper_equity": BROKER.account_summary()["equity"],
        "last_updated": _last_updated.strftime("%Y-%m-%d %H:%M:%S") if _last_updated else None,
    }


@app.get("/api/stocks")
def stocks():
    return {"stocks": [{"symbol": s, "name": _name(s)} for s in DATA]}


@app.get("/api/dashboard")
def dashboard():
    """总览：每只股票最新信号 + 账户摘要。"""
    if PREDICTOR is None:
        raise HTTPException(500, "模型未加载，请先运行 scripts/03_train.py")

    items = []
    for symbol, signal in SIGNALS.items():
        if signal.empty:
            continue
        latest = signal.iloc[-1]
        items.append({
            "symbol": symbol,
            "name": _name(symbol),
            "date": str(latest.name.date()),
            "close": float(latest["close"]),
            "prob_up": round(float(latest["prob_up"]), 4),
            "signal": int(latest["signal"]),
        })
    # 按上涨概率降序
    items.sort(key=lambda x: -x["prob_up"])
    return {
        "stocks": items,
        "account": BROKER.account_summary(),
        "last_updated": _last_updated.strftime("%Y-%m-%d %H:%M:%S") if _last_updated else None,
    }


@app.get("/api/predict/{symbol}")
def predict(symbol: str, days: int = 30):
    if PREDICTOR is None:
        raise HTTPException(500, "模型未加载，请先运行 scripts/03_train.py")
    if symbol not in DATA:
        raise HTTPException(404, f"股票 {symbol} 不在股票池中")

    signal = SIGNALS.get(symbol)
    if signal is None:
        raise HTTPException(500, "预测信号未生成")
    recent = signal.tail(days).reset_index()
    recent["date"] = recent["date"].astype(str)
    return {
        "symbol": symbol,
        "name": _name(symbol),
        "model": MODEL_LABEL,
        "recent": recent.to_dict(orient="records"),
    }


@app.get("/api/backtest/{symbol}")
def backtest(symbol: str):
    if symbol not in DATA:
        raise HTTPException(404, f"股票 {symbol} 不在股票池中")

    signal = SIGNALS.get(symbol)
    if signal is None:
        raise HTTPException(500, "预测信号未生成")
    bt = cfg["backtest"]
    engine = BacktestEngine(
        initial_capital=bt["initial_capital"], commission=bt["commission"],
        slippage=bt["slippage"], position_pct=bt["position_pct"],
    )
    strat_equity = engine.run(signal)["equity"]
    bh_equity = engine.buy_and_hold(signal)["equity"]

    return {
        "symbol": symbol,
        "name": _name(symbol),
        "strategy": summarize(strat_equity, "strategy"),
        "buy_and_hold": summarize(bh_equity, "buy&hold"),
    }


@app.get("/api/account")
def account():
    return BROKER.account_summary()


@app.get("/api/positions")
def positions():
    prices = {s: sig.iloc[-1]["close"] for s, sig in SIGNALS.items() if not sig.empty}
    risk = _risk_config()
    vol_map = _build_vol_map(risk)
    vol_cfg = _build_vol_cfg(risk) if vol_map else None
    result = []
    for pos in BROKER.query_positions():
        price = prices.get(pos.symbol, pos.avg_cost)
        market_value = pos.shares * price
        r = _position_risk(pos.symbol, pos.avg_cost, vol_map, vol_cfg)
        result.append({
            "symbol": pos.symbol,
            "name": _name(pos.symbol),
            "shares": round(pos.shares, 2),
            "avg_cost": round(pos.avg_cost, 3),
            "price": round(price, 3),
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(market_value - pos.shares * pos.avg_cost, 2),
            "pnl_pct": round(market_value / (pos.shares * pos.avg_cost) - 1, 4)
            if pos.shares * pos.avg_cost else 0.0,
            "stop_price": round(r["stop_price"], 3),
            "take_price": round(r["take_price"], 3),
            "stop_dist": round((price / r["stop_price"] - 1) * 100, 1),
            "take_dist": round((price / r["take_price"] - 1) * 100, 1),
            "sl_pct": r["sl_pct"], "tp_pct": r["tp_pct"],
            "mode": r["mode"], "atr_pct": r["atr_pct"],
        })
    return {"positions": result}


@app.get("/api/trades")
def trades(limit: int = 50):
    return {"trades": BROKER.trade_history(limit=limit)}


@app.get("/api/equity")
def equity():
    hist = BROKER.equity_history()
    if not hist:
        return {"equity_curve": []}
    df = pd.DataFrame(hist)
    return {
        "equity_curve": [
            {"date": r["date"], "cash": r["cash"],
             "market_value": r["market_value"], "equity": r["equity"]}
            for _, r in df.iterrows()
        ],
    }


# ---------- 分钟K线 ----------
@app.get("/api/minute/{symbol}")
def minute_bars(symbol: str, days: int = 1):
    """分钟K线（默认 5 分钟，最近 days 天），交易时段内附分钟级信号。"""
    if symbol not in DATA:
        raise HTTPException(404, f"股票 {symbol} 不在股票池")
    result = MINUTE_MANAGER.bars(symbol, days=days)

    # 附分钟级模型预测（只在交易时段内，收盘后"未来25分钟"不存在）
    if MINUTE_PREDICTOR is not None:
        df = MINUTE_STORE.load_symbol(symbol, scale=MINUTE_MANAGER.scale, days=1)
        if not df.empty:
            latest_t = df["datetime"].iloc[-1]
            now = datetime.now()
            hm = latest_t.hour * 100 + latest_t.minute
            is_trading = (latest_t.date() == now.date()
                          and ((930 <= hm <= 1130) or (1300 <= hm < 1500)))
            if is_trading:
                prob = MINUTE_PREDICTOR.latest_probability(
                    df.rename(columns={"datetime": "date"}))
                if prob is not None:
                    result["minute_prob"] = round(prob, 4)
                    result["minute_signal"] = (
                        "buy" if prob >= 0.55 else ("sell" if prob <= 0.45 else "hold"))
            else:
                result["minute_note"] = "收盘后无盘中信号（交易时段 9:30-11:30 / 13:00-15:00 内显示）"
    return result


# ---------- 动态选股权重 ----------
@app.get("/api/weights")
def dynamic_weights():
    """动态选股权重：市场因子有效性 + 用户模拟盘盈利偏好。"""
    prices = {s: sig.iloc[-1]["close"] for s, sig in SIGNALS.items() if not sig.empty}
    profit_symbols = []
    for pos in MANUAL_BROKER.query_positions():
        price = prices.get(pos.symbol, pos.avg_cost)
        pnl = pos.shares * price - pos.shares * pos.avg_cost
        if pnl > 0:
            profit_symbols.append(pos.symbol)
    return learn_weights(cfg, DATA, profit_symbols)


# ---------- 择时 ----------
def _market_proxy():
    """股票池等权平均，作为市场代理指数。"""
    closes = pd.DataFrame({s: df.set_index("date")["close"] for s, df in DATA.items()})
    return closes.mean(axis=1).dropna()


@app.get("/api/timing")
def timing():
    """择时：市场状态 + 自主选择的方法权重 + 每只股票买卖点信号。"""
    if PREDICTOR is None:
        raise HTTPException(500, "模型未加载")
    proxy = _market_proxy()
    regime = REGIME_DETECTOR.detect(proxy)
    weights = select_weights(regime["regime"])

    signals = []
    for symbol, bars in DATA.items():
        sig = SIGNALS.get(symbol)
        if sig is None or sig.empty:
            continue
        prob = float(sig.iloc[-1]["prob_up"])
        r = TIMING_ENGINE.analyze(symbol, bars, prob, regime["regime"])
        r["name"] = _name(symbol)
        r["close"] = round(float(sig.iloc[-1]["close"]), 3)
        signals.append(r)

    order = {"buy": 0, "sell": 1, "hold": 2}
    signals.sort(key=lambda x: (order[x["action"]], -x["score"]))
    return {
        "regime": regime,
        "weights": weights,
        "explain": timing_explain(regime["regime"]),
        "signals": signals,
    }


# ---------- 盘中实时行情 ----------
def _market_status(quotes: list[dict]) -> dict:
    """根据最新快照时间戳判断市场状态（处理新浪/腾讯两种时间格式）。"""
    if not quotes:
        return {"code": "unknown", "text": "加载中"}
    t = quotes[0].get("time", "")
    try:
        if len(t) == 19:          # 新浪: YYYY-MM-DD HH:MM:SS
            dt = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
        elif len(t) == 14:        # 腾讯: YYYYMMDDHHMMSS
            dt = datetime.strptime(t, "%Y%m%d%H%M%S")
        else:
            return {"code": "unknown", "text": "状态未知"}
    except ValueError:
        return {"code": "unknown", "text": "状态未知"}

    now = datetime.now()
    if dt.date() == now.date():
        hm = dt.hour * 100 + dt.minute
        if 930 <= hm <= 1130 or 1300 <= hm <= 1500:   # 交易时段内
            return {"code": "open", "text": "🟢 盘中实时"}
        return {"code": "closed_today", "text": f"⏸ 今日已收盘 {dt.strftime('%H:%M')}"}
    # 非今天：休市（周末/节假日），显示最近交易日快照
    return {"code": "market_closed", "text": f"🛑 休市中 · 最近交易日 {dt.strftime('%m-%d %H:%M')} 快照"}


@app.get("/api/realtime")
def realtime():
    """盘中实时快照（现价/涨跌幅/五档/量额），由后台轮询缓存提供。"""
    quotes = QUOTE_MANAGER.snapshot()
    last = QUOTE_MANAGER.last_update
    return {
        "quotes": quotes,
        "last_update": last.strftime("%Y-%m-%d %H:%M:%S") if last else None,
        "interval": QUOTE_MANAGER.interval,
        "market": _market_status(quotes),
    }


# ---------- 大盘指数（市场参考） ----------
@app.get("/api/market/indices")
def market_indices():
    """大盘指数：上证/深成/创业板/沪深300 点数与涨跌幅（腾讯源，短缓存）。"""
    indices = IndexQuoter().fetch()
    # 涨跌家数统计（股票池实时快照）：上涨数 vs 下跌数作市场温度
    quotes = QUOTE_MANAGER.snapshot()
    up = sum(1 for q in quotes if q.get("change_pct", 0) > 0)
    down = sum(1 for q in quotes if q.get("change_pct", 0) < 0)
    flat = max(0, len(quotes) - up - down)
    return {
        "indices": indices,
        "breadth": {"up": up, "down": down, "flat": flat, "total": len(quotes)},
        "last_update": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/market/indices/kline")
def market_indices_kline(code: str = "sh000001", days: int = 120):
    """某大盘指数的日 K 线（新浪源，30 分钟缓存）。"""
    try:
        from quant.realtime.indices import fetch_index_daily
        df = fetch_index_daily(code)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"指数日线获取失败: {exc}")
    tail = df.tail(max(1, min(days, 1000)))
    return {
        "code": code,
        "name": IndexQuoter().codes.get(code, code),
        "bars": [{"date": str(d.date()), "open": float(o), "high": float(h),
                  "low": float(l), "close": float(c), "volume": float(v)}
                 for d, o, h, l, c, v in zip(
                     tail["date"], tail["open"], tail["high"],
                     tail["low"], tail["close"], tail["volume"])],
    }


# ---------- 买卖决策辅助 ----------
@app.get("/api/advisor")
def advisor():
    """每只股票的买卖建议（买入/卖出/观望 + 理由）。"""
    if PREDICTOR is None:
        raise HTTPException(500, "模型未加载，请先运行 scripts/03_train.py")

    manual_positions = {p.symbol for p in MANUAL_BROKER.query_positions()}
    items = []
    for symbol, bars in DATA.items():
        sig = SIGNALS.get(symbol)
        if sig is None or sig.empty:
            continue
        prob = float(sig.iloc[-1]["prob_up"])
        adv = ADVISOR.analyze(symbol, bars, prob,
                              holding=symbol in manual_positions)
        items.append({
            "symbol": symbol,
            "name": _name(symbol),
            "action": adv.action,
            "label": adv.label,
            "prob_up": round(adv.prob_up, 4),
            "score": adv.score,
            "holding": adv.holding,
            "reasons": adv.reasons,
            "close": float(sig.iloc[-1]["close"]),
        })
    # 排序：买入 > 卖出 > 观望，组内按概率降序
    order = {"buy": 0, "sell": 1, "wait": 2}
    items.sort(key=lambda x: (order[x["action"]], -x["prob_up"]))
    return {"advice": items}


# ---------- 手动模拟盘 ----------
def _risk_config() -> dict:
    """止盈止损配置。"""
    return cfg.get("risk", {})


def _build_vol_map(risk: dict) -> dict:
    """按 risk 配置构建每只股票的 ATR 波动率信息（动态止盈止损用）。"""
    if not risk.get("dynamic_volatility", True):
        return {}
    from quant.risk.volatility import build_vol_map
    return build_vol_map(DATA, risk.get("vol_window", 20))


def _build_vol_cfg(risk: dict) -> dict:
    """提取动态止盈止损参数。"""
    from quant.risk.volatility import vol_cfg_from_risk
    return vol_cfg_from_risk(risk)


def _position_risk(symbol: str, cost: float, vol_map: dict, vol_cfg: dict) -> dict:
    """计算某持仓的止损/止盈位：动态波动率优先，回退固定百分比。

    Returns: {stop_price, take_price, sl_pct, tp_pct, mode(动态/固定), atr_pct}
    """
    risk = _risk_config()
    sl = risk.get("stop_loss_pct", 0.08)
    tp = risk.get("take_profit_pct", 0.15)
    mode, atr_pct = "fixed", None
    if vol_map and vol_cfg and cost > 0:
        v = vol_map.get(symbol)
        if v and v.get("atr", 0) > 0:
            from quant.risk.volatility import dynamic_pcts
            dyn = dynamic_pcts(
                v, cost,
                stop_mult=vol_cfg["atr_stop_mult"], take_mult=vol_cfg["atr_take_mult"],
                trail_mult=vol_cfg["atr_trailing_mult"],
                min_pct=vol_cfg["vol_min_pct"], max_pct=vol_cfg["vol_max_pct"],
                take_min_pct=vol_cfg["take_min_pct"], take_max_pct=vol_cfg["take_max_pct"],
            )
            if dyn["stop_pct"]:
                sl, tp = dyn["stop_pct"], dyn["take_pct"]
                mode, atr_pct = "dynamic", v["atr_pct"]
    return {
        "stop_price": cost * (1 - sl),
        "take_price": cost * (1 + tp),
        "sl_pct": sl, "tp_pct": tp,
        "mode": mode, "atr_pct": atr_pct,
    }


def _apply_manual_stops(prices: dict) -> list:
    """手动模拟盘止盈止损检查（查询时触发，触发则自动平仓）。"""
    risk = _risk_config()
    if not risk.get("enabled", True):
        return []
    dates = [sig.index[-1].date() for sig in SIGNALS.values() if not sig.empty]
    if not dates:
        return []
    vol_map = _build_vol_map(risk)
    vol_cfg = _build_vol_cfg(risk) if vol_map else None
    return MANUAL_BROKER.apply_stop_rules(
        str(max(dates)), prices,
        stop_loss_pct=risk.get("stop_loss_pct", 0.08),
        take_profit_pct=risk.get("take_profit_pct", 0.15),
        trailing_pct=risk.get("trailing_pct") if risk.get("trailing_stop", False) else None,
        vol=vol_map, vol_cfg=vol_cfg,
    )


def _sync_manual_equity():
    """手动盘净值快照：盘中每小时记一个实时点，收盘后对齐最新交易日（日点）。

    盘中用实时估值（_live_prices），让净值曲线时间轴细化到小时；
    同小时只快照一次，避免前端轮询重复写入。
    """
    try:
        hist = MANUAL_BROKER.equity_history()
        # 交易时段内：每小时整点快照一次（实时估值）
        if _in_trading_hours():
            stamp = datetime.now().strftime("%Y-%m-%d %H:00")
            hour_key = datetime.now().strftime("%Y-%m-%d %H")
            if hist and str(hist[-1]["date"]).startswith(hour_key):
                return
            MANUAL_BROKER.snapshot_equity(stamp, _live_prices())
            return
        # 非交易时段：对齐最新交易日（日线收盘点）
        dates = [sig.index[-1] for sig in SIGNALS.values() if not sig.empty]
        if not dates:
            return
        latest = max(dates).date()
        if hist and str(hist[-1]["date"]) >= str(latest):
            return
        prices = {s: float(sig.iloc[-1]["close"]) for s, sig in SIGNALS.items() if not sig.empty}
        MANUAL_BROKER.snapshot_equity(str(latest), prices)
    except Exception:  # noqa: BLE001
        logger.exception("手动盘净值同步失败")


def _live_prices() -> dict:
    """实时价优先（10 秒轮询快照），无实时价回退日线最新收盘。

    用于账户/持仓的实时估值——盘中总资产随实时行情同步。
    """
    prices = {}
    for q in QUOTE_MANAGER.snapshot():
        if q.get("price"):
            prices[q["symbol"]] = float(q["price"])
    for s, sig in SIGNALS.items():
        if s not in prices and sig is not None and not sig.empty:
            prices[s] = float(sig["close"].iloc[-1])
    return prices


@app.get("/api/manual/account")
def manual_account():
    prices = _live_prices()          # 实时价优先（盘中总资产随行情同步）
    _apply_manual_stops(prices)      # 止盈止损按最新价检查
    _sync_manual_equity()            # 净值曲线对齐最新交易日（历史快照仍按日线）
    return MANUAL_BROKER.live_summary(prices)


@app.get("/api/manual/positions")
def manual_positions():
    prices = _live_prices()          # 实时价优先（持仓现价/市值随行情同步）
    _apply_manual_stops(prices)      # 查询前先检查止盈止损（自动平仓）
    _sync_manual_equity()            # 净值对齐最新交易日
    risk = _risk_config()
    vol_map = _build_vol_map(risk)
    vol_cfg = _build_vol_cfg(risk) if vol_map else None
    result = []
    for pos in MANUAL_BROKER.query_positions():
        price = prices.get(pos.symbol, pos.avg_cost)
        mv = pos.shares * price
        r = _position_risk(pos.symbol, pos.avg_cost, vol_map, vol_cfg)
        result.append({
            "symbol": pos.symbol,
            "name": _name(pos.symbol),
            "shares": round(pos.shares, 2),
            "avg_cost": round(pos.avg_cost, 3),
            "price": round(price, 3),
            "market_value": round(mv, 2),
            "unrealized_pnl": round(mv - pos.shares * pos.avg_cost, 2),
            "pnl_pct": round(mv / (pos.shares * pos.avg_cost) - 1, 4)
            if pos.shares * pos.avg_cost else 0.0,
            "stop_price": round(r["stop_price"], 3),
            "take_price": round(r["take_price"], 3),
            "stop_dist": round((price / r["stop_price"] - 1) * 100, 1),
            "take_dist": round((price / r["take_price"] - 1) * 100, 1),
            "sl_pct": r["sl_pct"], "tp_pct": r["tp_pct"],
            "mode": r["mode"], "atr_pct": r["atr_pct"],
        })
    return {"positions": result}


@app.get("/api/manual/trades")
def manual_trades(limit: int = 50):
    return {"trades": MANUAL_BROKER.trade_history(limit=limit)}


@app.get("/api/manual/equity")
def manual_equity():
    return {"equity_curve": MANUAL_BROKER.equity_history()}


def _in_trading_hours() -> bool:
    """当前是否处于 A股交易时段（工作日 9:30-11:30 / 13:00-15:00）。"""
    if not cfg.get("manual", {}).get("enforce_trading_hours", True):
        return True
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    hm = now.hour * 100 + now.minute
    return 930 <= hm <= 1130 or 1300 <= hm <= 1500


class OrderRequest(BaseModel):
    symbol: str
    side: str        # 'buy' / 'sell'
    shares: float


@app.post("/api/manual/order")
def manual_order(order: OrderRequest):
    """手动下单：用最新收盘价在模拟盘成交（仅限交易时段）。"""
    # 交易时间限制：闭市/休市禁止下单（可配置关闭）
    if not _in_trading_hours():
        if datetime.now().weekday() >= 5:
            raise HTTPException(400, "今日休市（周末/节假日），无法下单")
        raise HTTPException(
            400, "非交易时段无法下单（A股交易时间 9:30-11:30 / 13:00-15:00，周一至周五）")
    if order.side not in ("buy", "sell"):
        raise HTTPException(400, "side 必须是 buy 或 sell")
    if order.shares <= 0:
        raise HTTPException(400, "数量必须为正")
    if order.shares > cfg["manual"]["max_order_shares"]:
        raise HTTPException(400, f"单笔最多 {cfg['manual']['max_order_shares']} 股（风控）")
    if order.symbol not in DATA:
        raise HTTPException(404, f"股票 {order.symbol} 不在股票池")

    sig = SIGNALS.get(order.symbol)
    if sig is None or sig.empty:
        raise HTTPException(500, "该股票无预测信号")
    if len(sig) < 2:
        raise HTTPException(500, "该股票历史数据不足，无法判断涨跌停")
    price = float(sig.iloc[-1]["close"])
    prev_close = float(sig.iloc[-2]["close"])
    today = str(sig.index[-1].date())

    # 涨跌停校验：一字涨停买不进、一字跌停卖不出（30/68 开头为创业板/科创板 ±20%）
    from quant.trading.rules import limit_prices
    limit_up, limit_down = limit_prices(prev_close, order.symbol)
    if order.side == "buy" and price >= limit_up - 1e-6:
        raise HTTPException(400, f"涨停封板（现价{price:.2f}=涨停价{limit_up:.2f}），无法买入")
    if order.side == "sell" and price <= limit_down + 1e-6:
        raise HTTPException(400, f"跌停封板（现价{price:.2f}=跌停价{limit_down:.2f}），无法卖出")

    if order.side == "buy":
        result = MANUAL_BROKER.buy(order.symbol, order.shares, price, today)
    else:
        result = MANUAL_BROKER.sell(order.symbol, order.shares, price, today)

    if not result.success:
        raise HTTPException(400, result.message)

    # 成交后按最新价快照净值
    latest_prices = {s: float(t["close"].iloc[-1]) for s, t in SIGNALS.items() if not t.empty}
    MANUAL_BROKER.snapshot_equity(today, latest_prices)

    return {
        "success": True,
        "trade": {"date": today, "symbol": order.symbol, "side": order.side,
                   "shares": result.shares, "price": result.price,
                   "fee": result.fee, "amount": result.amount},
        "account": MANUAL_BROKER.account_summary(),
    }


# ---------- 组合模式（多股票动态调仓 · 核心功能） ----------
def _minute_signal(symbol: str) -> dict | None:
    """盘中分钟模型信号（未来 25 分钟上涨概率）。非交易时段/无数据返回 None。

    解决「开盘前不挂单、盘中决策」的痛点：日线模型只能看昨天收盘后的信号，
    无法感知当日开盘走势（如开盘大跌）。分钟模型用当日最新 5 分钟 K 线，
    能实时捕捉盘中变化，作为日线决策的盘中修正。
    """
    if MINUTE_PREDICTOR is None:
        return None
    try:
        df = MINUTE_STORE.load_symbol(symbol, scale=MINUTE_MANAGER.scale, days=1)
        if df is None or df.empty:
            return None
        latest_t = df["datetime"].iloc[-1]
        now = datetime.now()
        hm = latest_t.hour * 100 + latest_t.minute
        if not (latest_t.date() == now.date()
                and ((930 <= hm <= 1130) or (1300 <= hm < 1500))):
            return None
        prob = MINUTE_PREDICTOR.latest_probability(
            df.rename(columns={"datetime": "date"}))
        if prob is None:
            return None
        sig = "buy" if prob >= 0.55 else ("sell" if prob <= 0.45 else "hold")
        return {"minute_prob": round(prob, 4), "minute_signal": sig}
    except Exception:  # noqa: BLE001
        return None


# ---------- 交易候选来源：大池每日选股（selection） ----------
def _display_name(symbol: str) -> str:
    """优先用最近每日选股候选里的真名（含池外股票），否则回退内置映射。"""
    for r in _selection_rows():
        if str(r.get("code")) == symbol and r.get("name"):
            return r["name"]
    return _name(symbol)


def _selection_rows() -> list[dict]:
    """最近一次每日选股结果（600 只大池 topN，含 price/综合分）。无则返回 []。"""
    sel = SELECTION_RESULT or {}
    return sel.get("candidates") or []


def _portfolio_from_selection(rows: list[dict], n: int):
    """组合目标 = 今日选股 topN（综合分=基本面0.6+技术0.4，0~100）。

    池内（40 训练池）候选叠加模型概率/择时/技术面/分钟作为参考列；池外只有
    大池选股综合分（仍可交易——价格来自选股收盘价 + 实时快照）。
    Returns: (targets, target_set, held, regime)，targets 内部字段与旧逻辑一致。
    """
    held = {p.symbol for p in MANUAL_BROKER.query_positions()}
    regime = None
    try:
        regime = REGIME_DETECTOR.detect(_market_proxy())
    except Exception:  # noqa: BLE001
        pass
    targets = []
    for r in rows[:n]:
        symbol = str(r["code"])
        close = r.get("price")
        prob = None
        timing = adv = minute = None
        reasons = []
        if symbol in DATA:                       # 在训练池：叠加模型参考
            sig = SIGNALS.get(symbol)
            if sig is not None and not sig.empty:
                prob = float(sig["prob_up"].iloc[-1])
                close = close or float(sig["close"].iloc[-1])
            base_p = prob if prob is not None else 0.5
            try:
                rr = TIMING_ENGINE.analyze(symbol, DATA[symbol], base_p,
                                           regime.get("regime") if regime else None)
                timing = {"action": rr["action"], "score": float(rr["score"])}
            except Exception:  # noqa: BLE001
                pass
            try:
                aa = ADVISOR.analyze(symbol, DATA[symbol], base_p,
                                     holding=symbol in held)
                adv = {"action": aa.action, "label": aa.label,
                       "score": float(aa.score), "reasons": aa.reasons}
                reasons = aa.reasons
            except Exception:  # noqa: BLE001
                pass
            minute = _minute_signal(symbol) if _in_trading_hours() else None
        targets.append({
            "symbol": symbol, "name": r.get("name", _name(symbol)),
            "prob": prob, "close": close,
            "score": float(r.get("total_score", 0.0)),   # 选股综合分（0~100）
            "score_scale": 100,                           # 标记 0~100 量纲（model40 为 0~1）
            "timing": timing, "advisor": adv, "minute": minute,
            "reasons": reasons, "in_universe": symbol in DATA,
        })
    target_set = {t["symbol"] for t in targets}
    return targets, target_set, held, regime


def _build_prices(symbols) -> dict[str, float]:
    """现价：实时快照优先 → SIGNALS 收盘 → 选股候选 price 兜底。"""
    symbols = set(symbols)
    prices: dict[str, float] = {}
    for s in symbols:
        sig = SIGNALS.get(s)
        if sig is not None and not sig.empty:
            prices[s] = float(sig["close"].iloc[-1])
    for r in _selection_rows():
        c = str(r.get("code"))
        if c in symbols and r.get("price"):
            prices.setdefault(c, float(r["price"]))
    try:
        for q in QUOTE_MANAGER.snapshot():
            if q.get("symbol") in symbols and q.get("price"):
                prices[q["symbol"]] = float(q["price"])
    except Exception:  # noqa: BLE001
        pass
    return prices


def _portfolio_with_reasons(n: int | None = None):
    """topN 组合目标（供组合面板与一键调仓）。

    交易候选来源（config trading.candidate_source）：
        - selection（默认）：今日选股 topN（600 只大池），综合分=基本面0.6+技术0.4；
          池内候选叠加模型概率/择时/分钟作参考
        - model40：旧逻辑——40 只训练池按模型综合分 topN
    返回 (targets, target_set, held, regime)
    """
    n = n or cfg.get("trading", {}).get("max_positions", 5)
    # ---- 大池每日选股候选（有价可交易才用，否则回退 model40） ----
    if cfg.get("trading", {}).get("candidate_source", "selection") == "selection":
        trade = [r for r in _selection_rows() if r.get("price")]
        if trade:
            return _portfolio_from_selection(trade, n)
        logger.warning("每日选股候选为空或无可交易价，回退 model40 组合")
    regime = None
    try:
        regime = REGIME_DETECTOR.detect(_market_proxy())
    except Exception:  # noqa: BLE001
        pass
    held = {p.symbol for p in MANUAL_BROKER.query_positions()}
    rows = []
    for symbol, sig in SIGNALS.items():
        if sig is None or sig.empty:
            continue
        prob = float(sig["prob_up"].iloc[-1])
        close = float(sig["close"].iloc[-1])
        # 择时信号
        timing = None
        try:
            r = TIMING_ENGINE.analyze(symbol, DATA[symbol], prob,
                                      regime["regime"] if regime else None)
            timing = {"action": r["action"], "score": float(r["score"])}
        except Exception:  # noqa: BLE001
            pass
        # 技术面建议（模型概率 + 均线/RSI/MACD/动量投票）
        adv = None
        try:
            a = ADVISOR.analyze(symbol, DATA[symbol], prob, holding=symbol in held)
            adv = {"action": a.action, "label": a.label, "score": float(a.score),
                   "reasons": a.reasons}
        except Exception:  # noqa: BLE001
            pass
        score = prob
        if timing:
            score += timing["score"] * 0.10
        if adv:
            score += {"buy": 0.05, "sell": -0.05}.get(adv["action"], 0.0)
        # 盘中分钟模型实时修正（捕捉当日开盘走势，收盘后无分钟信号）
        # 注意：分钟模型概率偏极端（未校准），故用「方向化」固定修正而非概率放大，
        # 避免一个极端概率过度压垮日线综合分
        minute = _minute_signal(symbol) if _in_trading_hours() else None
        if minute:
            minute_adj = cfg.get("trading", {}).get("minute_adj", 0.05)
            score += {"buy": minute_adj, "sell": -minute_adj, "hold": 0.0}.get(
                minute["minute_signal"], 0.0)
        rows.append({"symbol": symbol, "prob": prob, "close": close,
                     "timing": timing, "advisor": adv, "minute": minute,
                     "score": score})
    rows.sort(key=lambda r: r["score"], reverse=True)
    targets = rows[:n]
    target_set = {t["symbol"] for t in targets}
    return targets, target_set, held, regime


def _portfolio_targets(n: int | None = None):
    """（兼容旧接口）只取目标列表。"""
    targets, target_set, held, _ = _portfolio_with_reasons(n)
    return targets, target_set, held


def _market_weakness() -> dict:
    """大盘弱势检测：沪深300/上证当日跌幅 < 阈值 或 市场状态为 downtrend。

    用于组合调仓风控——大盘普跌日（如 9/2 创业板 -2.4%）满仓会被系统性拖累。
    """
    pr = cfg.get("portfolio_risk", {})
    thr = float(pr.get("market_weak_threshold", -1.0))
    idx_pct = None
    try:
        for i in IndexQuoter().fetch():
            if i.get("code") in ("sh000300", "sh000001"):
                p = i.get("change_pct")
                idx_pct = min(idx_pct, p) if idx_pct is not None else p
    except Exception:  # noqa: BLE001
        pass
    regime = None
    try:
        regime = REGIME_DETECTOR.detect(_market_proxy()).get("regime")
    except Exception:  # noqa: BLE001
        pass
    weak = bool((idx_pct is not None and idx_pct < thr) or regime == "downtrend")
    return {"weak": weak,
            "index_pct": round(idx_pct, 2) if idx_pct is not None else None,
            "regime": regime, "threshold": thr}


def _portfolio_allocation(targets: list, target_set: set, held: set,
                          prices: dict,
                          market_weak: dict | None = None) -> list[dict]:
    """基于现有资金 + 风控决策：等权目标、个股上限、大盘弱势降仓、分钟否决。

    规则（对应复盘改进）：
        - 总仓位预算 = 总资产 × position_pct（默认 95%）；**大盘弱势降到 weak_position_pct**
        - 每只目标等权 = 预算/目标数，但**不超过单票上限 max_stock_pct×总资产**
        - 已持有达标 → 持有；不足 → 加仓；未持有 → 买入（整手 100）
        - **盘中分钟 sell（未来25分钟看空）且 minute_veto → 暂停买入该目标**（action=skip）
    """
    pr = cfg.get("portfolio_risk", {})
    cash = MANUAL_BROKER.query_cash()
    held_value = 0.0
    held_vals = {}
    for p in MANUAL_BROKER.query_positions():
        v = prices.get(p.symbol, p.avg_cost) * p.shares
        held_value += v
        held_vals[p.symbol] = v
    total_assets = cash + held_value
    weak = bool(market_weak and market_weak.get("weak"))
    pos_pct = (pr.get("weak_position_pct", 0.5) if weak
               else cfg["backtest"].get("position_pct", 0.95))
    budget = total_assets * pos_pct
    n = max(len(targets), 1)
    per = budget / n
    cap = total_assets * pr.get("max_stock_pct", 0.20)      # 单票上限
    target_val = min(per, cap)
    slip = MANUAL_BROKER.slippage
    out = []
    for t in targets:
        symbol = t["symbol"]
        price = prices.get(symbol)
        if price is None or price <= 0:
            continue
        cur = held_vals.get(symbol, 0.0)
        # 盘中分钟否决：未来25分钟强烈看空 → 暂停买入/加仓
        minute = t.get("minute")
        if pr.get("minute_veto", True) and minute \
                and minute.get("minute_signal") == "sell":
            out.append({"symbol": symbol, "name": _display_name(symbol),
                        "action": "skip", "reason": "盘中分钟看空，暂停买入",
                        "current_value": round(cur, 2),
                        "target_value": round(target_val, 2),
                        "est_shares": 0, "est_amount": 0.0,
                        "price": round(price, 3)})
            continue
        if symbol in held and cur >= target_val - 1:
            out.append({"symbol": symbol, "name": _display_name(symbol),
                        "action": "hold", "reason": "已达目标仓位，持有",
                        "current_value": round(cur, 2),
                        "target_value": round(target_val, 2),
                        "est_shares": 0, "est_amount": 0.0})
            continue
        need = target_val - cur if symbol in held else target_val
        shares = int(need / (price * (1 + slip)) // 100) * 100
        amount = shares * price
        if shares < 100:
            # 目标价位太高/现金不足一手：明确暂停并给出原因，避免"buy 0股"误导
            out.append({"symbol": symbol, "name": _display_name(symbol),
                        "action": "skip", "reason": "资金不足以买入一手(100股)",
                        "current_value": round(cur, 2),
                        "target_value": round(target_val, 2),
                        "est_shares": 0, "est_amount": 0.0,
                        "price": round(price, 3)})
            continue
        out.append({
            "symbol": symbol, "name": _display_name(symbol),
            "action": "add" if symbol in held else "buy",
            "reason": "加仓至目标仓位" if symbol in held else "新进组合",
            "current_value": round(cur, 2), "target_value": round(target_val, 2),
            "est_shares": shares, "est_amount": round(amount, 2),
            "price": round(price, 3),
        })
    return out


@app.get("/api/portfolio")
def portfolio():
    """组合模式（核心）：topN 目标（融合模型+择时+技术面+分钟）+ 持仓 + 调仓清单 + 资金决策预览。"""
    targets, target_set, held, regime = _portfolio_with_reasons()
    market_weak = _market_weakness()
    actions = []
    for t in targets:
        if t["symbol"] not in held:
            actions.append({"symbol": t["symbol"], "name": _display_name(t["symbol"]),
                            "prob": round(t["prob"], 4) if t["prob"] is not None else None,
                            "side": "buy",
                            "reason": "新进组合 topN"})
    for p in MANUAL_BROKER.query_positions():
        if p.symbol not in target_set:
            actions.append({"symbol": p.symbol, "name": _display_name(p.symbol),
                            "prob": None, "side": "sell", "reason": "掉出组合 topN"})

    # 资金决策预览（基于现有现金与持仓 + 大盘弱势风控）；价格覆盖池外候选
    syms = {t["symbol"] for t in targets} | \
        {p.symbol for p in MANUAL_BROKER.query_positions()}
    prices = _build_prices(syms)
    allocation = _portfolio_allocation(targets, target_set, held, prices, market_weak)
    cash = MANUAL_BROKER.query_cash()
    # 总资产 = 现金 + 全部持仓市值（含不在目标内的持仓，它们会触发卖出）
    held_value = sum(
        p.shares * prices.get(p.symbol, p.avg_cost)
        for p in MANUAL_BROKER.query_positions())
    total_assets = cash + held_value

    return {
        "top_n": len(targets),
        "candidate_source": cfg.get("trading", {}).get("candidate_source", "selection"),
        "targets": [{
            "symbol": t["symbol"], "name": t["name"],
            "prob": round(t["prob"], 4) if t["prob"] is not None else None,
            "close": t["close"],
            "score": round(t["score"] if t.get("score_scale") == 100
                          else t["score"] * 100, 1),   # 统一展示为 0~100
            "in_universe": t.get("in_universe", t["symbol"] in DATA),
            "timing_action": (t["timing"] or {}).get("action"),
            "timing_score": round((t["timing"] or {}).get("score", 0.0), 3),
            "advisor_action": (t["advisor"] or {}).get("action"),
            "advisor_label": (t["advisor"] or {}).get("label"),
            "minute_prob": (t.get("minute") or {}).get("minute_prob"),
            "minute_signal": (t.get("minute") or {}).get("minute_signal"),
            "reasons": t.get("reasons", []),
        } for t in targets],
        "actions": actions,
        "allocation": allocation,
        "market_weak": market_weak,
        "cash": round(cash, 2),
        "total_assets": round(total_assets, 2),
        "regime": {"regime": regime.get("regime") if regime else None,
                   "text": regime.get("text") if regime else None},
        "trading_hours": _in_trading_hours(),
    }


@app.post("/api/portfolio/apply")
def portfolio_apply():
    """一键组合调仓：卖掉落出 topN 的，按资金决策买入/加仓新进与不足的 topN（整手）。"""
    if not _in_trading_hours():
        raise HTTPException(
            400, "非交易时段无法调仓（A股交易 9:30-11:30 / 13:00-15:00，周一至周五）")
    targets, target_set, held, _ = _portfolio_with_reasons()
    if not targets:
        raise HTTPException(500, "无组合目标信号")
    market_weak = _market_weakness()

    # 最新交易日与价格：实时快照优先 → 信号收盘 → 选股候选收盘（覆盖池外）
    all_syms = {t["symbol"] for t in targets} | \
        {p.symbol for p in MANUAL_BROKER.query_positions()}
    prices = _build_prices(all_syms)
    today = None
    for s in all_syms:
        sig = SIGNALS.get(s)
        if sig is not None and not sig.empty:
            today = today or str(sig.index[-1].date())
    if today is None:
        today = (SELECTION_RESULT or {}).get("date") or datetime.now().strftime("%Y-%m-%d")
    if not prices:
        raise HTTPException(500, "无行情数据")

    executed = []
    risk_notes = []
    # 0. 大盘弱势风控提示（不阻止卖出，但可能降仓买入）
    if market_weak.get("weak"):
        idx_pct = market_weak.get("index_pct")
        risk_notes.append(
            f"⚠ 大盘弱势（沪深300/上证 {idx_pct}%，阈值 {market_weak.get('threshold')}%），"
            f"买入仓位降至 "
            f"{cfg.get('portfolio_risk', {}).get('weak_position_pct', 0.5)*100:.0f}%")

    # 1. 卖掉落出 topN 的持仓
    for p in MANUAL_BROKER.query_positions():
        if p.symbol not in target_set and p.symbol in prices:
            r = MANUAL_BROKER.sell(p.symbol, p.shares, prices[p.symbol], today,
                                   remark="组合调仓·掉出topN")
            if r.success:
                executed.append({"symbol": p.symbol, "name": _display_name(p.symbol),
                                 "side": "sell", "price": round(r.price, 2),
                                 "shares": round(r.shares, 2)})
            else:
                executed.append({"symbol": p.symbol, "name": _display_name(p.symbol),
                                 "side": "sell", "error": r.message})

    # 2. 基于现有资金 + 风控决策买入/加仓（大盘弱势降仓 + 分钟否决 + 个股上限）
    allocation = _portfolio_allocation(targets, target_set, held, prices, market_weak)
    for a in allocation:
        if a["action"] == "skip":
            # 被风控暂停的买入也要透出到结果（前端可见"未买入+原因"，避免静默跳过）
            executed.append({"symbol": a["symbol"], "name": a["name"],
                             "side": "buy", "error": "⏸ " + a["reason"]})
            risk_notes.append(f"⏸ {a['symbol']} {a['name']}：{a['reason']}")
            continue
        if a["action"] not in ("buy", "add"):
            continue
        shares = a["est_shares"]
        if shares >= 100:
            r = MANUAL_BROKER.buy(a["symbol"], shares, prices[a["symbol"]], today,
                                  remark="组合调仓·新进topN" if a["action"] == "buy"
                                  else "组合调仓·加仓至目标")
            if r.success:
                executed.append({"symbol": a["symbol"], "name": a["name"],
                                 "side": "buy", "price": round(r.price, 2),
                                 "shares": shares, "amount": round(r.amount, 2)})
            else:
                executed.append({"symbol": a["symbol"], "name": a["name"],
                                 "side": "buy", "error": r.message})
        else:
            executed.append({"symbol": a["symbol"], "name": a["name"],
                             "side": "buy", "error": "资金不足一手(100股)"})

    # 快照净值
    latest = {s: float(sig["close"].iloc[-1]) for s, sig in SIGNALS.items() if not sig.empty}
    MANUAL_BROKER.snapshot_equity(today, latest)
    return {"executed": executed, "notes": risk_notes,
            "market_weak": market_weak,
            "account": MANUAL_BROKER.account_summary()}


# ---------- 每日选股（优质股推荐） ----------
def _run_daily_selection(force: bool = False):
    """执行每日选股：沪深300 → 流动性 → 基本面 + 技术面 → topN。更新全局并写盘。"""
    global SELECTION_RESULT
    try:
        from quant.data.selector import select_daily  # noqa: PLC0415
        sel_cfg = cfg.get("selection", {})
        rows = select_daily(n=sel_cfg.get("n", 12))
        now = datetime.now()
        SELECTION_RESULT = {
            "date": now.strftime("%Y-%m-%d"),
            "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            "candidates": rows,
        }
        path = cfg.resolve("results") / "daily_selection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(SELECTION_RESULT, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        logger.info("[selection] 每日选股完成 %d 只（%s）", len(rows), now.strftime("%m-%d %H:%M"))
    except Exception as exc:  # noqa: BLE001
        logger.error("[selection] 每日选股失败: %s", exc)


@app.get("/api/selection")
def selection():
    """读取最近一次每日选股结果。"""
    if SELECTION_RESULT and SELECTION_RESULT.get("candidates"):
        return SELECTION_RESULT
    return {"date": None, "candidates": [], "message": "尚未选股，可点击「立即选股」"}


@app.post("/api/selection/run")
def selection_run():
    """手动触发一次每日选股（后台运行，约 1-2 分钟）。"""
    if getattr(selection_run, "_busy", False):
        raise HTTPException(409, "选股进行中，请稍候")
    selection_run._busy = True

    def _worker():
        try:
            _run_daily_selection(force=True)
        finally:
            selection_run._busy = False

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "message": "选股已启动，约 1-2 分钟后完成（可稍后刷新查看）"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=cfg["api"]["host"], port=cfg["api"]["port"])
