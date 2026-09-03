"""优质选股器：分层筛选流动性好、基本面优质的股票。

选股逻辑（可解释、可复现）：
    第 1 层 硬过滤    ：排除 ST / *ST（退市风险）
    第 2 层 流动性    ：日成交额 >= min_amount（实时快照批量获取，保证可交易）
    第 3 层 基本面打分：低市盈率(PE) + 高净资产收益率(ROE) → 综合质量分
    第 4 层 取前 N    只，写入股票池

说明：
    - 流动性用「实时快照」批量一次拿全（新浪/腾讯），快速高效
    - 基本面（PE/ROE）逐只抓取，较慢（一次性操作可接受）
    - 抓取失败/数据缺失的股票自动跳过，用流动性补足

每日选股（select_daily）：
    在基本面基础上叠加「技术面」（动量/趋势/波动），用日线并发计算，
    适合每天收盘后自动跑：寻找当前基本面好且技术面走强的沪深300优质股。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..realtime.quoter import fetch_quotes
from .fetcher import fetch_daily
from .universe import fetch_hs300

logger = logging.getLogger(__name__)

# 因子缓存：供「动态选股权重」复用（避免每次重复抓基本面）
FACTOR_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "factor_cache.json"
# 大池日线缓存：22_scan 下载的中证500/1000 等，供「今日选股」扩展候选宇宙
LARGE_CACHE_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "large_scan_cache.pkl"

MIN_AMOUNT = 1e8                # 日成交额下限：1 亿元
MIN_PE = 1.0                    # 合理 PE 下限（排除异常低值）
PE_BASE = 30.0                  # PE 打分基准（PE=30 → 满分档）
ROE_BASE = 15.0                 # ROE 打分基准（ROE=15% → 满分档）
_REQUEST_GAP = 0.3              # 基本面请求间隔，控制频率

# 每日选股参数
BASIC_TOPK = 80                 # 抓基本面的候选数（流动性高者优先）
TECH_TOPK = 40                  # 算技术面的候选数（基本面分高者优先）
TECH_WORKERS = 4                # 日线拉取并发数
CACHE_MAX_AGE_HOURS = 24        # 基本面缓存有效期（小时）

# 技术面因子权重（mom/trd/vol/rev），regime 门控可切换（见 fetch_market_regime）
# 实证（29_regime_split_validation，600池 h20）：价量 alpha 是「条件性」的——
#   下跌市 低波动+60日反转 最强（RankIC 0.118 / 超额 +1.4%）→ 防守/超跌反弹
#   上涨市 低波/反转 负暴露（RankIC -0.072 / 超额 -1.8%）→ 动量/趋势主导
DEFAULT_TECH_WEIGHTS = {"mom": 25, "trd": 15, "vol": 30, "rev": 30}
TECH_REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "uptrend":   {"mom": 45, "trd": 25, "vol": 15, "rev": 15},  # 上涨趋势：追涨
    "range":     {"mom": 20, "trd": 10, "vol": 35, "rev": 35},  # 震荡：均值回归
    "downtrend": {"mom": 10, "trd": 10, "vol": 35, "rev": 45},  # 下跌：防守/超跌反弹
}
_REGIME_CN = {"uptrend": "上涨趋势", "range": "震荡整理", "downtrend": "下跌趋势"}


def pick_tech_weights(regime: str | None) -> dict[str, float]:
    """regime → 技术面权重（未知/数据不足回退默认），各分量和为 100。"""
    w = dict(TECH_REGIME_WEIGHTS.get(regime or "", DEFAULT_TECH_WEIGHTS))
    total = float(sum(w.values())) or 1.0
    return {k: round(v / total * 100, 1) for k, v in w.items()}



# ---------- 数据获取 ----------
def fetch_liquidity(codes: Iterable[str], batch: int = 80) -> dict[str, float]:
    """分批获取实时成交额（避免单次请求代码过多）。返回 {code: amount}。"""
    code_list = list(codes)
    result: dict[str, float] = {}
    for i in range(0, len(code_list), batch):
        chunk = code_list[i:i + batch]
        quotes = fetch_quotes(chunk)
        result.update({c: q["amount"] for c, q in quotes.items()})
        time.sleep(_REQUEST_GAP)
    logger.info("流动性获取完成: %d/%d 只", len(result), len(code_list))
    return result


def _fetch_pe(code: str) -> float | None:
    """最新市盈率 TTM（百度估值），负值（亏损）返回 None。"""
    import akshare as ak
    df = ak.stock_zh_valuation_baidu(
        symbol=code, indicator="市盈率(TTM)", period="近一年")
    if df is None or df.empty:
        return None
    v = float(df["value"].iloc[-1])
    return v if v >= MIN_PE else None


def _fetch_roe(code: str) -> float | None:
    """最新净资产收益率 %（新浪财务指标），动态定位列名。"""
    import akshare as ak
    df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2024")
    if df is None or df.empty:
        return None
    col = next((c for c in df.columns if "净资产收益率" in str(c)), None)
    if not col:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.iloc[-1]) if len(vals) else None


def fetch_fundamental(code: str) -> dict | None:
    """抓取单只股票基本面 {pe, roe}，失败返回 None。"""
    try:
        pe = _fetch_pe(code)
        time.sleep(_REQUEST_GAP)
        roe = _fetch_roe(code)
        time.sleep(_REQUEST_GAP)
        if pe is None or roe is None:
            return None
        return {"pe": pe, "roe": roe}
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s 基本面抓取失败: %s", code, exc)
        return None


# ---------- 打分 ----------
def score_stock(fund: dict | None) -> float | None:
    """综合质量分（0~100）：低 PE + 高 ROE。

    PE 分：PE=30 → 50 分，PE 越高越低（越便宜越好）
    ROE 分：ROE=15% → 50 分，越高越好（越能赚钱越好）
    """
    if not fund:
        return None
    pe_score = max(0.0, min(50.0, 50.0 * (PE_BASE / max(fund["pe"], 1.0))))
    roe_score = max(0.0, min(50.0, 50.0 * (fund["roe"] / ROE_BASE)))
    return round(pe_score + roe_score, 1)


# ---------- 主流程 ----------
def select_universe(n: int = 40, min_amount: float = MIN_AMOUNT,
                    seed: int = 42) -> list[dict]:
    """分层选股，返回 [{code, name}]，按代码排序。"""
    stocks = fetch_hs300()
    names = {s["code"]: s["name"] for s in stocks}
    codes = [s["code"] for s in stocks]

    # ---- 第 1+2 层：硬过滤 + 流动性 ----
    liq = fetch_liquidity(codes)
    cand: list[str] = []
    for code in codes:
        if "ST" in names.get(code, "").upper():      # 排除 ST
            continue
        if liq.get(code, 0) < min_amount:            # 排除低流动性
            continue
        cand.append(code)
    logger.info("硬过滤+流动性筛选后: %d / %d 只", len(cand), len(codes))
    if not cand:
        raise RuntimeError("筛选后候选池为空，请降低 min_amount")

    # ---- 第 3 层：基本面打分 ----
    scored: list[tuple[str, float]] = []
    factor_cache: dict[str, dict] = {}
    for code in cand:
        fund = fetch_fundamental(code)
        s = score_stock(fund)
        if s is not None:
            scored.append((code, s))
            factor_cache[code] = {"pe": fund["pe"], "roe": fund["roe"], "score": s}
    logger.info("基本面打分完成: %d 只（其余数据缺失跳过）", len(scored))

    # 缓存因子数据（供动态选股权重用）
    try:
        FACTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        import json
        FACTOR_CACHE_PATH.write_text(json.dumps(factor_cache), encoding="utf-8")
        logger.info("因子缓存已写入 %s（%d 只）", FACTOR_CACHE_PATH, len(factor_cache))
    except OSError as exc:  # noqa: BLE001
        logger.warning("因子缓存写入失败: %s", exc)

    # 基本面分高的优先，不足 N 只时用流动性补足
    scored.sort(key=lambda x: -x[1])
    top = [c for c, _ in scored[:n]]
    if len(top) < n:
        have = set(top)
        # 按成交额从高到低补足（剔除已有的）
        by_liq = sorted((c for c in cand if c not in have), key=lambda c: -liq.get(c, 0))
        top.extend(by_liq[: n - len(top)])
    logger.info("最终选股 %d 只", len(top))

    result = [{"code": c, "name": names[c]} for c in top]
    result.sort(key=lambda s: s["code"])
    return result


def _load_large_codes() -> list[str]:
    """读取大池日线缓存里的股票代码（中小盘，来自 22_scan）；无缓存返回空。"""
    try:
        import pickle
        with open(LARGE_CACHE_PATH, "rb") as f:
            return sorted(pickle.load(f).get("data", {}).keys())
    except Exception:  # noqa: BLE001 - 无缓存时回退 hs300
        return []


def _quotes_names_liq(codes: list[str]) -> tuple[dict, dict]:
    """分批实时快照 → (names, liq)，供大池候选使用（含 name 可过滤 ST）。"""
    names, liq = {}, {}
    for i in range(0, len(codes), 80):
        chunk = codes[i:i + 80]
        try:
            q = fetch_quotes(chunk)
            for c, v in q.items():
                names[c] = v.get("name", c)
                liq[c] = v.get("amount", 0.0)
        except Exception as exc:  # noqa: BLE001
            logger.warning("实时快照批次失败(%d~%d): %s", i, i + len(chunk), exc)
        time.sleep(_REQUEST_GAP)
    return names, liq


# ==========================================================================
# 每日选股（寻找大盘优质股，叠加技术面）
# ==========================================================================
def _load_fundamental_cache(max_age_hours: float = CACHE_MAX_AGE_HOURS) -> dict:
    """读取基本面缓存 {code: {pe, roe, score, fetched_at}}；过期/异常返回空。"""
    if not FACTOR_CACHE_PATH.exists():
        return {}
    try:
        import json
        data = json.loads(FACTOR_CACHE_PATH.read_text(encoding="utf-8"))
        now = time.time()
        out = {}
        for code, v in data.items():
            ts = v.get("_fetched_at", 0)
            if now - ts <= max_age_hours * 3600 and v.get("pe") and v.get("roe"):
                out[code] = {"pe": v["pe"], "roe": v["roe"],
                             "score": v.get("score"), "fetched_at": ts}
        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("基本面缓存读取失败: %s", exc)
        return {}


def _save_fundamental_cache(fund_map: dict):
    """写回基本面缓存（含时间戳，供每日选股/动态权重复用）。"""
    try:
        import json
        FACTOR_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = {c: {**v, "_fetched_at": v.get("fetched_at", time.time())}
                for c, v in fund_map.items()}
        FACTOR_CACHE_PATH.write_text(json.dumps(data), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001
        logger.warning("基本面缓存写入失败: %s", exc)


def _fetch_fundamental_into(codes: list[str], fund_map: dict) -> None:
    """对缓存缺失的基本面逐只抓取并写入 fund_map。"""
    missing = [c for c in codes if c not in fund_map]
    logger.info("抓取基本面 %d 只（缓存缺失）...", len(missing))
    for i, code in enumerate(missing):
        fund = fetch_fundamental(code)
        if fund:
            fund_map[code] = {**fund, "score": score_stock(fund), "fetched_at": time.time()}
        if i % 20 == 0:
            logger.info("  基本面进度 %d/%d", i + 1, len(missing))


def _daily_tech(code: str, days: int = 120) -> dict | None:
    """拉单只日线算技术面（动量/趋势/低波动/中期反转）。失败返回 None。"""
    try:
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        df = fetch_daily(code, start, end)
        if df is None or len(df) < 25:
            return None
        close = df["close"]
        mom20 = close.iloc[-1] / close.iloc[-21] - 1 if len(close) >= 21 else 0.0
        ma20 = close.rolling(20).mean().iloc[-1]
        trend = close.iloc[-1] / ma20 - 1 if ma20 and ma20 > 0 else 0.0
        vol20 = float(df["close"].pct_change().rolling(20).std().iloc[-1]) or 0.0
        # 中期反转：过去 60 天累计跌幅（>0 表示跌得多 → 未来倾向跑赢，实证 IC≈0.045）
        rev60 = 0.0
        if len(close) >= 61:
            rev60 = float(close.iloc[-61] / close.iloc[-1] - 1)
        return {"mom20": float(mom20), "trend": float(trend),
                "vol20": vol20, "rev60": rev60,
                "close": float(close.iloc[-1])}
    except Exception as exc:  # noqa: BLE001
        logger.debug("%s 技术面失败: %s", code, exc)
        return None


def _score_tech(t: dict | None,
                weights: dict | None = None) -> float | None:
    """技术面分（0~100）：(动量+趋势+低波动+中期反转) × 权重。

    默认权重 动量25/趋势15/低波动30/反转30（600 只实证：低波 vol20≈-0.05、
    60日反转 rev60≈+0.045 最强）；regime 门控时传入 pick_tech_weights(regime)
    的权重，实现「弱市重防守/强市重追涨」。
    """
    if not t:
        return None
    w = weights or DEFAULT_TECH_WEIGHTS
    mom = max(0.0, min(1.0, (t["mom20"] + 0.05) / 0.25))       # -5%~20% 动量
    trd = max(0.0, min(1.0, (t["trend"] + 0.05) / 0.10))        # -5%~5% 偏离MA
    vol = max(0.0, min(1.0, 1.0 - t["vol20"] / 0.03))           # 日波动<3% 满分
    rev = max(0.0, min(1.0, (t.get("rev60", 0.0) + 0.20) / 0.30))  # 过去60天跌→反转加分
    return round(mom * w["mom"] + trd * w["trd"] + vol * w["vol"] + rev * w["rev"], 1)


def fetch_market_regime() -> dict | None:
    """市场状态判定（regime 门控代理）：用沪深300 日 K 线趋势 → 上涨/震荡/下跌。

    实证（29_regime_split_validation）：价量因子 alpha 是「条件性」的——弱市低波/反转
    是真 alpha、强市动量/趋势才有效。故每日选股用**可观察代理**（沪深300 近 20 日趋势，
    非未来收益）切换技术面权重。失败/数据不足返回 None → 调用方回退默认权重。
    """
    try:
        from ..realtime.indices import fetch_index_daily  # noqa: PLC0415
        from ..timing.regime import MarketRegime          # noqa: PLC0415
        df = fetch_index_daily("sh000300")
        close = df.set_index("date")["close"].sort_index()
        r = MarketRegime().detect(close)
        r["source"] = "sh000300 沪深300(日线趋势)"
        return r
    except Exception as exc:  # noqa: BLE001 - 判定失败不阻断选股
        logger.debug("市场状态判定失败(回退默认权重): %s", exc)
        return None


def select_daily(n: int = 12, basic_topk: int = BASIC_TOPK,
                 tech_topk: int = TECH_TOPK, min_amount: float = MIN_AMOUNT,
                 workers: int = TECH_WORKERS,
                 universe: str | None = None,
                 regime_gating: bool | None = None) -> list[dict]:
    """每日选股：流动性 + 基本面(缓存优先) + 技术面 → 综合分排名。

    流程：
        1. 候选宇宙（config selection.universe）：
           - large（默认）：现池 40 ∪ 中小盘缓存 ~560（600 只，实证低波动/反转所在）
           - hs300：仅沪深300（旧行为）
           实时流动性过滤（排除 ST）
        2. 流动性 top basic_topk → 基本面 PE/ROE（缓存缺失增量抓取）
        3. 基本面 top tech_topk → 并发拉日线算技术面
           （动量/趋势/低波动/60日中期反转；regime_gating=true 时按市场状态
           沪深300 趋势切换权重：上涨→动量主导 / 震荡→低波+反转 / 下跌→反转+低波防守）
        4. 综合分 = 基本面×0.6 + 技术面×0.4 → top n

    regime_gating: None = 读 config selection.regime_gating（默认 true）；
                   false 恒用默认权重（向后兼容旧行为）。

    Returns:
        [{code, name, pe, roe, fund_score, mom20, trend, vol20, rev60,
          tech_score, total_score, in_universe, regime, tech_weights}]
    """
    # ---- regime 门控开关（None → 读 config，默认开启） ----
    if regime_gating is None:
        try:
            from ..config import load_config
            regime_gating = bool((load_config().get("selection") or {}).get(
                "regime_gating", True))
        except Exception:  # noqa: BLE001
            regime_gating = True

    # ---- 候选宇宙 ----
    if universe is None:
        try:
            from ..config import load_config
            universe = (load_config().get("selection") or {}).get("universe", "large")
        except Exception:  # noqa: BLE001
            universe = "large"

    if universe == "large":
        base_codes = _load_large_codes()
        own: set[str] = set()
        try:
            from ..config import load_config
            own = set((load_config().get("data") or {}).get("universe", []))
        except Exception:  # noqa: BLE001
            pass
        codes = sorted(set(base_codes) | set(own))
        if codes:
            names, liq = _quotes_names_liq(codes)
            logger.info("大池候选: %d 只（现池 %d + 中小盘缓存 %d）",
                        len(codes), len(own), len(base_codes))
        else:                       # 无大池缓存 → 自动回退沪深300
            stocks = fetch_hs300()
            names = {s["code"]: s["name"] for s in stocks}
            codes = [s["code"] for s in stocks]
            liq = fetch_liquidity(codes)
    else:
        stocks = fetch_hs300()
        names = {s["code"]: s["name"] for s in stocks}
        codes = [s["code"] for s in stocks]
        liq = fetch_liquidity(codes)

    # 1. 硬过滤 + 流动性
    cand = [c for c in codes
            if "ST" not in names.get(c, "").upper() and liq.get(c, 0) >= min_amount]
    logger.info("硬过滤+流动性后候选: %d / %d", len(cand), len(codes))
    if not cand:
        raise RuntimeError("候选池为空，请降低 min_amount")

    # 2. 基本面（流动性高者优先抓，缓存复用）
    fund_map = _load_fundamental_cache()
    liq_rank = sorted(cand, key=lambda c: -liq.get(c, 0))[:basic_topk]
    _fetch_fundamental_into(liq_rank, fund_map)
    _save_fundamental_cache(fund_map)

    funded = [c for c in liq_rank if c in fund_map]
    funded.sort(key=lambda c: -(fund_map[c]["score"] or 0))
    logger.info("基本面打分完成: %d 只", len(funded))

    # 3. 技术面（基本面 top tech_topk 并发拉日线）
    tech_top = funded[:tech_topk]
    tech_map: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for code, t in zip(tech_top, ex.map(_daily_tech, tech_top)):
            if t:
                tech_map[code] = t
    logger.info("技术面计算完成: %d 只", len(tech_map))

    # 4. 综合打分（regime 门控：先判市场状态 → 选技术面因子权重）
    regime_code, weights = None, DEFAULT_TECH_WEIGHTS
    if regime_gating:
        rinfo = fetch_market_regime()
        if rinfo:
            regime_code = rinfo.get("regime")
            weights = pick_tech_weights(regime_code)
        logger.info("市场状态: %s 技术面权重=%s",
                    _REGIME_CN.get(regime_code, "未知(默认)"), weights)
    in_universe = set(load_universe_codes())
    rows = []
    for c in funded:
        fund = fund_map[c]
        ts = tech_map.get(c)
        fund_score = fund["score"] or 0.0
        tech_score = _score_tech(ts, weights)
        if tech_score is None:                 # 无技术面则只靠基本面
            total = fund_score
        else:
            total = fund_score * 0.6 + tech_score * 0.4
        rows.append({
            "code": c, "name": names.get(c, c),
            "pe": round(fund["pe"], 2), "roe": round(fund["roe"], 2),
            "fund_score": fund_score,
            "price": round(ts["close"], 2) if ts else None,   # 最新收盘（组合买卖成交价用）
            "mom20": round(ts["mom20"], 4) if ts else None,
            "trend": round(ts["trend"], 4) if ts else None,
            "vol20": round(ts["vol20"], 4) if ts else None,
            "rev60": round(ts["rev60"], 4) if ts else None,
            "tech_score": tech_score,
            "total_score": round(total, 1),
            "in_universe": c in in_universe,
            "regime": regime_code,
            "tech_weights": dict(weights),
        })
    rows.sort(key=lambda r: -r["total_score"])
    return rows[:n]


def load_universe_codes() -> set[str]:
    """读取当前股票池代码集合（供每日选股标注 in_universe）。"""
    try:
        from ..config import load_config
        cfg = load_config()
        return set(cfg.get("data", {}).get("universe", []))
    except Exception:  # noqa: BLE001
        return set()
