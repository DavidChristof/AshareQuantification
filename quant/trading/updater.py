"""自动更新工具：数据刷新 + 信号重建 + 自动调仓。

供 API 后台定时任务复用，也便于在脚本中调用。
把 05_update.py 的核心逻辑抽出来，避免多处重复。
"""
from __future__ import annotations

import logging

from ..config import Config
from ..data.fetcher import fetch_universe
from ..data.loader import load_all
from ..data.storage import MarketDB
from ..models.predict import ModelPredictor
from .engine import TradingEngine

logger = logging.getLogger(__name__)


def refresh_market_data(cfg: Config) -> int:
    """拉取最新行情并 upsert 到 SQLite，返回写入行数。"""
    data_cfg = cfg["data"]
    df = fetch_universe(data_cfg["universe"], data_cfg["start_date"], data_cfg["end_date"])
    db = MarketDB(cfg.resolve(data_cfg["db_path"]))
    n = db.save_bars(df)
    logger.info("行情已更新: %d 行", n)
    return n


def rebuild_signals(cfg: Config, predictor: ModelPredictor, data=None):
    """（重新）加载数据并生成每只股票的信号表。

    Returns:
        (data, signals)
        data:    {symbol: DataFrame}
        signals: {symbol: DataFrame(date/close/prob_up/signal)}
    """
    if data is None:
        data = load_all(cfg)
    threshold = cfg["backtest"]["threshold"]
    # v2 横截面模型：一次喂全池批量预测；旧模型：逐股预测
    if hasattr(predictor, "make_signals_all"):
        signals = predictor.make_signals_all(data)
    else:
        signals = {
            symbol: predictor.make_signal(bars, threshold=threshold)
            for symbol, bars in data.items()
        }
    return data, signals


def rebalance_auto(cfg: Config, broker, signals: dict, date: str, data: dict | None = None
                ) -> dict:
    """对自动纸面盘执行一次调仓（止盈止损 → 目标持仓/买卖动作/净值快照）。

    data: {symbol: 日线bars}，可选；提供时按 ATR 动态止盈止损。
    """
    bt = cfg["backtest"]
    max_positions = cfg.get("trading", {}).get("max_positions", 3)
    prices = {s: float(t["close"].iloc[-1]) for s, t in signals.items() if not t.empty}

    # 1. 先执行止盈止损（止损/止盈/移动止损，支持按波动率动态）
    risk = cfg.get("risk", {})
    if risk.get("enabled", True):
        from ..risk.volatility import build_vol_map, vol_cfg_from_risk

        vol = vol_cfg = None
        if risk.get("dynamic_volatility", True) and data:
            vol = build_vol_map(data, risk.get("vol_window", 20))
            vol_cfg = vol_cfg_from_risk(risk)
        stops = broker.apply_stop_rules(
            date, prices,
            stop_loss_pct=risk.get("stop_loss_pct", 0.08),
            take_profit_pct=risk.get("take_profit_pct", 0.15),
            trailing_pct=risk.get("trailing_pct")
            if risk.get("trailing_stop", False) else None,
            vol=vol, vol_cfg=vol_cfg,
        )
        for s in stops:
            logger.info("自动盘止盈止损触发 %s: %s", s["symbol"], s["reason"])

    # 2. 调仓
    engine = TradingEngine(
        broker,
        threshold=bt["threshold"],
        position_pct=bt["position_pct"],
        max_positions=max_positions,
    )
    probs = {s: float(t["prob_up"].iloc[-1]) for s, t in signals.items() if not t.empty}
    return engine.rebalance(date, probs, prices)


def latest_trade_date(data: dict) -> str:
    """所有股票中最新的交易日。"""
    dates = [bars.iloc[-1]["date"] for bars in data.values() if len(bars)]
    if not dates:
        return "2020-01-01"
    return str(max(dates).date())
