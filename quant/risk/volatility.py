"""波动率相关：ATR 计算与「波动率 → 止盈止损百分比」换算。

把「ATR → 动态止损/止盈百分比」的换算集中在这里，
被 paper.py（撮合）、updater.py（自动盘）、api/main.py（手动盘/持仓展示）共用，
保证各处动态止盈止损口径一致。
"""
from __future__ import annotations

import pandas as pd

from ..features.technical import compute_technical


def latest_atr(bars: pd.DataFrame, window: int = 20) -> float:
    """计算 bars 最新的 ATR（价格单位）。bars 需含 high/low/close 列。"""
    if bars is None or len(bars) < 2:
        return 0.0
    feat = compute_technical(bars, ["atr"])
    series = feat["atr"].dropna()
    return float(series.iloc[-1]) if len(series) else 0.0


def build_vol_map(data: dict, window: int = 20) -> dict:
    """{symbol: 日线bars} → {symbol: {"atr","close","atr_pct"}}。

    atr_pct = ATR / 最新收盘价，即「日均波动幅度占比」，便于横截面比较。
    """
    out = {}
    for symbol, bars in data.items():
        if bars is None or len(bars) < 2:
            continue
        atr = latest_atr(bars, window)
        close = float(bars["close"].iloc[-1])
        out[symbol] = {
            "atr": atr,
            "close": close,
            "atr_pct": (atr / close) if close else 0.0,
        }
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def dynamic_pcts(vol: dict, cost: float, high: float | None = None,
                 stop_mult: float = 2.5, take_mult: float = 3.5,
                 trail_mult: float = 2.5, min_pct: float = 0.03,
                 max_pct: float = 0.15, take_min_pct: float = 0.05,
                 take_max_pct: float = 0.30) -> dict:
    """按 ATR 计算单只股票的动态止损/止盈/移动止损百分比。

    Args:
        vol: 单只股票的波动率信息 {"atr": float, ...}（build_vol_map 的输出项）
        cost: 持仓成本（止损/止盈的参考价）
        high: 持仓期间最高价（移动止损参考价；None 则退化为用 cost）
        stop_mult/take_mult/trail_mult: ATR 倍数（止损/止盈/移动止损）
        min_pct/max_pct: 止损百分比的上下限（防止极端）
        take_min_pct/take_max_pct: 止盈百分比的上下限

    Returns:
        {"stop_pct","take_pct","trail_pct","stop_price","take_price"}
        波动率无效时各值为 None。
    """
    atr = float(vol.get("atr", 0.0) or 0.0)
    ref = high if high and high > 0 else cost
    if atr <= 0 or cost <= 0:
        return {"stop_pct": None, "take_pct": None, "trail_pct": None,
                "stop_price": None, "take_price": None}
    stop_pct = _clamp(stop_mult * atr / cost, min_pct, max_pct)
    take_pct = _clamp(take_mult * atr / cost, take_min_pct, take_max_pct)
    if take_pct <= stop_pct:            # 保证止盈始终 > 止损
        take_pct = stop_pct * 1.5
    trail_pct = _clamp(trail_mult * atr / ref, min_pct, max_pct)
    return {
        "stop_pct": stop_pct,
        "take_pct": take_pct,
        "trail_pct": trail_pct,
        "stop_price": cost * (1 - stop_pct),
        "take_price": cost * (1 + take_pct),
    }


def vol_cfg_from_risk(risk: dict) -> dict:
    """从 config.yaml 的 risk 段提取动态止盈止损参数（供 apply_stop_rules 使用）。"""
    return {
        "atr_stop_mult": float(risk.get("atr_stop_mult", 2.5)),
        "atr_take_mult": float(risk.get("atr_take_mult", 3.5)),
        "atr_trailing_mult": float(risk.get("atr_trailing_mult", 2.5)),
        "vol_min_pct": float(risk.get("vol_min_pct", 0.03)),
        "vol_max_pct": float(risk.get("vol_max_pct", 0.15)),
        "take_min_pct": float(risk.get("take_min_pct", 0.05)),
        "take_max_pct": float(risk.get("take_max_pct", 0.30)),
        "trailing_enabled": bool(risk.get("trailing_stop", True)),
    }
