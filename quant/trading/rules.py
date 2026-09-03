"""A股交易规则辅助函数（涨跌停价计算等）。

与撮合层（paper.py 的 T+1/整手/印花税）分离，方便单测与复用。
"""
from __future__ import annotations


def limit_pct(symbol: str) -> float:
    """涨跌幅限制：创业板(30)/科创板(68) ±20%，主板(60/00/其他) ±10%。"""
    return 0.20 if symbol.startswith(("30", "68")) else 0.10


def limit_prices(prev_close: float, symbol: str) -> tuple[float, float]:
    """根据前收盘价和板块返回 (涨停价, 跌停价)。"""
    pct = limit_pct(symbol)
    return round(prev_close * (1 + pct), 2), round(prev_close * (1 - pct), 2)
