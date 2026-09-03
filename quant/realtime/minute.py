"""分钟K线抓取（新浪分时接口）—— 阶段二：分钟级数据。

新浪接口：CN_MarketDataService.getKLineData
    symbol=sh600519, scale=5/15/30/60, datalen=条数上限

返回字段：datetime / open / high / low / close / volume / amount
"""
from __future__ import annotations

import logging

import pandas as pd
import requests

from .quoter import _prefix

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_KLINE_URL = ("https://quotes.sina.cn/cn/api/json_v2.php/"
              "CN_MarketDataService.getKLineData")

_COLS = ["datetime", "open", "high", "low", "close", "volume", "amount"]


def fetch_minute_bars(symbol: str, scale: int = 5, datalen: int = 1023) -> pd.DataFrame:
    """抓取单只股票的分钟K线（最新在前，接口按时间倒序返回）。

    Args:
        symbol: 6 位 A 股代码。
        scale: 分钟级别，5 / 15 / 30 / 60。
        datalen: 返回条数上限（新浪最大约 1023；5 分钟 ≈ 21 个交易日）。

    Returns:
        DataFrame：symbol / datetime / open / high / low / close / volume / amount
    """
    resp = requests.get(
        _KLINE_URL,
        params={"symbol": _prefix(symbol), "scale": scale, "ma": "no", "datalen": datalen},
        headers={"User-Agent": _UA, "Referer": "https://finance.sina.com.cn"},
        timeout=8,
    )
    resp.raise_for_status()
    data = resp.json()

    if not isinstance(data, list) or not data:
        return pd.DataFrame(columns=["symbol"] + _COLS)

    rows = [{
        "datetime": pd.to_datetime(it["day"]),
        "open": float(it["open"]),
        "high": float(it["high"]),
        "low": float(it["low"]),
        "close": float(it["close"]),
        "volume": float(it.get("volume", 0)),
        "amount": float(it.get("amount", 0)),
    } for it in data]
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    df.insert(0, "symbol", symbol)
    return df
