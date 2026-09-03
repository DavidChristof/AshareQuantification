"""数据获取模块：通过 akshare 拉取 A 股日线行情。

默认使用新浪源（stock_zh_a_daily），稳定且无需 token。
东方财富源（stock_zh_a_hist）作为自动回退。

统一输出列：symbol, date, open, high, low, close, volume, amount
其中价格均为前复权价格（qfq），保证收益率计算不受分红送股影响。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

# 新浪源返回的列名（英文，无需映射，直接选取）
_SINA_COLS = ["date", "open", "high", "low", "close", "volume", "amount"]

_DATE_FMT = "%Y%m%d"


def _to_ak_symbol(symbol: str) -> str:
    """6 位 A 股代码 → 带交易所前缀的代码（akshare 需要）。

    - 60xxxx / 68xxxx → sh（沪市主板 / 科创板）
    - 00xxxx / 30xxxx → sz（深市主板 / 创业板）
    - 其他（8xxxxx / 4xxxxx）→ bj（北交所）
    """
    if symbol.startswith(("sh", "sz", "bj")):
        return symbol
    if symbol.startswith(("60", "68")):
        return f"sh{symbol}"
    if symbol.startswith(("00", "30")):
        return f"sz{symbol}"
    return f"bj{symbol}"


def fetch_daily(symbol: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    """获取单只股票的日线数据。

    Args:
        symbol: 6 位 A 股代码，如 "600519"。
        start_date / end_date: "YYYY-MM-DD" 格式。
        adjust: 复权方式，qfq=前复权（默认），hfq=后复权，""=不复权。
    """
    import akshare as ak  # 延迟导入，避免 import 过慢

    start = datetime.strptime(start_date, "%Y-%m-%d").strftime(_DATE_FMT)
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime(_DATE_FMT)
    ak_symbol = _to_ak_symbol(symbol)

    raw = None
    last_err: Exception | None = None

    # 源 1：新浪（稳定）
    try:
        logger.info("正在获取 %s（新浪源）...", symbol)
        raw = ak.stock_zh_a_daily(
            symbol=ak_symbol, start_date=start, end_date=end, adjust=adjust
        )
    except Exception as exc:  # noqa: BLE001
        last_err = exc
        logger.warning("新浪源获取 %s 失败: %s", symbol, exc)

    # 源 2：东方财富（回退）
    if raw is None or raw.empty:
        try:
            logger.info("回退到东方财富源获取 %s ...", symbol)
            raw = ak.stock_zh_a_hist(
                symbol=symbol, period="daily",
                start_date=start, end_date=end, adjust=adjust,
            )
        except Exception as exc:  # noqa: BLE001
            if last_err:
                logger.error("两个数据源均失败: %s / %s", last_err, exc)
            else:
                logger.error("东方财富源获取 %s 失败: %s", symbol, exc)

    if raw is None or raw.empty:
        logger.warning("股票 %s 未获取到数据", symbol)
        return pd.DataFrame(columns=["symbol"] + _SINA_COLS)

    # 统一列名
    df = raw[_SINA_COLS].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df.insert(0, "symbol", symbol)
    return df


def fetch_universe(symbols: Iterable[str], start_date: str, end_date: str,
                   delay: float = 0.6) -> pd.DataFrame:
    """批量获取多只股票数据，拼接成一张长表。

    若配置的 end_date 早于今天，自动用今天作为最新日期——
    保证每天/每次拉取都能拿到最新数据，无需手动改 config。

    delay: 每次请求间的间隔秒数，避免触发数据源限流。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    if end_date < today:
        logger.info("配置 end_date=%s 已过时，自动使用今天 %s", end_date, today)
        end_date = today

    frames = []
    for symbol in symbols:
        try:
            df = fetch_daily(symbol, start_date, end_date)
            if not df.empty:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001 - 单只股票失败不应中断整个流程
            logger.error("获取 %s 失败: %s", symbol, exc)
        time.sleep(delay)

    if not frames:
        raise RuntimeError("所有标的均获取失败，请检查网络或 akshare 接口")
    return pd.concat(frames, ignore_index=True)
