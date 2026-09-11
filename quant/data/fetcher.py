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


# 新浪源额外返回的两列（fetch_daily 会丢弃，fetch_daily_full 保留）
_EXTRA_COLS = ["outstanding_share", "turnover"]


def fetch_daily_full(symbol: str, start_date: str, end_date: str,
                     adjust: str = "qfq", retries: int = 2) -> pd.DataFrame:
    """日线 + **流通股本/换手率**（时点宇宙重建与换手率回测用）。

    与 `fetch_daily` 的区别（**不改动 fetch_daily 本身**，那是线上路径）：
      1. 多返回两列：`outstanding_share`（流通股本）与 `turnover`（换手率）。
         经逐股核验，这两列是**逐日时点值**（会随增发/解禁变化），**不是当前快照回填**
         ⇒ 历史回测**无未来函数**。turnover 恒等于 volume/outstanding_share。
      2. 默认 `adjust="qfq"`（与项目其它库一致）：**收益率必须用复权价**，否则分红除权
         那天会出现假跌。
      3. **复权不改变 volume/amount/turnover/outstanding_share**（已逐股实测 4/4 完全相等），
         所以一次请求即可同时满足"复权价算收益"与"不复权量算市值"两类需求。
      4. ⚠️ **市值不要用 `close × outstanding_share` 算**——`close` 是前复权价，
         用它算市值实测偏高 **3%~11.5%**（分红越多的票偏得越狠）。
         正确口径：**流通市值 = amount / turnover**（= 成交均价 × 流通股本，与复权无关）。

    只走新浪源（东财源的 `stock_zh_a_hist` 不带 outstanding_share）；失败重试后返回空表。

    Args:
        symbol: 6 位 A 股代码，如 "600519"。
        start_date / end_date: "YYYY-MM-DD" 格式。
        adjust: 复权方式，默认 "qfq"（前复权）。
        retries: 失败后的重试次数（新浪偶发 RemoteDisconnected）。
    """
    import akshare as ak  # 延迟导入，避免 import 过慢
    import time

    start = datetime.strptime(start_date, "%Y-%m-%d").strftime(_DATE_FMT)
    end = datetime.strptime(end_date, "%Y-%m-%d").strftime(_DATE_FMT)
    ak_symbol = _to_ak_symbol(symbol)

    raw = None
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            raw = ak.stock_zh_a_daily(symbol=ak_symbol, start_date=start,
                                      end_date=end, adjust=adjust)
            if raw is not None and not raw.empty:
                break
        except Exception as exc:  # noqa: BLE001 - 单只失败不该中断批量
            last_err = exc
            logger.debug("%s 第 %d 次获取失败: %s", symbol, attempt + 1, exc)
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))

    cols = ["symbol"] + _SINA_COLS + _EXTRA_COLS
    if raw is None or raw.empty or "turnover" not in raw.columns:
        if last_err is not None:
            logger.debug("%s 无换手率数据: %s", symbol, last_err)
        return pd.DataFrame(columns=cols)

    keep = [c for c in _SINA_COLS + _EXTRA_COLS if c in raw.columns]
    df = raw[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _EXTRA_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)
    df.insert(0, "symbol", symbol)
    for col in _EXTRA_COLS:                  # 缺列时补齐，保证列集合稳定
        if col not in df.columns:
            df[col] = pd.NA
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
