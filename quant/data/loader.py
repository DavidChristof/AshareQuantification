"""加载与预处理模块：从数据库读取并整理成 DataFrame。"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import Config
from .storage import MarketDB

logger = logging.getLogger(__name__)


def load_all(cfg: Config) -> dict[str, pd.DataFrame]:
    """加载股票池中所有标的的日线，返回 {symbol: DataFrame}。

    每份 DataFrame 列：date, open, high, low, close, volume, amount
    """
    db = MarketDB(cfg["data"]["db_path"])
    symbols = db.list_symbols()
    if not symbols:
        raise RuntimeError("数据库为空，请先运行 scripts/01_fetch_data.py 下载数据")

    data = {s: db.load_symbol(s) for s in symbols}
    logger.info("已加载 %d 只股票", len(data))
    return data


def align_panel(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """把多只股票的收盘价转成宽表（每列一只股票），并丢弃缺失交易日。"""
    closes = {s: df.set_index("date")["close"] for s, df in data.items()}
    panel = pd.DataFrame(closes).sort_index()
    return panel.dropna(how="any")
