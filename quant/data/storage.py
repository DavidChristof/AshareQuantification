"""存储模块：把行情数据写入 SQLite。

表结构（长表，一行 = 某股票某天的一根 K 线）：
    daily_bars(symbol, date, open, high, low, close, volume, amount)
    PRIMARY KEY (symbol, date)
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_bars (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    amount REAL,
    PRIMARY KEY (symbol, date)
);
"""


class MarketDB:
    """轻量封装 SQLite 行情库。"""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def save_bars(self, df: pd.DataFrame, replace: bool = True) -> int:
        """写入 K 线数据（默认覆盖式 upsert）。

        Returns: 实际写入的行数。
        """
        if df.empty:
            return 0
        cols = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
        data = df[cols].copy()
        data["date"] = data["date"].astype(str)

        with self._connect() as conn:
            conn.execute("BEGIN")
            for _, row in data.iterrows():
                if replace:
                    conn.execute(
                        "INSERT OR REPLACE INTO daily_bars "
                        "(symbol, date, open, high, low, close, volume, amount) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        tuple(row),
                    )
                else:
                    conn.execute(
                        "INSERT OR IGNORE INTO daily_bars "
                        "(symbol, date, open, high, low, close, volume, amount) "
                        "VALUES (?,?,?,?,?,?,?,?)",
                        tuple(row),
                    )
            conn.commit()
        logger.info("写入 %d 行到 %s", len(data), self.db_path)
        return len(data)

    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """读取单只股票，按日期升序。"""
        query = (
            "SELECT date, open, high, low, close, volume, amount FROM daily_bars "
            "WHERE symbol = ? ORDER BY date"
        )
        df = pd.read_sql_query(query, self._connect(), params=(symbol,))
        df["date"] = pd.to_datetime(df["date"])
        return df

    def list_symbols(self) -> list[str]:
        query = "SELECT DISTINCT symbol FROM daily_bars ORDER BY symbol"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        return [r[0] for r in rows]

    def latest_date(self) -> str | None:
        """库内全部股票的最新日期（YYYY-MM-DD 文本）；空库返回 None。

        用于判断「行情是否推进到新交易日」（区分休市/节假日空跑）。
        """
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(date) FROM daily_bars").fetchone()
        return row[0] if row and row[0] else None

    def stats(self) -> dict:
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM daily_bars").fetchone()[0]
            symbols = conn.execute("SELECT COUNT(DISTINCT symbol) FROM daily_bars").fetchone()[0]
            span = conn.execute("SELECT MIN(date), MAX(date) FROM daily_bars").fetchone()
        return {"rows": total, "symbols": symbols, "date_range": span}
