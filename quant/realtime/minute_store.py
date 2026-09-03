"""分钟K线存储：SQLite 增量 upsert，独立于日线库。

表结构 minute_bars(symbol, scale, datetime, open, high, low, close, volume, amount)
主键 (symbol, scale, datetime)，幂等写入，重复抓取不会产生脏数据。
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS minute_bars (
    symbol   TEXT NOT NULL,
    scale    INTEGER NOT NULL,
    datetime TEXT NOT NULL,
    open   REAL, high REAL, low REAL, close REAL,
    volume REAL, amount REAL,
    PRIMARY KEY (symbol, scale, datetime)
);
CREATE INDEX IF NOT EXISTS idx_minute_symbol
    ON minute_bars (symbol, scale, datetime);
"""

# SQLite 单文件不支持多线程并发写，用一把写入锁把写操作串行化
# （抓取可以并发，写库串行，配合批量插入足够快）
_save_lock = threading.Lock()


class MinuteStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")   # 提升并发读性能

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path), timeout=30)

    def save_bars(self, df: pd.DataFrame, scale: int) -> int:
        """批量 upsert 分钟K线（串行写 + executemany），返回写入行数。"""
        if df.empty:
            return 0
        data = df[["symbol", "datetime"]].copy()
        data["scale"] = scale
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            data[col] = df[col].astype(float)
        data["datetime"] = data["datetime"].astype(str)

        cols = ["symbol", "scale", "datetime", "open", "high", "low", "close", "volume", "amount"]
        rows = [tuple(r) for r in data[cols].to_numpy()]

        with _save_lock:   # 串行化写，避免 SQLite database is locked
            with self._connect() as conn:
                conn.execute("BEGIN")
                conn.executemany(
                    "INSERT OR REPLACE INTO minute_bars "
                    "(symbol, scale, datetime, open, high, low, close, volume, amount) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
        logger.debug("分钟K线写入 %d 行 (scale=%d)", len(data), scale)
        return len(data)

    def load_symbol(self, symbol: str, scale: int = 5, days: int = 1) -> pd.DataFrame:
        """读取某只股票最近 days 天的分钟K线（升序）。"""
        query = (
            "SELECT datetime, open, high, low, close, volume, amount "
            "FROM minute_bars WHERE symbol = ? AND scale = ? "
            "AND date(datetime) >= date('now', ?) ORDER BY datetime"
        )
        df = pd.read_sql_query(query, self._connect(), params=(symbol, scale, f"-{days} days"))
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df

    def latest_datetime(self, symbol: str, scale: int) -> str | None:
        """该股票该级别最近一根K线的时间。"""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(datetime) FROM minute_bars WHERE symbol=? AND scale=?",
                (symbol, scale)).fetchone()
        return row[0] if row else None

    def stats(self, symbol: str | None = None, scale: int | None = None) -> dict:
        sql = "SELECT COUNT(*), COUNT(DISTINCT symbol), MAX(datetime) FROM minute_bars"
        params: tuple = ()
        conds = []
        if symbol:
            conds.append("symbol=?")
            params += (symbol,)
        if scale:
            conds.append("scale=?")
            params += (scale,)
        if conds:
            sql += " WHERE " + " AND ".join(conds)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return {"rows": row[0], "symbols": row[1], "latest": row[2]}
