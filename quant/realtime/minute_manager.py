"""MinuteManager：分钟K线后台增量更新。

- 启动时先全量拉一次历史（可回放约 20 个交易日）
- 交易时段（工作日 9:30-11:30 / 13:00-15:00）定时增量刷新
- 非交易时段不打扰数据源，盘后保持最后一根收盘K线
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterable

from .minute import fetch_minute_bars
from .minute_store import MinuteStore

logger = logging.getLogger(__name__)


class MinuteManager:
    def __init__(self, store: MinuteStore, symbols: Iterable[str],
                 scale: int = 5, interval: int = 60, datalen: int = 1023,
                 concurrency: int = 4):
        self.store = store
        self.symbols = list(symbols)
        self.scale = scale
        self.interval = max(interval, 30)      # 下限 30 秒
        self.datalen = datalen
        self.concurrency = max(concurrency, 1)
        self._started = False
        self._stop = threading.Event()
        self._last_refresh: datetime | None = None

    # ---------- 生命周期 ----------
    def start(self):
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._loop, daemon=True, name="minute-updater").start()
        logger.info("分钟K线更新已启动：scale=%d, 间隔 %d 秒, 标的 %s",
                    self.scale, self.interval, self.symbols)

    def stop(self):
        self._stop.set()

    # ---------- 抓取 ----------
    def refresh_symbol(self, symbol: str) -> int:
        """抓取并入库单只股票的分钟K线，返回写入行数。"""
        df = fetch_minute_bars(symbol, scale=self.scale, datalen=self.datalen)
        return self.store.save_bars(df, self.scale)

    def refresh_all(self) -> dict[str, int]:
        """并发刷新全部股票，返回 {symbol: 写入行数}。"""
        result: dict[str, int] = {}

        def _one(symbol: str) -> tuple[str, int]:
            try:
                return symbol, self.refresh_symbol(symbol)
            except Exception as exc:  # noqa: BLE001 - 单只失败不影响其他
                logger.warning("分钟K线 %s 刷新失败: %s", symbol, exc)
                return symbol, 0

        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            for symbol, n in ex.map(_one, self.symbols):
                result[symbol] = n
        self._last_refresh = datetime.now()
        return result

    # ---------- 调度 ----------
    def _is_trading_time(self) -> bool:
        now = datetime.now()
        if now.weekday() >= 5:                # 周末
            return False
        hm = now.hour * 100 + now.minute
        return (930 <= hm <= 1130) or (1300 <= hm <= 1500)

    def _loop(self):
        # 启动先全量拉一次（含历史），便于立即回放
        try:
            self.refresh_all()
            logger.info("分钟K线初始化完成，最新到 %s",
                        self.store.stats(self.symbols[0], self.scale)["latest"] if self.symbols else "-")
        except Exception as exc:  # noqa: BLE001
            logger.error("分钟K线初始化失败: %s", exc)

        while not self._stop.is_set():
            self._stop.wait(self.interval)
            if self._is_trading_time():
                self.refresh_all()

    # ---------- 查询 ----------
    def bars(self, symbol: str, days: int = 1) -> dict:
        """读取最近 days 天的分钟K线（datetime 转字符串便于 JSON 序列化）。"""
        df = self.store.load_symbol(symbol, scale=self.scale, days=days)
        records = []
        for _, r in df.iterrows():
            records.append({
                "datetime": r["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
                "open": round(float(r["open"]), 3),
                "high": round(float(r["high"]), 3),
                "low": round(float(r["low"]), 3),
                "close": round(float(r["close"]), 3),
                "volume": round(float(r["volume"]), 0),
            })
        return {
            "symbol": symbol,
            "scale": self.scale,
            "days": days,
            "bars": records,
        }

    @property
    def last_refresh(self) -> datetime | None:
        return self._last_refresh
