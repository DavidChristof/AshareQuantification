"""QuoteManager：后台线程轮询实时行情，缓存最新快照供 API 读取。

- 后台 daemon 线程每隔 interval 秒抓取一次
- 线程安全（读快照返回副本，避免外部修改污染缓存）
- 抓取失败自动降级（保留上次数据），不中断循环
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Iterable

from .quoter import fetch_quotes

logger = logging.getLogger(__name__)


class QuoteManager:
    def __init__(self, symbols: Iterable[str], interval: float = 10.0):
        self.symbols = list(symbols)
        self.interval = max(interval, 3.0)      # 下限 3 秒，遵守接口频率
        self._quotes: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._last_update: datetime | None = None
        self._started = False
        self._stop = threading.Event()

    # ---------- 生命周期 ----------
    def start(self):
        """启动后台轮询线程（幂等）。"""
        if self._started:
            return
        self._started = True
        threading.Thread(target=self._loop, daemon=True, name="realtime-quoter").start()
        logger.info("实时行情轮询已启动，间隔 %.0f 秒，标的 %s",
                    self.interval, self.symbols)

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            self.refresh()
            self._stop.wait(self.interval)

    # ---------- 抓取与缓存 ----------
    def refresh(self):
        """立即抓取一次并更新缓存。"""
        try:
            quotes = fetch_quotes(self.symbols)
            if quotes:
                with self._lock:
                    self._quotes = quotes
                    self._last_update = datetime.now()
        except Exception as exc:  # noqa: BLE001 - 单次失败保留上次数据
            logger.error("实时行情刷新失败: %s", exc)

    # ---------- 读取 ----------
    def snapshot(self) -> list[dict]:
        """返回最新快照列表（副本，按股票池顺序）。"""
        with self._lock:
            return [dict(self._quotes[s]) for s in self.symbols if s in self._quotes]

    def get(self, symbol: str) -> dict | None:
        with self._lock:
            q = self._quotes.get(symbol)
            return dict(q) if q else None

    @property
    def last_update(self) -> datetime | None:
        with self._lock:
            return self._last_update
