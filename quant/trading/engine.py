"""交易引擎：把「预测信号」变成「实际调仓动作」。

每日调仓策略（第一版，简单可解释）：
    1. 选出 上涨概率 > threshold 的股票，按概率从高到低排序
    2. 最多持有 max_positions 只（等权重分配资金）
    3. 不在目标持仓中的 → 全部卖出
    4. 目标持仓中仓位不足的 → 补仓到目标市值
    5. 收盘后按当日收盘价成交（盘后决策的近似，回测惯例）

⚠️ 假设说明：用「当日预测 + 当日收盘价」成交，等价于每天收盘后
    根据预测结果下单、以收盘价成交，这是日线级模拟的标准近似。
"""
from __future__ import annotations

import logging

from .base import Broker
from .paper import PaperBroker

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(self, broker: Broker, threshold: float = 0.55,
                 position_pct: float = 0.95, max_positions: int = 3):
        self.broker = broker
        self.threshold = threshold
        self.position_pct = position_pct
        self.max_positions = max_positions

    def rebalance(self, date: str, signals: dict[str, float],
                  prices: dict[str, float]) -> dict:
        """执行一次调仓。

        Args:
            date: 交易日（YYYY-MM-DD）。
            signals: {symbol: 上涨概率}。
            prices:  {symbol: 最新收盘价}。

        Returns:
            调仓摘要（买卖动作、目标持仓）。
        """
        actions = []

        # ---- 1. 选目标持仓 ----
        candidates = sorted(
            ((s, p) for s, p in signals.items() if p > self.threshold),
            key=lambda x: -x[1],
        )
        target_symbols = [s for s, _ in candidates[: self.max_positions]]
        logger.info("目标持仓（概率>%.2f，最多%d只）: %s",
                    self.threshold, self.max_positions, target_symbols)

        # ---- 2. 卖出不在目标中的持仓 ----
        for pos in self.broker.query_positions():
            if pos.symbol not in target_symbols:
                price = prices.get(pos.symbol)
                if price:
                    r = self.broker.sell(pos.symbol, pos.shares, price, date)
                    actions.append(f"卖出 {pos.symbol} {pos.shares:.0f}股 -> {r.message or 'OK'}")

        # ---- 3. 计算目标仓位并买入/补仓 ----
        equity = self._current_equity(prices)
        if target_symbols:
            budget_per = equity * self.position_pct / len(target_symbols)
            for symbol in target_symbols:
                price = prices.get(symbol)
                if not price:
                    continue
                current_value = self._position_value(symbol, price)
                # 仓位明显不足才补仓（避免频繁小额交易）
                if current_value < budget_per * 0.85:
                    buy_value = budget_per - current_value
                    shares_to_buy = buy_value / price
                    if shares_to_buy > 0:
                        r = self.broker.buy(symbol, shares_to_buy, price, date)
                        actions.append(f"买入 {symbol} {shares_to_buy:.0f}股 -> {r.message or 'OK'}")

        # ---- 4. 净值快照 ----
        snapshot = self.broker.snapshot_equity(date, prices)
        logger.info("净值快照 %s: %.2f", date, snapshot)

        return {
            "date": date,
            "target": target_symbols,
            "actions": actions,
            "equity": snapshot,
        }

    # ---------- 内部工具 ----------
    def _current_equity(self, prices: dict[str, float]) -> float:
        cash = self.broker.query_cash()
        mv = sum(pos.shares * prices.get(pos.symbol, 0)
                 for pos in self.broker.query_positions())
        return cash + mv

    def _position_value(self, symbol: str, price: float) -> float:
        for pos in self.broker.query_positions():
            if pos.symbol == symbol:
                return pos.shares * price
        return 0.0
