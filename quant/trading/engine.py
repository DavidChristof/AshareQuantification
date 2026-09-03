"""交易引擎：把「预测信号」变成「实际调仓动作」。

两种模式：

【dual-threshold 多日持有】（clear_threshold 不为 None，自动纸面盘默认）
    目标：低换手 + 持有多日，只有信号明显转弱才动、用部分减持代替一刀切清仓。
    1. 核心榜 = 上涨概率 > threshold 里概率最高的 max_positions 只（目标满仓权重）
    2. 已持有但掉出核心榜、且概率仍 ≥ clear_threshold → 不卖出，
       只减持到「1 个权重槽 × hold_trim_pct」的观察仓（约半仓），继续观察
    3. 已持有且概率 < clear_threshold（或无当日信号）→ 全部卖出
    4. 核心榜内不足 → 补仓到目标市值；现金不足按可用现金买或跳过
    止盈/止损/移动止损由上层 apply_stop_rules 先行处理，本引擎不重复判断。

【classic】（clear_threshold=None，保持第一版行为）
    1. 只持有 prob > threshold 的 top max_positions
    2. 不在目标持仓中的 → 全部卖出；目标中不足 → 补仓

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
                 position_pct: float = 0.95, max_positions: int = 3,
                 clear_threshold: float | None = None,
                 hold_trim_pct: float = 0.5):
        """Args:
            clear_threshold: 清仓线。为 None 用旧逻辑（掉出核心榜即清仓）；
                             否则用「双阈值」：≥ 此线但掉出核心榜 → 减持到观察仓。
            hold_trim_pct:   观察仓占一个权重槽的比例（默认 0.5 = 半仓）。
        """
        self.broker = broker
        self.threshold = threshold
        self.position_pct = position_pct
        self.max_positions = max_positions
        self.clear_threshold = clear_threshold
        self.hold_trim_pct = hold_trim_pct

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

        # ---- 1. 选目标持仓（核心榜：prob>threshold 取前 max_positions）----
        probs = {s: float(p) for s, p in signals.items()}
        candidates = sorted(
            ((s, p) for s, p in probs.items() if p > self.threshold),
            key=lambda x: -x[1],
        )
        core = [s for s, _ in candidates[: self.max_positions]]
        core_set = set(core)
        logger.info("目标持仓（概率>%.2f，最多%d只）: %s",
                    self.threshold, self.max_positions, core)

        if self.clear_threshold is None:
            # ---------- classic：掉出核心榜即清仓 ----------
            for pos in self.broker.query_positions():
                if pos.symbol not in core_set:
                    price = prices.get(pos.symbol)
                    if price:
                        r = self.broker.sell(pos.symbol, pos.shares, price, date)
                        actions.append(f"卖出 {pos.symbol} {pos.shares:.0f}股 -> {r.message or 'OK'}")
            self._top_up_core(date, core, prices, actions)
            return self._finish(date, core, prices, actions)

        # ---------- dual-threshold：多日持有 + 部分减持 ----------
        total0 = self._current_equity(prices)
        slot = total0 * self.position_pct / max(1, self.max_positions)
        trim_target = slot * self.hold_trim_pct     # 观察仓目标市值

        # ---- 2. 处理现有持仓：跌破清仓线 → 清仓；掉出核心 → 减持到观察仓 ----
        held = {p.symbol: p for p in self.broker.query_positions()}
        for sym, pos in held.items():
            price = prices.get(sym)
            if not price:
                continue
            p = probs.get(sym)
            if p is None or p < self.clear_threshold:
                # 跌破清仓线 / 无当日信号 → 全部卖出
                reason = ("无当日信号，清仓" if p is None
                          else f"跌破清仓线 {self.clear_threshold:.2f}（概率 {p:.2f}）")
                r = self.broker.sell(sym, pos.shares, price, date, remark=reason)
                actions.append(f"清仓 {sym} {pos.shares:.0f}股（{reason}）"
                               f"-> {r.message or 'OK'}")
            elif sym not in core_set:
                # 掉出核心榜但未破线 → 减持到观察仓（若已 ≤ 观察仓则不动）
                cur_val = pos.shares * price
                if cur_val > trim_target * 1.15:
                    sell_sh = (cur_val - trim_target) / price
                    if sell_sh > 0:
                        reason = (f"掉出核心榜，减持至 {self.hold_trim_pct:.0%} 观察仓"
                                  f"（概率 {p:.2f} ≥ {self.clear_threshold:.2f}）")
                        r = self.broker.sell(sym, sell_sh, price, date, remark=reason)
                        actions.append(f"减持 {sym} {sell_sh:.0f}股（{reason}）"
                                       f"-> {r.message or 'OK'}")
            # else: 仍在核心榜 → 下面补仓处理

        # ---- 3. 核心榜补仓 / 新进 ----
        self._top_up_core(date, core, prices, actions)

        return self._finish(date, core, prices, actions)

    # ---------- 内部工具 ----------
    def _top_up_core(self, date: str, core: list, prices: dict, actions: list) -> None:
        """对核心榜补仓/新进：不足 85% 目标 → 补到目标（现金不足则按可用现金买）。"""
        if not core:
            return
        equity = self._current_equity(prices)
        budget_per = equity * self.position_pct / len(core)
        for sym in core:
            price = prices.get(sym)
            if not price:
                continue
            cur_val = self._position_value(sym, price)
            if cur_val >= budget_per * 0.85:      # 已达/接近目标 → 持有
                continue
            need = budget_per - cur_val
            if need <= 0:
                continue
            buy_price_per = price * (1 + self.broker.slippage)
            commission = getattr(self.broker, "commission", 0.0)
            cash = self.broker.query_cash()
            spend = need
            max_spend = cash / (1 + commission) if commission < 1 else cash
            if spend > max_spend:
                spend = max_spend
            if spend <= 1e-6:
                actions.append(f"买入 {sym} 跳过：现金不足（目标差 {need:.2f}）")
                continue
            shares_to_buy = spend / buy_price_per
            if shares_to_buy <= 0:
                continue
            r = self.broker.buy(sym, shares_to_buy, price, date)
            actions.append(f"买入 {sym} {shares_to_buy:.0f}股 -> {r.message or 'OK'}")

    def _finish(self, date: str, core: list, prices: dict, actions: list) -> dict:
        """净值快照并返回标准摘要。"""
        snapshot = self.broker.snapshot_equity(date, prices)
        logger.info("净值快照 %s: %.2f", date, snapshot)
        return {
            "date": date,
            "target": core,
            "actions": actions,
            "equity": snapshot,
        }

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
