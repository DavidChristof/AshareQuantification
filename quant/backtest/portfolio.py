"""组合交易策略回测：多股票同时持仓，定期按概率排名调仓（等权）。

比单标的回测更贴近真实组合投资：
    - 每 N 个交易日调仓一次（减少换手成本）
    - 按最新上涨概率排序，选 top N 等权持有
    - 支持市场状态过滤（下跌市空仓，组合级择时）
    - 计入佣金 + 滑点 + 印花税

与 quant/trading/engine.py 的 TradingEngine 区别：这是「历史回测」版
（纯内存、逐日模拟），TradingEngine 是「实时调仓」版（PaperBroker 撮合）。
"""
from __future__ import annotations

import pandas as pd


class PortfolioBacktest:
    """组合回测引擎：现金 + 多股票持仓，定期再平衡。"""

    def __init__(self, initial_capital: float = 100_000.0,
                 commission: float = 0.0003, slippage: float = 0.0002,
                 stamp_tax: float = 0.0005,
                 max_positions: int = 5, position_pct: float = 0.95):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax
        self.max_positions = max_positions
        self.position_pct = position_pct

    def run(self, close_panel: pd.DataFrame, prob_panel: pd.DataFrame,
            rebalance_every: int = 5,
            regime_series: pd.Series | None = None,
            flat_on_downtrend: bool = True) -> pd.DataFrame:
        """执行组合回测。

        Args:
            close_panel: DataFrame，列=股票代码，行=date，值=收盘价（可含 NaN）。
            prob_panel: DataFrame，列=股票代码，行=date，值=上涨概率（NaN=不可交易）。
            rebalance_every: 每 N 个交易日调仓一次。
            regime_series: 每日期市场状态（uptrend/range/downtrend），可选。
            flat_on_downtrend: 下跌市整体空仓（组合级择时）。

        Returns:
            DataFrame（date/cash/holdings/equity/n_positions），
            调仓记录存在 result.attrs["trades"]。
        """
        dates = close_panel.index
        cash = float(self.initial_capital)
        shares: dict[str, float] = {}
        trades: list[dict] = []
        last_rb = -10 ** 9
        rows = []

        for i, date in enumerate(dates):
            do_rb = (i == 0) or (i - last_rb >= rebalance_every)
            regime = regime_series.loc[date] if regime_series is not None else None

            if do_rb:
                targets = self._select_targets(prob_panel, date, regime, flat_on_downtrend)
                cash, shares = self._rebalance(date, close_panel, cash, shares,
                                               targets, trades)
                last_rb = i

            # 组合市值
            mv = 0.0
            for sym, sh in shares.items():
                if sym in close_panel.columns and pd.notna(close_panel.loc[date, sym]):
                    mv += sh * float(close_panel.loc[date, sym])
            equity = cash + mv
            rows.append({"date": date, "cash": cash, "holdings": mv,
                         "equity": equity, "n_positions": len(shares)})

        result = pd.DataFrame(rows).set_index("date")
        result.attrs["trades"] = trades
        return result

    def _select_targets(self, prob_panel: pd.DataFrame, date, regime,
                        flat_on_downtrend: bool) -> list[str]:
        """按当日上涨概率排序取 top N（下跌市可空仓）。"""
        if regime == "downtrend" and flat_on_downtrend:
            return []
        if date not in prob_panel.index:
            return []
        row = prob_panel.loc[date].dropna()
        return row.sort_values(ascending=False).head(self.max_positions).index.tolist()

    def _rebalance(self, date, close_panel, cash, shares, targets, trades):
        """卖出掉出榜的 → 把持仓调整到等权目标（计入费用）。"""
        prices = {}
        for sym in set(list(shares) + list(targets)):
            if sym in close_panel.columns and pd.notna(close_panel.loc[date, sym]):
                prices[sym] = float(close_panel.loc[date, sym])

        # 1. 卖出不在目标榜内的持仓
        for sym in list(shares):
            if sym not in targets and sym in prices:
                sh = shares.pop(sym)
                sell_price = prices[sym] * (1 - self.slippage)
                proceeds = sh * sell_price
                fee = proceeds * (self.commission + self.stamp_tax)
                cash += proceeds - fee
                trades.append({"date": date, "symbol": sym, "side": "sell",
                               "shares": round(sh, 2), "price": round(sell_price, 3),
                               "reason": "掉出前N"})

        # 2. 计算总资产，目标等权
        mv = sum(sh * prices[s] for s, sh in shares.items() if s in prices)
        total = cash + mv
        if not targets or total <= 0:
            return cash, shares
        per = total * self.position_pct / len(targets)

        for sym in targets:
            if sym not in prices:
                continue
            price = prices[sym]
            cur_value = shares.get(sym, 0.0) * price
            diff = per - cur_value
            if diff > 1e-6:
                buy_price = price * (1 + self.slippage)
                sh = diff / buy_price
                cost = sh * buy_price
                fee = cost * self.commission
                if cash >= cost + fee:
                    cash -= cost + fee
                    shares[sym] = shares.get(sym, 0.0) + sh
                    trades.append({"date": date, "symbol": sym, "side": "buy",
                                   "shares": round(sh, 2), "price": round(buy_price, 3),
                                   "reason": "买入/加仓"})
            elif diff < -1e-6:
                sell_price = price * (1 - self.slippage)
                sh = min(-diff / sell_price, shares.get(sym, 0.0))
                proceeds = sh * sell_price
                fee = proceeds * (self.commission + self.stamp_tax)
                cash += proceeds - fee
                shares[sym] -= sh
                if shares[sym] < 1e-6:
                    shares.pop(sym, None)
                trades.append({"date": date, "symbol": sym, "side": "sell",
                               "shares": round(sh, 2), "price": round(sell_price, 3),
                               "reason": "减仓"})
        return cash, shares
