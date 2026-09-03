"""回测引擎：基于信号表模拟单标的交易，输出净值曲线。

规则（简化但符合直觉）：
    - 每日收盘时检查信号。
    - signal=1 且当前空仓 → 用可用资金的 position_pct 全仓买入。
    - signal=0 且当前持仓 → 清仓。
    - 每次交易都扣除佣金和滑点（卖出另计印花税）。
    - 可选：持仓期间按 ATR 动态止盈止损（止损/止盈/移动止损）。
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    def __init__(self, initial_capital: float = 100_000.0,
                 commission: float = 0.0003,
                 slippage: float = 0.0002,
                 position_pct: float = 0.95,
                 stamp_tax: float = 0.0005):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_pct = position_pct
        self.stamp_tax = stamp_tax

    def run(self, signal_table: pd.DataFrame, risk_cfg: dict | None = None,
            atr: pd.Series | None = None) -> pd.DataFrame:
        """执行回测。

        Args:
            signal_table: 含 close 和 signal 列的 DataFrame（index=date）。
            risk_cfg: 动态止盈止损参数（可选，见 quant.risk.volatility.vol_cfg_from_risk）。
            atr: 每日 ATR 序列（index 与 signal_table 对齐；配合 risk_cfg 使用）。

        Returns:
            DataFrame：date / close / signal / cash / holdings / equity（含 stop_reason）
        """
        data = signal_table.copy()
        data = data[data["close"].notna()]

        cash = float(self.initial_capital)
        shares = 0.0
        cost = 0.0          # 持仓成本（买入成交价，含滑点）
        max_price = 0.0     # 持仓期间最高价（移动止损基准）
        records = []
        risk_cfg = risk_cfg or {}

        for date, row in data.iterrows():
            close = float(row["close"])
            signal = int(row["signal"])
            stop_reason = None

            # ---- 持仓期：先更新最高价，再检查 ATR 动态止盈止损 ----
            if shares > 0:
                if close > max_price:
                    max_price = close
                if risk_cfg and atr is not None and date in atr.index:
                    atr_val = float(atr.loc[date])
                    if atr_val > 0 and cost > 0:
                        from ..risk.volatility import dynamic_pcts
                        dyn = dynamic_pcts(
                            {"atr": atr_val}, cost, high=max_price,
                            stop_mult=risk_cfg.get("atr_stop_mult", 2.5),
                            take_mult=risk_cfg.get("atr_take_mult", 3.5),
                            trail_mult=risk_cfg.get("atr_trailing_mult", 2.5),
                            min_pct=risk_cfg.get("vol_min_pct", 0.03),
                            max_pct=risk_cfg.get("vol_max_pct", 0.15),
                            take_min_pct=risk_cfg.get("take_min_pct", 0.05),
                            take_max_pct=risk_cfg.get("take_max_pct", 0.30),
                        )
                        if dyn["stop_pct"]:
                            if close <= dyn["stop_price"]:
                                stop_reason = f"止损{close:.2f}≤{dyn['stop_price']:.2f}"
                            elif close >= dyn["take_price"]:
                                stop_reason = f"止盈{close:.2f}≥{dyn['take_price']:.2f}"
                            elif max_price > cost and close <= max_price * (1 - dyn["trail_pct"]):
                                stop_reason = f"移动止损{close:.2f}≤{max_price:.2f}×{1 - dyn['trail_pct']:.0%}"

            # ---- 交易决策（当日收盘价成交）----
            if shares == 0:
                if signal == 1:
                    # 买入：投入可用资金的一定比例
                    budget = cash * self.position_pct
                    buy_price = close * (1 + self.slippage)   # 滑点抬高买价
                    shares = budget / buy_price
                    cost = buy_price
                    max_price = buy_price
                    fee = shares * buy_price * self.commission
                    cash -= (shares * buy_price + fee)
            else:
                if signal == 0 or stop_reason is not None:
                    # 卖出：滑点压低卖价（含印花税）
                    sell_price = close * (1 - self.slippage)
                    proceeds = shares * sell_price
                    fee = proceeds * (self.commission + self.stamp_tax)
                    cash += (proceeds - fee)
                    shares = 0.0
                    cost = 0.0
                    max_price = 0.0

            equity = cash + shares * close
            records.append({"date": date, "close": close, "signal": signal,
                            "cash": cash, "holdings": shares * close,
                            "equity": equity, "stop_reason": stop_reason})

        result = pd.DataFrame(records).set_index("date")
        return result

    def buy_and_hold(self, signal_table: pd.DataFrame) -> pd.DataFrame:
        """基准策略：第一天买入并一直持有到结束（不交易）。"""
        data = signal_table[signal_table["close"].notna()].copy()
        close_first = data["close"].iloc[0]
        shares = (self.initial_capital * self.position_pct) / (close_first * (1 + self.slippage))
        fee = shares * close_first * (1 + self.slippage) * self.commission
        cash = self.initial_capital - shares * close_first * (1 + self.slippage) - fee

        equity = cash + shares * data["close"]
        return pd.DataFrame({
            "close": data["close"],
            "signal": 1,
            "cash": cash,
            "holdings": shares * data["close"],
            "equity": equity,
        })
