"""Broker 抽象基类：定义「券商/交易通道」的统一契约。

这是系统支持实盘的关键设计：
    - 现在用 PaperBroker（模拟撮合）跑通全流程
    - 未来开通 QMT / Ptrade 后，写一个 QMTBroker 继承本基类，
      把 buy/sell/query 映射到券商 API，系统其余部分零改动即可切换实盘。

所有实现都必须返回统一的数据结构，上层代码不关心底层是模拟还是真券商。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class TradeResult:
    """一次成交的结果。"""
    symbol: str
    side: str                 # 'buy' / 'sell'
    shares: float
    price: float              # 成交价（已含滑点）
    fee: float                # 手续费
    amount: float             # 成交金额（含费用）
    success: bool = True
    message: str = ""


@dataclass
class Position:
    """持仓快照。"""
    symbol: str
    shares: float
    avg_cost: float           # 平均成本价（不含费）
    market_value: float = 0.0 # 按最新价计算的市场价值
    unrealized_pnl: float = 0.0

    def __post_init__(self):
        self.unrealized_pnl = self.market_value - self.shares * self.avg_cost


class Broker(ABC):
    """交易通道抽象基类。"""

    @abstractmethod
    def query_cash(self) -> float:
        """返回可用现金。"""

    @abstractmethod
    def query_positions(self) -> list[Position]:
        """返回全部持仓。"""

    @abstractmethod
    def buy(self, symbol: str, shares: float, price: float, date: str) -> TradeResult:
        """买入。"""

    @abstractmethod
    def sell(self, symbol: str, shares: float, price: float, date: str) -> TradeResult:
        """卖出。"""
