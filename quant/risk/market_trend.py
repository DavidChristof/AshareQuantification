"""大盘趋势闸门（纯函数）：指数跌破均线 → 当日不开新仓。

背景（2026-09-11）：用户问"今天这种差行情除止损/仓位控制还有别的风控吗"。
与其去约束"持仓之间的相关性"（已回测证伪：候选池本身同质，约束只会让你取到更差的票），
不如直接看**大盘趋势**——跌破 MA20 就别开新仓。

实证（600 池 top12 等权·5 日调仓·2020-2026·含换手费·沪深300 真实数据）：

| 方案 | 年化 | 最大回撤 | Sharpe |
|---|---|---|---|
| 基准 | −0.7% | −51.4% | 0.08 |
| 仅回撤熔断（已上线） | 0.9% | −32.5% | 0.15 |
| **熔断 + 跌破MA20不开新仓** | **7.4%** | **−24.1%** | **0.57** |
| 熔断 + 跌破MA20**清仓** | −0.0% | −26.4% | 0.06 |

关键结论：**"不开新仓"对，"清仓"错**（清仓把收益全砍掉）；逐年 6/9 段不劣于仅熔断。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class TrendGate:
    level: float          # 指数当前点位
    ma: float             # 均线值
    ma_days: int
    below: bool           # 是否跌破均线
    reason: str

    def to_dict(self) -> dict:
        return {"level": round(self.level, 2), "ma": round(self.ma, 2),
                "ma_days": self.ma_days, "below": self.below, "reason": self.reason}


def trend_gate(closes: Sequence[float], ma_days: int = 20) -> TrendGate:
    """指数收盘序列 → 是否跌破 ma_days 均线。数据不足时视为**未跌破**（不干预）。"""
    vals: list[float] = []
    for c in closes or []:
        try:
            f = float(c)
        except (TypeError, ValueError):
            continue
        if f == f and f > 0:          # 非 NaN 且为正
            vals.append(f)
    if len(vals) < ma_days:
        return TrendGate(level=vals[-1] if vals else 0.0, ma=0.0, ma_days=ma_days,
                         below=False, reason="指数数据不足，趋势闸门不干预")
    level = vals[-1]
    ma = sum(vals[-ma_days:]) / ma_days
    below = level < ma
    if below:
        reason = (f"大盘 {level:.2f} 跌破 {ma_days} 日均线 {ma:.2f}"
                  f"（低 {(1 - level / ma) * 100:.2f}%）→ 当日不开新仓")
    else:
        reason = f"大盘 {level:.2f} 在 {ma_days} 日均线 {ma:.2f} 上方，趋势闸门放行"
    return TrendGate(level=level, ma=ma, ma_days=ma_days, below=below, reason=reason)
