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


def completed_closes(dates: Sequence, closes: Sequence, today) -> list[float]:
    """只保留 **today 之前** 已收盘交易日的收盘价 —— 让闸门对「今天」的口径唯一。

    ## 为什么必须这样做（2026-09-14）

    `fetch_index_daily` 返回的是**日线**，且**盘中不含当天、收盘后含当天**。
    直接取 `closes[-1]` 会让同一个交易日给出两个不同结论：

        09:31（自动调仓时）   : 最后一根 = 昨日 → close(D-1) vs MA20(≤D-1)
        15:30 之后（看板/手动）: 最后一根 = 今日 → close(D)   vs MA20(≤D)

    实测（沪深300 2020-2026，1624 个交易日）：两种口径有 **12.9%** 的交易日结论相反
    ⇒ 看板上显示的闸门口径，与当日实际约束交易的口径**不是同一个**。

    而且 **close(D) 要到收盘后才知道**，所以「09:31 那种口径」才是真正可执行的；
    回测若用 close(D)，验证的是一个**无法落地**的信号（见 docs）。

    这里统一定为「**决策日当天及以后的数据一律不用**」：闸门永远只看 `today` 之前
    已收盘的交易日。于是同一交易日内无论何时求值都得到同一答案，到次日自动前进一根。
    """
    out: list[float] = []
    key = str(today)[:10]
    for d, c in zip(dates or [], closes or []):
        if str(d)[:10] >= key:          # 今天及以后一律剔除（未收盘 / 未来）
            continue
        try:
            f = float(c)
        except (TypeError, ValueError):
            continue
        if f == f and f > 0:            # 非 NaN 且为正
            out.append(f)
    return out


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
