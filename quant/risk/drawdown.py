"""账户回撤熔断（纯函数，无 IO）。

背景（2026-09-11 用户问「今天这种行情除止损和仓位控制还有别的风控吗」）：
系统原本只有「大盘当天弱 → 限制新买」，缺「**我自己亏多了**」的刹车。二者性质不同：
前者看天，后者才覆盖「我判断错了」的情形。

实证（600 池 top12 等权 · 2020-2026 · 含换手费，`docs/2026-09-11-risk-control.md`）：

| 方案 | 总收益 | 年化 | 最大回撤 | Sharpe |
|---|---|---|---|---|
| 无风控 | −4% | −0.7% | **−51.4%** | 0.08 |
| + 回撤熔断 8%/4% | +6% | +0.9% | **−32.5%** | 0.15 |
| + 回撤熔断 + 弱势减半 | +8% | +1.3% | −32.5% | **0.18** |
| 熔断更紧 6%/3% | +6% | +0.9% | −32.5% | 0.15 |

⇒ 8%/4% 是本样本的最佳档；更紧并不更好。

带**滞回**（hysteresis）：触发后不因回撤小幅收窄就立刻解除，要回到 `release_pct` 以内才解除，
避免"触发-解除-再触发"来回抖动（`tripped_before` 由调用方持久化，如 logs/drawdown_brake.json）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class DrawdownState:
    equity: float
    peak: float
    dd_pct: float            # 当前回撤（正数表示回撤幅度，如 8.3 表示 -8.3%）
    window_days: int
    tripped: bool
    reason: str

    def to_dict(self) -> dict:
        return {"equity": round(self.equity, 2), "peak": round(self.peak, 2),
                "dd_pct": round(self.dd_pct, 2), "window_days": self.window_days,
                "tripped": self.tripped, "reason": self.reason}


def _points(curve: Sequence) -> list[float]:
    """接受 [equity, ...] 或 [{"equity": x}, ...]（与 Broker.equity_history 同形）。"""
    out: list[float] = []
    for p in curve or []:
        v: Any = p.get("equity") if isinstance(p, dict) else p
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and f > 0:            # 非 NaN 且为正
            out.append(f)
    return out


def drawdown_pct(curve: Sequence, window_days: int = 60) -> float:
    """当前相对「近 window_days 个净值点峰值」的回撤百分比（正数 = 回撤）。"""
    pts = _points(curve)
    if not pts:
        return 0.0
    win = pts[-window_days:] if window_days and window_days > 0 else pts
    peak = max(win)
    if peak <= 0:
        return 0.0
    return max(0.0, (peak - pts[-1]) / peak * 100.0)


def evaluate(curve: Sequence, trip_pct: float = 8.0, release_pct: float = 4.0,
             window_days: int = 60, tripped_before: bool = False) -> DrawdownState:
    """评估熔断状态（带滞回）。

    - 未触发时：回撤 ≥ trip_pct → 触发。
    - 已触发时：回撤 < release_pct → 解除；否则维持触发（滞回区 [release, trip) 保持原状态）。
    """
    pts = _points(curve)
    equity = pts[-1] if pts else 0.0
    win = pts[-window_days:] if window_days and window_days > 0 else pts
    peak = max(win) if win else equity
    dd = drawdown_pct(pts, window_days)

    if tripped_before:
        tripped = dd >= float(release_pct)          # 回撤收窄到 release 以内才解除
        reason = (f"账户回撤 {dd:.1f}% 仍在警戒（回到 {release_pct:.0f}% 以内解除）"
                  if tripped else f"账户回撤已收窄至 {dd:.1f}%，熔断解除")
    else:
        tripped = dd >= float(trip_pct)
        reason = (f"账户自近 {window_days} 日高点 {peak:.2f} 回撤 {dd:.1f}% ≥ {trip_pct:.0f}% → 触发熔断"
                  if tripped else f"账户回撤 {dd:.1f}%（未达 {trip_pct:.0f}%）")
    return DrawdownState(equity=equity, peak=peak, dd_pct=dd, window_days=window_days,
                         tripped=tripped, reason=reason)
