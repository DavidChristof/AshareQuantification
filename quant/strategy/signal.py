"""信号生成：模型输出（上涨概率）→ 交易信号（买入/空仓）。

这里的逻辑刻意保持简单、可解释：
    上涨概率 > threshold  → 持有（signal=1）
    否则                   → 空仓（signal=0）

你可以替换/扩展这个模块实现更复杂的策略，例如：
    1. 多档位仓位（0.3 / 0.6 / 1.0）
    2. 结合风险指标（波动率目标仓位）
    3. 多标的轮动打分
"""
from __future__ import annotations

import pandas as pd


def prob_to_signal(prob: pd.Series, threshold: float = 0.55) -> pd.Series:
    """概率 → 二值信号。"""
    return (prob > threshold).astype(int)


def build_signal_table(bars: pd.DataFrame, prob: pd.Series,
                       threshold: float = 0.55) -> pd.DataFrame:
    """组装最终信号表：date / close / prob_up / signal。"""
    out = bars.set_index("date")[["close"]].copy()
    out["prob_up"] = prob
    out["signal"] = prob_to_signal(out["prob_up"], threshold)
    return out.dropna(subset=["prob_up"])


class ThresholdStrategy:
    """最简单的阈值策略类，方便以后在回测中组合使用。"""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def generate(self, bars: pd.DataFrame, prob: pd.Series) -> pd.DataFrame:
        return build_signal_table(bars, prob, self.threshold)
