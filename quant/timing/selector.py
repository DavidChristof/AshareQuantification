"""自主选择择时方法：根据市场状态给各方法分配权重。

核心思想：没有"万能"的择时方法，不同市场环境下应侧重不同方法——
    - 上涨趋势：追涨类方法有效 → 动量 + 趋势加权
    - 震荡整理：高抛低吸有效 → 均值回归加权
    - 下跌趋势：防守为主 → 模型概率主导（等概率高再买）

模型概率始终作为主信号参与，保证"模型判断买卖点"贯穿始终。
"""
from __future__ import annotations

# 各市场状态下的方法权重（可调）
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    "uptrend": {
        "probability": 0.40, "momentum": 0.30, "trend": 0.20, "mean_reversion": 0.10,
    },
    "range": {
        "probability": 0.30, "momentum": 0.15, "trend": 0.15, "mean_reversion": 0.40,
    },
    "downtrend": {
        "probability": 0.50, "momentum": 0.10, "trend": 0.10, "mean_reversion": 0.30,
    },
}

_REGIME_CN = {"uptrend": "上涨趋势", "range": "震荡整理", "downtrend": "下跌趋势"}


def select_weights(regime: str) -> dict[str, float]:
    """根据市场状态返回 {method: weight}（权重和为 1）。"""
    w = dict(REGIME_WEIGHTS.get(regime, REGIME_WEIGHTS["range"]))
    total = sum(w.values())
    return {k: round(v / total, 3) for k, v in w.items()}


def explain(regime: str) -> str:
    """生成「为什么这样配权重」的可读说明。"""
    note = {
        "uptrend": "市场处于上涨趋势，动量/趋势类方法更有效，故加权重；震荡指标降权",
        "range": "市场震荡整理，均值回归（超卖买/超买卖）更有效，故加权重",
        "downtrend": "市场处于下跌趋势，以防守为主，等待模型概率走高再介入",
    }[regime]
    return f"当前{_REGIME_CN[regime]}：{note}"
