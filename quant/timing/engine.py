"""择时引擎：把「多方法加权投票」汇总为每只股票的买卖点信号。

对每只股票：
    1. 用 4 个择时方法分别打分（+1/0/-1 × 置信度）
    2. 按市场状态给出的方法权重加权 → 综合分（-1 ~ +1）
    3. 综合分过阈值 → 买入/卖出，否则观望
    4. 附上触发理由（哪些方法、什么信号）
"""
from __future__ import annotations

import pandas as pd

from ..features.technical import compute_technical
from .methods import METHODS, method_cn
from .selector import select_weights

BUY_THRESHOLD = 0.25      # 综合分 >= 0.25 → 买入
SELL_THRESHOLD = -0.25    # 综合分 <= -0.25 → 卖出


class TimingEngine:
    def __init__(self, threshold: float = 0.55, sell_line: float = 0.45):
        self.threshold = threshold
        self.sell_line = sell_line

    def analyze(self, symbol: str, bars: pd.DataFrame, prob_up: float,
                regime: str) -> dict:
        """分析单只股票，返回择时信号。

        Args:
            symbol: 股票代码。
            bars: 原始行情表。
            prob_up: 模型上涨概率。
            regime: 市场状态（uptrend/range/downtrend）。

        Returns:
            {symbol, name, action, score, method_signals, reasons}
        """
        weights = select_weights(regime)
        feat = compute_technical(bars.copy(), ["ma", "rsi", "macd"])

        method_signals = {}
        reasons = []
        total_score = 0.0

        for name, method in METHODS.items():
            if name == "probability":
                sig, conf, reason = method(prob_up, self.threshold, self.sell_line)
            else:
                sig, conf, reason = method(feat)
            method_signals[name] = {
                "signal": sig,
                "confidence": round(conf, 2),
                "reason": reason,
                "weight": weights[name],
            }
            weighted = sig * conf * weights[name]
            total_score += weighted
            if sig != 0:
                reasons.append(f"{method_cn(name)}：{reason}")

        score = round(total_score, 3)
        if score >= BUY_THRESHOLD:
            action = "buy"
        elif score <= SELL_THRESHOLD:
            action = "sell"
        else:
            action = "hold"

        return {
            "symbol": symbol,
            "action": action,
            "score": score,
            "method_signals": method_signals,
            "reasons": reasons or ["各方法信号均中性，观望"],
        }
