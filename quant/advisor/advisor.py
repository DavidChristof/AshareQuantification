"""TradeAdvisor：买卖决策辅助引擎。

综合两方面信息生成建议：
    1. 模型信号：LSTM/Transformer 预测的未来上涨概率
    2. 技术面投票：均线、RSI、MACD、短期动量四个维度各自投出多/空票

决策规则（透明、可解释）：
    - 买入：模型概率 > 阈值 且 技术面净票数 >= 1
    - 卖出：模型概率 < 卖出线  或 (技术面净票数 <= -2 且 概率 < 0.5)
    - 观望：其余情况

输出：Advice(action, prob_up, score, reasons) —— reasons 是人话理由列表。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Advice:
    """一条买卖建议。"""
    symbol: str
    action: str                 # 'buy' / 'sell' / 'wait'
    prob_up: float              # 模型上涨概率
    score: int = 0              # 技术面多空净票数（-4 ~ +4）
    reasons: list = field(default_factory=list)
    holding: bool = False       # 当前是否持仓（前端展示用）

    @property
    def label(self) -> str:
        return {"buy": "买入", "sell": "卖出", "wait": "观望"}.get(self.action, self.action)


class TradeAdvisor:
    def __init__(self, threshold: float = 0.55, prob_sell: float = 0.45):
        self.threshold = threshold
        self.prob_sell = prob_sell

    def analyze(self, symbol: str, bars: pd.DataFrame,
                prob_up: float, holding: bool = False) -> Advice:
        """分析一只股票。

        Args:
            symbol: 股票代码。
            bars: 原始行情表（含 date/open/high/low/close/volume 列），
                  内部基于真实价格计算技术指标，保证判断可解释。
            prob_up: 模型预测的上涨概率（0~1）。
            holding: 当前是否持仓。
        """
        from ..features.technical import compute_technical

        # 用原始价格计算指标（不做标准化），判断基于真实价格
        feat = compute_technical(bars.copy(), ["ma", "rsi", "macd", "returns"])
        bull, bear, notes = self._technical_votes(feat)
        score = len(bull) - len(bear)
        prob = float(prob_up)

        # ---- 决策规则 ----
        if prob > self.threshold and score >= 1:
            action = "buy"
            reasons = [f"模型看涨概率 {prob:.0%}，超过买入阈值 {self.threshold:.0%}"]
        elif prob < self.prob_sell or (score <= -2 and prob < 0.5):
            action = "sell"
            reasons = [f"模型看涨概率 {prob:.0%}，低于卖出线 {self.prob_sell:.0%}"]
        else:
            action = "wait"
            reasons = [f"信号不够强（概率 {prob:.0%}），建议观望等更明确信号"]

        # 技术面理由
        reasons.extend(notes)

        # 持仓状态提示
        if holding:
            reasons.append("你当前持有该股")
        return Advice(symbol=symbol, action=action, prob_up=prob,
                      score=score, reasons=reasons, holding=holding)

    # ---------- 技术面投票 ----------
    def _technical_votes(self, feat: pd.DataFrame) -> tuple[list, list, list]:
        """从特征表最新一行提取四个技术维度，返回 (多头名单, 空头名单, 说明文字)。"""
        bull, bear, notes = [], [], []
        last = feat.iloc[-1]
        close = float(last["close"])

        # 1. 均线趋势
        ma20 = last.get("ma20")
        if pd.notna(ma20):
            if close > float(ma20):
                bull.append("ma")
                notes.append(f"价格 {close:.2f} 站上 20 日均线 {ma20:.2f}，趋势偏多")
            else:
                bear.append("ma")
                notes.append(f"价格 {close:.2f} 跌破 20 日均线 {ma20:.2f}，趋势偏空")

        # 2. RSI 超买/超卖
        rsi = last.get("rsi")
        if pd.notna(rsi):
            rsi = float(rsi)
            if rsi < 30:
                bull.append("rsi")
                notes.append(f"RSI {rsi:.0f} 超卖，存在超跌反弹机会")
            elif rsi > 70:
                bear.append("rsi")
                notes.append(f"RSI {rsi:.0f} 超买，注意回调风险")
            else:
                notes.append(f"RSI {rsi:.0f} 处于中性区间")

        # 3. MACD 金叉/死叉
        dif, dea = last.get("macd_dif"), last.get("macd_dea")
        if pd.notna(dif) and pd.notna(dea):
            if float(dif) > float(dea):
                bull.append("macd")
                notes.append("MACD 金叉（DIF 在 DEA 上方），多头动能")
            else:
                bear.append("macd")
                notes.append("MACD 死叉（DIF 在 DEA 下方），空头动能")

        # 4. 短期动量
        ret5 = last.get("ret_5d")
        if pd.notna(ret5):
            ret5 = float(ret5)
            if ret5 > 0:
                bull.append("mom")
                notes.append(f"近 5 日上涨 {ret5:.1%}，短期动量向上")
            else:
                bear.append("mom")
                notes.append(f"近 5 日下跌 {ret5:.1%}，短期动量向下")

        return bull, bear, notes
