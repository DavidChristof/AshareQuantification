"""择时方法库：每种方法对单只股票输出信号。

信号约定：+1 = 买入，0 = 观望，-1 = 卖出
每个方法返回 (signal, confidence, reason)：
    signal      买卖信号
    confidence  方法置信度（0~1）
    reason      触发原因（可读文本）

方法输入统一为「已计算技术指标的特征表」（含 ma20/rsi/macd_*），
概率方法额外需要模型上涨概率 prob_up。
"""
from __future__ import annotations

import pandas as pd

BUY, HOLD, SELL = 1, 0, -1
_CN = {"probability": "概率择时", "momentum": "动量择时",
       "mean_reversion": "均值回归", "trend": "趋势择时"}


def probability_method(prob_up: float, threshold: float = 0.55,
                       sell_line: float = 0.45):
    """模型概率择时：概率高买入，概率低卖出。"""
    prob = float(prob_up)
    if prob >= threshold:
        conf = min(1.0, (prob - threshold) / 0.15)
        return BUY, conf, f"模型上涨概率 {prob:.0%} ≥ 买入阈值 {threshold:.0%}"
    if prob <= sell_line:
        conf = min(1.0, (sell_line - prob) / 0.15)
        return SELL, conf, f"模型上涨概率 {prob:.0%} ≤ 卖出线 {sell_line:.0%}"
    return HOLD, 0.0, "模型概率处于中性区间"


def momentum_method(feat: pd.DataFrame):
    """动量择时：价格站上 MA20 且 MA20 向上 → 买；跌破且向下 → 卖。"""
    last = feat.iloc[-1]
    close, ma20 = float(last["close"]), float(last["ma20"])
    if pd.isna(ma20) or ma20 <= 0:
        return HOLD, 0.0, "均线数据不足"
    ma20_prev = float(feat["ma20"].iloc[-6]) if len(feat) >= 6 else ma20
    ma_rising = ma20 > ma20_prev

    if close > ma20 and ma_rising:
        return BUY, 0.6, f"价格 {close:.2f} 站上 MA20 {ma20:.2f} 且均线上行"
    if close < ma20 and not ma_rising:
        return SELL, 0.6, f"价格 {close:.2f} 跌破 MA20 {ma20:.2f} 且均线下行"
    return HOLD, 0.3, "动量信号中性"


def mean_reversion_method(feat: pd.DataFrame):
    """均值回归择时：RSI 超卖买 / 超买卖。"""
    last = feat.iloc[-1]
    rsi = float(last["rsi"])
    if pd.isna(rsi):
        return HOLD, 0.0, "RSI 数据不足"
    if rsi < 30:
        return BUY, 0.7, f"RSI {rsi:.0f} 超卖，超跌反弹机会"
    if rsi > 70:
        return SELL, 0.7, f"RSI {rsi:.0f} 超买，注意回调"
    return HOLD, 0.2, f"RSI {rsi:.0f} 中性"


def trend_method(feat: pd.DataFrame):
    """趋势择时：MACD 金叉+站上均线 → 买；死叉+破均线 → 卖。"""
    last = feat.iloc[-1]
    dif, dea = float(last["macd_dif"]), float(last["macd_dea"])
    close, ma20 = float(last["close"]), float(last["ma20"])
    if pd.isna(dif) or pd.isna(ma20):
        return HOLD, 0.0, "MACD/均线数据不足"
    bull = dif > dea and close > ma20
    bear = dif < dea and close < ma20
    if bull:
        return BUY, 0.6, "MACD 金叉且价格站上 MA20"
    if bear:
        return SELL, 0.6, "MACD 死叉且价格跌破 MA20"
    return HOLD, 0.2, "趋势信号中性"


METHODS = {
    "probability": probability_method,
    "momentum": momentum_method,
    "mean_reversion": mean_reversion_method,
    "trend": trend_method,
}


def method_cn(name: str) -> str:
    return _CN.get(name, name)
