"""逐日滚动择时信号（回测用，无未来泄漏）。

一次性算好技术指标全序列，逐日切片调用 4 个择时方法（概率/动量/均值回归/趋势），
按当日市场状态选取方法权重加权 → 综合分 → 买卖信号。

与 quant/timing/engine.py 的唯一区别：engine 只对「最新一天」分析，
这里对回测期的每一天都产出信号，且只用截至当天的数据（无前视偏差）。
"""
from __future__ import annotations

import pandas as pd

from ..features.technical import compute_technical
from .methods import METHODS
from .selector import select_weights

BUY_THRESHOLD = 0.25      # 综合分 >= 0.25 → 买入
SELL_THRESHOLD = -0.25    # 综合分 <= -0.25 → 卖出


def roll_regime(market_close: pd.Series, ma_window: int = 20) -> pd.Series:
    """逐日市场状态：价格 vs MA20 + MA20 方向（与 MarketRegime 判定一致）。

    Args:
        market_close: 市场代理指数（如股票池等权收盘）收盘价序列。
        ma_window: 均线窗口。

    Returns:
        每日期 regime（uptrend / range / downtrend）。
    """
    ma = market_close.rolling(ma_window).mean()
    ma_prev = ma.shift(5)          # 5 天前均线（判断均线方向）
    regimes = []
    for date in market_close.index:
        p, m, mp = market_close.loc[date], ma.loc[date], ma_prev.loc[date]
        if pd.isna(m) or pd.isna(mp) or m <= 0:
            regimes.append("range")
        elif p > m and m > mp:
            regimes.append("uptrend")
        elif p < m and m < mp:
            regimes.append("downtrend")
        else:
            regimes.append("range")
    return pd.Series(regimes, index=market_close.index)


def roll_timing_signal(bars: pd.DataFrame, prob_up: pd.Series,
                       regime_series: pd.Series,
                       threshold: float = 0.55, sell_line: float = 0.45
                       ) -> tuple[pd.Series, pd.Series]:
    """对每只股票逐日产出择时综合分与动作。

    Args:
        bars: 原始行情（含 high/low/close），index=date。
        prob_up: 模型上涨概率序列（与 bars 对齐）。
        regime_series: 每日期市场状态（roll_regime 输出）。

    Returns:
        (score_series, action_series)：综合分(-1~1) 与动作(buy/sell/hold)。
    """
    feat = compute_technical(bars.copy(), ["ma", "rsi", "macd"])
    idx = bars.index
    scores, actions = [], []
    for i, date in enumerate(idx):
        if i < 1:
            scores.append(0.0)
            actions.append("hold")
            continue
        # 按日期标签取值（bars 与 prob_up / regime 索引可能不完全一致）
        prob = (float(prob_up.loc[date])
                if date in prob_up.index and not pd.isna(prob_up.loc[date]) else 0.5)
        regime = regime_series.loc[date] if date in regime_series.index else "range"
        weights = select_weights(regime)
        f = feat.iloc[: i + 1]        # 只用截至当天的数据（无未来泄漏）
        total = 0.0
        for name, method in METHODS.items():
            if name == "probability":
                sig, conf, _ = method(prob, threshold, sell_line)
            else:
                sig, conf, _ = method(f)
            total += sig * conf * weights[name]
        scores.append(round(total, 3))
        actions.append("buy" if total >= BUY_THRESHOLD
                       else ("sell" if total <= SELL_THRESHOLD else "hold"))
    return pd.Series(scores, index=idx), pd.Series(actions, index=idx)
