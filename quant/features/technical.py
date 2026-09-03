"""技术指标模块：纯 pandas 实现，方便理解公式（不依赖 ta-lib）。

所有指标均基于「收盘价」和「成交量」计算，输出为与输入等长的 Series。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_ma(df: pd.DataFrame, windows=(5, 10, 20)) -> pd.DataFrame:
    """移动平均线 MA，及价格相对均线的偏离度 (close/MA - 1)。"""
    for w in windows:
        ma = df["close"].rolling(w).mean()
        df[f"ma{w}"] = ma
        df[f"close_ma{w}_ratio"] = df["close"] / ma - 1.0
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """相对强弱指标 RSI：衡量近期涨跌幅力量对比，0~100。"""
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # 使用 ewm（指数加权）近似 Wilder 平滑
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)
    df["rsi"] = df["rsi"].fillna(50.0)
    return df


def add_macd(df: pd.DataFrame, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD 指标：DIF / DEA / 柱状图，反映趋势强度。"""
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    df["macd_dif"] = dif
    df["macd_dea"] = dea
    df["macd_hist"] = (dif - dea) * 2
    return df


def add_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """历史波动率：对数收益率的滚动标准差。"""
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["volatility"] = log_ret.rolling(period).std()
    return df


def add_volume_ratio(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """量比：当日成交量 / 过去 period 日均量，>1 表示放量。"""
    avg_vol = df["volume"].rolling(period).mean()
    df["volume_ratio"] = df["volume"] / avg_vol
    return df


def add_returns(df: pd.DataFrame, horizons=(1, 3, 5)) -> pd.DataFrame:
    """多周期收益率（过去 n 日的涨跌幅），作为模型输入特征。"""
    for h in horizons:
        df[f"ret_{h}d"] = df["close"].pct_change(h)
    return df


def add_atr(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """平均真实波幅 ATR：价格波动幅度的绝对度量（价格单位）。

    真实波幅 TR = max(最高-最低, |最高-前收|, |最低-前收|)，
    ATR = TR 的 period 日滚动均值。用于动态止盈止损。
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(period).mean()
    return df


# 注册表：按名称调用对应函数
_REGISTRY = {
    "ma": add_ma,
    "rsi": add_rsi,
    "macd": add_macd,
    "volatility": add_volatility,
    "volume_ratio": add_volume_ratio,
    "returns": add_returns,
    "atr": add_atr,
}


def compute_technical(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """根据配置中的指标名批量计算技术指标。"""
    df = df.copy()
    for name in indicators:
        if name not in _REGISTRY:
            raise ValueError(f"未知指标: {name}，可选 {list(_REGISTRY)}")
        _REGISTRY[name](df)
    return df
