"""市场状态判断（Regime Detection）。

用「股票池等权平均」作为市场代理指数（无需额外数据源），
根据均线位置 + 方向 + 波动率，判断当前处于哪种市场环境：
    - uptrend    上涨趋势（追涨环境）
    - range      震荡整理（高抛低吸环境）
    - downtrend  下跌趋势（防守环境）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_REGIME_CN = {"uptrend": "上涨趋势", "range": "震荡整理", "downtrend": "下跌趋势"}


class MarketRegime:
    def __init__(self, ma_window: int = 20, vol_window: int = 20):
        self.ma_window = ma_window
        self.vol_window = vol_window

    def detect(self, proxy_close: pd.Series) -> dict:
        """用代理指数收盘价判断市场状态。

        Args:
            proxy_close: 等权平均后的收盘价序列（DatetimeIndex）。

        Returns:
            {regime, price, ma20, volatility, description, trend_score}
        """
        close = proxy_close.dropna()
        if len(close) < self.ma_window + 6:
            return {"regime": "range", "price": None, "ma20": None,
                    "volatility": 0.0, "description": "数据不足", "trend_score": 0.0}

        ma = close.rolling(self.ma_window).mean()
        price, ma_now = float(close.iloc[-1]), float(ma.iloc[-1])
        ma_prev5 = float(ma.iloc[-6]) if len(ma) >= 6 else ma_now
        ma_rising = ma_now > ma_prev5

        # 趋势判定
        if price > ma_now and ma_rising:
            regime = "uptrend"
        elif price < ma_now and not ma_rising:
            regime = "downtrend"
        else:
            regime = "range"

        # 趋势强度：价格相对均线的偏离（归一化）
        trend_score = float((price / ma_now - 1) * 100) if ma_now else 0.0

        # 波动率（年化）
        ret = close.pct_change().dropna()
        vol = float(ret.tail(self.vol_window).std() * np.sqrt(252))

        return {
            "regime": regime,
            "price": round(price, 2),
            "ma20": round(ma_now, 2),
            "volatility": round(vol, 3),
            "description": _REGIME_CN[regime],
            "trend_score": round(trend_score, 2),
        }
