"""特征流水线：原始 K 线 → 标准化特征 → 时序窗口 (X, y)。

核心设计：
1. 计算技术指标得到原始特征列。
2. 滚动 z-score 标准化：只用过去 window 天数据求均值/方差，
   —— 这是防止「未来信息泄漏」的关键（训练时偷看未来会让回测失真）。
3. 滑动窗口切分：X[i] = 第 i 天往前 window 天的特征，y[i] = 未来 horizon 天的收益方向。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .technical import compute_technical

logger = logging.getLogger(__name__)


class FeaturePipeline:
    def __init__(self, window: int = 30, horizon: int = 5,
                 indicators: list[str] | None = None):
        self.window = window
        self.horizon = horizon
        self.indicators = indicators or [
            "ma", "rsi", "macd", "volatility", "volume_ratio", "returns",
        ]

        # 所有参与建模的特征列（顺序固定，训练/推理必须一致）。
        # 注意：预测器加载 checkpoint 后会覆盖这个列表，保证与训练时完全一致。
        self.feature_columns = [
            # 原始价格信息
            "open", "high", "low", "close", "volume",
            # 技术指标
            "ma5", "ma10", "ma20",
            "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
            "rsi", "macd_dif", "macd_dea", "macd_hist",
            "volatility", "volume_ratio",
            "ret_1d", "ret_3d", "ret_5d",
        ]

    # ---------- 核心流程 ----------
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """K线 → 标准化后的特征表（含 date 列）。"""
        df = df.copy().sort_values("date").reset_index(drop=True)
        df = compute_technical(df, self.indicators)

        # 只保留建模需要的列 + 日期
        feat = df[["date"] + self.feature_columns].copy()

        # 滚动标准化：每个值用 [t-window, t] 窗口内的均值/标准差归一
        for col in self.feature_columns:
            mean = feat[col].rolling(self.window, min_periods=1).mean()
            std = feat[col].rolling(self.window, min_periods=1).std()
            feat[col] = (feat[col] - mean) / std.replace(0, np.nan)
            feat[col] = feat[col].fillna(0.0)

        return feat

    def make_label(self, df: pd.DataFrame) -> pd.Series:
        """标签：未来 horizon 天收益是否为正（二分类）。"""
        future_ret = df["close"].shift(-self.horizon) / df["close"] - 1.0
        return (future_ret > 0).astype(np.float32)

    def make_windows(self, df: pd.DataFrame):
        """生成 (X, y, dates)。

        X:     (N, window, n_features) float32
        y:     (N,) float32，1=未来 horizon 天上涨，0=下跌
        dates: (N,) 每个样本对应的日期
        """
        feat = self.build_features(df)
        y = self.make_label(df)
        dates = feat["date"].values

        X, Y, D = [], [], []
        n = len(feat)
        for i in range(self.window, n):
            # 窗口终点 i 对应样本日期为 feat 的第 i 行
            x_win = feat.iloc[i - self.window: i][self.feature_columns].values
            X.append(x_win)
            Y.append(y.iloc[i])
            D.append(dates[i])

        return (np.asarray(X, dtype=np.float32),
                np.asarray(Y, dtype=np.float32),
                np.asarray(D, dtype="datetime64[D]"))
