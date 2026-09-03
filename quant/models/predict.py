"""预测与信号生成：用训练好的模型对行情数据输出上涨概率和交易信号。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..features.pipeline import FeaturePipeline
from .trainer import load_checkpoint


class ModelPredictor:
    """封装加载模型 + 生成概率 + 生成信号。"""

    def __init__(self, checkpoint_path: str):
        ckpt = load_checkpoint(checkpoint_path)
        self.model = ckpt["model"].eval()
        self.meta = ckpt["meta"]

        # 用 checkpoint 里保存的窗口/horizon/特征列，保证与训练一致
        self.window = self.meta["window"]
        self.horizon = self.meta["horizon"]
        self.feature_columns = self.meta["feature_columns"]
        self.pipeline = FeaturePipeline(
            window=self.window, horizon=self.horizon,
            indicators=["ma", "rsi", "macd", "volatility", "volume_ratio", "returns"],
        )
        # 覆盖为保存的特征列（顺序必须严格一致）
        self.pipeline.feature_columns = self.feature_columns  # type: ignore[assignment]
        # 训练时在验证集上调优的最优阈值（类别不平衡时优于默认 0.5/0.55）
        self.best_threshold = float(self.meta.get("best_threshold", 0.55))
        # 可选 Platt 校准：先 clip logit 到验证区间，再 p = sigmoid(A*logit + B)
        calib = self.meta.get("calib")
        self._calib = dict(calib) if isinstance(calib, dict) else \
            (tuple(calib) if calib else None)

    def _prob(self, logit: float) -> float:
        if self._calib:
            if isinstance(self._calib, dict):
                a, b = self._calib["A"], self._calib["B"]
                lo, hi = self._calib.get("lo"), self._calib.get("hi")
                x = logit
                if lo is not None and hi is not None:
                    x = min(max(x, float(lo)), float(hi))   # 防分布外极端外推
                return float(1.0 / (1.0 + np.exp(-(a * x + b))))
            a, b = self._calib
            return float(1.0 / (1.0 + np.exp(-(a * logit + b))))
        return float(1.0 / (1.0 + np.exp(-logit)))

    def predict_probability(self, df: pd.DataFrame) -> pd.Series:
        """对整段行情逐日预测上涨概率，返回与日期对齐的 Series。

        Args:
            df: 含 date/open/high/low/close/volume 的行情表。
        """
        feat = self.pipeline.build_features(df)
        dates = feat["date"].values

        probs = np.full(len(feat), np.nan)
        n = len(feat)
        device = next(self.model.parameters()).device

        with torch.no_grad():
            for i in range(self.window, n):
                x = feat.iloc[i - self.window: i][self.feature_columns].values
                x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
                logit = self.model(x)
                probs[i] = self._prob(logit.item())

        return pd.Series(probs, index=dates, name="prob_up")

    def latest_probability(self, df: pd.DataFrame) -> float | None:
        """只预测「最新一个窗口」的上涨概率（实时用，速度快）。

        Args:
            df: 含 date/open/high/low/close/volume 的行情表。
        """
        feat = self.pipeline.build_features(df)
        if len(feat) < self.window:
            return None
        x = feat.iloc[-self.window:][self.feature_columns].values
        x = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            return self._prob(self.model(x.to(device)).item())

    def make_signal(self, df: pd.DataFrame, threshold: float | None = None
                    ) -> pd.DataFrame:
        """生成交易信号表：date / close / prob_up / signal（1=买，0=空仓）。

        未显式传 threshold 时，使用训练时在验证集上调优的最优阈值。
        """
        prob = self.predict_probability(df)
        out = df.set_index("date")[["close"]].copy()
        out["prob_up"] = prob
        thr = self.best_threshold if threshold is None else threshold
        out["signal"] = (prob > thr).astype(int)
        return out.dropna(subset=["prob_up"])
