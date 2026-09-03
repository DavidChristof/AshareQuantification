"""PyTorch Dataset 与「按时间顺序」切分工具。

⚠️ 时间序列切分不能用 sklearn 的随机 train_test_split！
    随机打乱会破坏时间依赖，且模型会"看到"未来数据（信息泄漏）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """把 (X, y) 包成 PyTorch Dataset。X 形状 (N, window, features)。"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def temporal_split(X: np.ndarray, y: np.ndarray, val_ratio: float = 0.2):
    """按时间顺序切分为训练集 / 验证集（后 20% 作为验证集）。"""
    n = len(y)
    split = int(n * (1 - val_ratio))
    return (TimeSeriesDataset(X[:split], y[:split]),
            TimeSeriesDataset(X[split:], y[split:]))
