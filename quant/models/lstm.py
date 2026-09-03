"""LSTM 预测模型：用过去 window 天的特征序列预测未来 horizon 天上涨概率。

结构：
    输入 (B, T, F) → LSTM 编码时序 → 取最后一个时间步的隐状态 → 全连接 → 概率 logit
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,          # 输入形状为 (batch, seq_len, features)
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)  # 输出单个 logit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)                # out: (B, T, hidden)
        last = out[:, -1, :]                 # 取最后一个时间步
        last = self.dropout(last)
        return self.fc(last).squeeze(-1)     # (B,)
