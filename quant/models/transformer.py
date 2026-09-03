"""Transformer 预测模型（轻量版）。

与 LSTM 的差异：用自注意力直接建模任意两个时间步之间的关系，
对长序列更友好，但小样本下容易过拟合（本项目默认用 LSTM）。

结构：
    输入 (B, T, F) → 线性嵌入 → 位置编码 → TransformerEncoder → 最后时间步 → 全连接
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """经典 sin/cos 位置编码，让模型感知时间顺序。"""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.2,
                 nhead: int = 4, max_len: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_enc = PositionalEncoding(hidden_size, max_len=max_len, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout,
            batch_first=True, dim_feedforward=hidden_size * 4,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)          # (B, T, F) -> (B, T, hidden)
        x = self.pos_enc(x)
        x = self.encoder(x)             # (B, T, hidden)
        last = x[:, -1, :]
        return self.fc(last).squeeze(-1)
