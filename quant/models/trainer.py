"""训练器：训练循环 + 时间序列切分 + 早停 + 模型保存。

流程：
    加载 (X, y) → 按时间切训练/验证 → 训练 → 验证集早停 → 保存最佳 checkpoint
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import TimeSeriesDataset, temporal_split
from .lstm import LSTMModel
from .metrics import best_threshold, classification_report, precision_recall_f1
from .transformer import TransformerModel

logger = logging.getLogger(__name__)


def build_model(model_type: str, input_size: int, hidden_size: int,
                num_layers: int, dropout: float) -> nn.Module:
    """根据配置创建模型。"""
    if model_type == "lstm":
        return LSTMModel(input_size, hidden_size, num_layers, dropout)
    if model_type == "transformer":
        return TransformerModel(input_size, hidden_size, num_layers, dropout)
    raise ValueError(f"未知模型类型: {model_type}")


def set_seed(seed: int):
    """固定随机种子，保证结果可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fit_model(train_ds, val_ds, cfg: dict, device: str) -> dict:
    """给定训练/验证 Dataset 训练一个模型，返回最佳结果。

    Returns:
        dict: {model, history, best_val_loss, best_threshold, val_f1, val_report}
    """
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch_size", 64), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch_size", 64), shuffle=False)

    # ---- 模型 ----
    model = build_model(
        cfg["type"], int(train_ds.X.shape[-1]),
        cfg.get("hidden_size", 64),
        cfg.get("num_layers", 2),
        cfg.get("dropout", 0.2),
    ).to(device)

    # 样本不平衡（教材 2.4.3）：正样本比例低时用 pos_weight 放大正类损失
    y_train = train_ds.y.numpy()
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = torch.tensor([neg / pos]) if pos > 0 else torch.tensor([1.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    logger.info("类别权重 pos_weight=%.2f（正样本 %.1f%%）", pos_weight.item(),
                pos / len(y_train) * 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3))

    epochs = cfg.get("epochs", 50)
    patience = cfg.get("patience", 8)

    best_val_loss = float("inf")
    best_score = -1.0          # 最佳模型按 F1 选择（类别不平衡下更全面）
    best_state = None
    best_val_probs = None
    best_val_trues = None
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [],
               "val_precision": [], "val_recall": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        # ---------- 训练 ----------
        model.train()
        total_loss, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # 梯度裁剪防爆炸
            optimizer.step()
            total_loss += loss.item() * len(yb)
            n += len(yb)
        train_loss = total_loss / n

        # ---------- 验证 ----------
        val_loss, n = 0.0, 0
        val_trues, val_probs = [], []
        model.eval()
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                prob = torch.sigmoid(logits)
                val_loss += loss.item() * len(yb)
                n += len(yb)
                val_trues.append(yb.cpu().numpy())
                val_probs.append(prob.cpu().numpy())
        val_loss /= n
        y_true = np.concatenate(val_trues)
        y_prob = np.concatenate(val_probs)
        val_acc = float(((y_prob > 0.5).astype(int) == y_true).mean())
        precision, recall, f1 = precision_recall_f1(y_true, (y_prob > 0.5).astype(int))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1)

        if epoch % 10 == 0 or epoch == 1:
            logger.info("epoch %3d | train %.4f | val %.4f | acc %.3f | p %.3f r %.3f f1 %.3f",
                        epoch, train_loss, val_loss, val_acc, precision, recall, f1)

        # 最佳模型按 F1 保存
        if f1 > best_score:
            best_score = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_val_probs, best_val_trues = y_prob, y_true
        # 早停仍按 val_loss（防过拟合）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("早停于 epoch %d（验证集 %d 轮未提升）", epoch, patience)
                break

    model.load_state_dict(best_state)
    # 阈值调优：验证集上找 F1 最优阈值
    best_th, best_f1 = best_threshold(best_val_probs, best_val_trues)
    report = classification_report(best_val_trues, best_val_probs, threshold=best_th)
    logger.info("验证集 F1=%.3f，最优阈值 %.2f（默认 0.5）", best_f1, best_th)
    return {"model": model, "history": history, "best_val_loss": best_val_loss,
            "best_threshold": best_th, "val_f1": best_f1, "val_report": report}


def train_model(X: np.ndarray, y: np.ndarray, cfg: dict,
                val_ratio: float = 0.2,
                device: str = "auto") -> dict:
    """训练入口：按时间切分后调用 _fit_model。

    Args:
        X, y: 特征窗口与标签。
        cfg: 模型配置 dict（model 段）。
        val_ratio: 验证集比例（按时间切尾部）。
        device: auto/cpu/cuda。

    Returns:
        dict: {model, history, best_val_loss, best_threshold, val_f1, val_report}
    """
    # ---- 设备 ----
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用设备: %s", device)

    # ---- 切分 ----
    train_ds, val_ds = temporal_split(X, y, val_ratio)
    return _fit_model(train_ds, val_ds, cfg, device)


def save_checkpoint(model: nn.Module, path: str | Path, cfg: dict,
                    feature_columns: list[str], window: int, horizon: int,
                    best_threshold: float = 0.5,
                    calib: tuple[float, float] | dict | None = None):
    """保存模型权重 + 建模配置，方便后续加载推理。

    calib: Platt 校准。可为 (A, B) 或 dict{'A','B','lo','hi'}（lo/hi 为验证集
    logit 截断区间，预测端先 clip 再 p=sigmoid(A*logit+B)，防分布外极端外推）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": model.state_dict(),
        "model_type": cfg["type"],
        "hidden_size": cfg.get("hidden_size", 64),
        "num_layers": cfg.get("num_layers", 2),
        "dropout": cfg.get("dropout", 0.2),
        "input_size": len(feature_columns),
        "feature_columns": feature_columns,
        "window": window,
        "horizon": horizon,
        "best_threshold": best_threshold,
        "config": cfg,
    }
    if isinstance(calib, dict):
        ckpt["calib"] = {k: float(calib[k]) for k in ("A", "B", "lo", "hi")}
    elif calib is not None:
        ckpt["calib"] = [float(calib[0]), float(calib[1])]
    torch.save(ckpt, path)
    logger.info("模型已保存到 %s（最优阈值 %.2f%s）", path, best_threshold,
                f"，Platt校准 {ckpt.get('calib')}" if "calib" in ckpt else "")


def load_checkpoint(path: str | Path) -> dict:
    """加载 checkpoint 并重建模型。"""
    path = Path(path)
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(
        ckpt["model_type"], ckpt["input_size"],
        ckpt["hidden_size"], ckpt["num_layers"], ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return {"model": model, "meta": ckpt}
