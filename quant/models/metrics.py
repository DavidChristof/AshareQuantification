"""分类模型评价指标（教材 2.4）：准确率 / 精确率 / 召回率 / F1 / 混淆矩阵。

样本不平衡场景（教材 2.4.3）：目标类别占比低时，单独准确率无意义，
必须结合精确率(Precision)、召回率(Recall)综合评价。
"""
from __future__ import annotations

import numpy as np


def confusion_counts(y_true, y_pred) -> tuple[int, int, int, int]:
    """返回 (tp, fp, tn, fn)——教材 2.4.2 四类基础结果。"""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, fp, tn, fn


def precision_recall_f1(y_true, y_pred) -> tuple[float, float, float]:
    """精确率(预测上涨中真涨的比例) / 召回率(真涨中被抓住的比例) / F1。"""
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def classification_report(y_true, y_prob, threshold: float = 0.5) -> dict:
    """完整分类报告：准确率 / 精确率 / 召回率 / F1 / 混淆矩阵。"""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob > threshold).astype(int)
    acc = float((y_pred == y_true).mean())
    p, r, f1 = precision_recall_f1(y_true, y_pred)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    return {
        "accuracy": round(acc, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def best_threshold(y_prob, y_true, grid=None) -> tuple[float, float]:
    """在验证集上遍历阈值，返回 F1 最优的 (阈值, F1)。

    类别不平衡时，0.5 阈值往往不是最优——通过调优阈值提升 F1。
    """
    y_prob = np.asarray(y_prob).ravel()
    y_true = np.asarray(y_true).ravel()
    if grid is None:
        grid = np.arange(0.30, 0.71, 0.02)
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        p, r, f1 = precision_recall_f1(y_true, (y_prob > t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    return best_t, best_f1
