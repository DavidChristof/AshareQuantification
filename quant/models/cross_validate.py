"""时序 K 折交叉验证：评估模型泛化稳定性，替代单次时序切分。

为什么不能随机 K 折（教材）：
    - 随机打乱会把未来样本放进训练集 → **信息泄漏**，指标虚高
    - 时序数据必须按时间顺序切分：训练永远在验证之前

做法（类 sklearn TimeSeriesSplit）：
    - 把样本按时间顺序切 K 折，fold i 用前 i 段训练、第 i+1 段验证
    - 每折独立训练 + 输出 acc/P/R/F1 + 最优阈值
    - 汇总 K 折均值 ± 标准差 → 指标不再依赖单一切分点

注意：本模块**不保存任何模型**，纯粹评估泛化能力。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .dataset import TimeSeriesDataset
from .trainer import _fit_model, set_seed

logger = logging.getLogger(__name__)


def time_series_split(n: int, n_splits: int, gap: int = 0,
                      test_size: int | None = None):
    """按时间顺序切 K 折（训练恒在验证之前，无泄漏）。

    Args:
        n: 样本总数
        n_splits: 折数
        gap: 训练与验证之间的间隔（可选，进一步防泄漏）
        test_size: 每折验证集大小（默认 n//(n_splits+1)，与 sklearn 一致）

    Yields:
        (train_idx, valid_idx)：均为递增的整数数组
    """
    if test_size is None:
        test_size = n // (n_splits + 1)
    if test_size <= 0:
        return
    for i in range(n_splits):
        train_end = (i + 1) * test_size
        valid_start = train_end + gap
        valid_end = min(valid_start + test_size, n)
        if train_end <= 0 or valid_start >= n:
            continue
        yield (np.arange(0, train_end),
               np.arange(valid_start, valid_end))


def cross_validate(X: np.ndarray, y: np.ndarray, cfg: dict,
                   n_splits: int = 5, device: str = "auto",
                   seed: int = 42) -> dict:
    """时序 K 折交叉验证。

    Args:
        X, y: 特征窗口与标签（保持时间顺序）。
        cfg: 模型配置 dict（model 段）。
        n_splits: 折数。
        device: auto/cpu/cuda。

    Returns:
        dict: {folds: DataFrame, mean: Series, std: Series}
    """
    if device == "auto":
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    logger.info("使用设备: %s · 时序 %d 折交叉验证", device, n_splits)

    rows = []
    for i, (tr_idx, va_idx) in enumerate(time_series_split(len(y), n_splits)):
        set_seed(seed + i)                      # 每折不同种子，避免巧合
        train_ds = TimeSeriesDataset(X[tr_idx], y[tr_idx])
        val_ds = TimeSeriesDataset(X[va_idx], y[va_idx])
        logger.info("[fold %d/%d] 训练 %d · 验证 %d",
                    i + 1, n_splits, len(tr_idx), len(va_idx))
        res = _fit_model(train_ds, val_ds, cfg, device)
        rep = res["val_report"]
        rows.append({
            "fold": i + 1,
            "train_n": len(tr_idx),
            "valid_n": len(va_idx),
            "pos_pct": round(float(y[tr_idx].mean()) * 100, 1),
            "accuracy": rep["accuracy"],
            "precision": rep["precision"],
            "recall": rep["recall"],
            "f1": rep["f1"],
            "threshold": rep["threshold"],
        })
        logger.info("[fold %d/%d] acc %.3f | p %.3f r %.3f f1 %.3f | th %.2f",
                    i + 1, n_splits, rep["accuracy"], rep["precision"],
                    rep["recall"], rep["f1"], rep["threshold"])

    df = pd.DataFrame(rows)
    cols = ["accuracy", "precision", "recall", "f1"]
    return {
        "folds": df,
        "mean": df[cols].mean(),
        "std": df[cols].std(),
    }


def _format_report(result: dict) -> str:
    """把 cross_validate 结果格式化成可读文本。"""
    df = result["folds"]
    if df.empty:
        return "（无有效折）"
    lines = ["时序 K 折交叉验证（每折独立训练，无未来泄漏）",
             df.to_string(index=False),
             ""]
    lines.append("均值 ± 标准差：")
    for col in ["accuracy", "precision", "recall", "f1"]:
        m = result["mean"][col]
        s = result["std"][col]
        lines.append(f"  {col:9s} = {m:.3f} ± {s:.3f}")
    lines.append("")
    # 诊断：accuracy≈0.5 但 recall≈1 → F1 高是「几乎全预测上涨」的假象
    mean_acc = result["mean"]["accuracy"]
    mean_rec = result["mean"]["recall"]
    if mean_acc < 0.52 and mean_rec > 0.9:
        lines.append("!! 诊断：accuracy≈0.5 但 recall≈1 → F1 高来自『几乎全预测为上涨』(阈值过低)，")
        lines.append("    模型无真实分类能力，F1 虚高。应结合 accuracy/precision 而非单看 F1。")
    else:
        lines.append("判定：std 越小泛化越稳定；accuracy 明显高于 0.5 且有正 F1 才有实际预测力")
    return "\n".join(lines)
