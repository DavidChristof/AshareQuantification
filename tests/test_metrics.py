"""模型分类指标 单元测试：Precision / Recall / F1 / 混淆矩阵 / 阈值调优。

运行：python -m pytest tests/test_metrics.py -v  或  python tests/test_metrics.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.models.metrics import (       # noqa: E402
    best_threshold, classification_report, confusion_counts,
    precision_recall_f1,
)


def test_confusion_and_prf():
    """教材 2.4.2：四类结果 + P/R/F1 计算正确。"""
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    rep = classification_report(y_true, y_prob, threshold=0.5)
    assert rep["confusion"] == {"tp": 3, "fp": 1, "tn": 6, "fn": 0}
    assert abs(rep["precision"] - 0.75) < 1e-6     # 3/4
    assert abs(rep["recall"] - 1.0) < 1e-6          # 3/3
    assert abs(rep["f1"] - 0.857142857) < 1e-4
    assert rep["accuracy"] == 0.9


def test_prf_zero_division():
    """无正样本/无预测正类时 P/R/F1 为 0（不崩溃）。"""
    p, r, f1 = precision_recall_f1([0, 0, 0], [1, 1, 1])   # 全假阳
    assert p == 0.0 and r == 0.0 and f1 == 0.0
    p, r, f1 = precision_recall_f1([0, 0, 0], [0, 0, 0])   # 全真阴
    assert p == 0.0 and r == 0.0 and f1 == 0.0


def test_best_threshold_improves_f1():
    """类别不平衡时调优阈值可提升 F1。"""
    y_true = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    y_prob = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    # 阈值 0.5 时 F1≈0.857；提高到 >0.5 后 FP 消失 → F1=1.0
    t, f1 = best_threshold(y_prob, y_true)
    assert t > 0.5
    assert abs(f1 - 1.0) < 1e-6


if __name__ == "__main__":
    tests = [test_confusion_and_prf, test_prf_zero_division,
             test_best_threshold_improves_f1]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
