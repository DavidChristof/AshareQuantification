"""时序 K 折交叉验证 单元测试：切分无泄漏 + 结构正确 + 小规模可跑通。

运行：python -m pytest tests/test_cross_validate.py -v  或  python tests/test_cross_validate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402

from quant.models.cross_validate import (           # noqa: E402
    cross_validate, time_series_split,
)


def test_split_no_leakage():
    """训练恒在验证之前（无未来泄漏），验证段连续递增。"""
    n = 100
    splits = list(time_series_split(n, n_splits=4))
    assert len(splits) == 4
    prev_end = None
    for tr, va in splits:
        assert len(tr) > 0 and len(va) > 0
        assert tr[-1] < va[0]                 # 关键：训练结束 < 验证开始
        if prev_end is not None:
            assert va[0] == prev_end          # 验证段连续（上一折末尾接本折开头）
        assert list(va) == list(range(va[0], va[0] + len(va)))
        prev_end = va[-1] + 1


def test_split_expanding_train():
    """训练集随时间扩张（fold 递增时训练样本增多）。"""
    n = 100
    splits = list(time_series_split(n, n_splits=4))
    sizes = [len(tr) for tr, _ in splits]
    assert sizes == sorted(sizes)
    assert len(sizes) == len(set(sizes))       # 每折训练量不同（扩张式）


def test_split_small_n():
    """样本太少时优雅降级（不崩、无重叠）。"""
    splits = list(time_series_split(n=8, n_splits=5))
    for tr, va in splits:
        assert tr[-1] < va[0]


def test_split_gap():
    """gap>0 时训练与验证之间留空（进一步防泄漏）。"""
    tr, va = next(time_series_split(100, n_splits=4, gap=3))
    assert va[0] - tr[-1] == 4               # 中间留 3 个样本间隔


def test_cross_validate_small_run():
    """极小配置下 2 折能跑通并返回结构（快）。"""
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 10, 4)).astype(np.float32)
    y = (rng.random(n) > 0.5).astype(np.float32)
    cfg = {"type": "lstm", "hidden_size": 8, "num_layers": 1, "dropout": 0.0,
           "lr": 1e-2, "epochs": 2, "patience": 1, "batch_size": 64}
    res = cross_validate(X, y, cfg, n_splits=2, device="cpu", seed=7)
    assert len(res["folds"]) == 2
    assert {"fold", "accuracy", "precision", "recall", "f1",
            "threshold"}.issubset(res["folds"].columns)
    assert set(res["mean"].index) == {"accuracy", "precision", "recall", "f1"}
    assert (res["folds"]["accuracy"] >= 0).all() and (res["folds"]["accuracy"] <= 1).all()


if __name__ == "__main__":
    tests = [test_split_no_leakage, test_split_expanding_train,
             test_split_small_n, test_split_gap,
             test_cross_validate_small_run]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
