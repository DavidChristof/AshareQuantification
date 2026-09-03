"""横截面增强模型 v2 单元测试：相对标签 / 增强特征 / 样本装配 / GBM / 截面评估 / 端到端。

运行：python -m pytest tests/test_cross_model.py -v  或   python tests/test_cross_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.models.cross_dataset import (             # noqa: E402
    build_enhanced_features, make_samples, relative_label_panel,
    split_by_date,
)
from quant.models.cross_model import (               # noqa: E402
    _cross_metrics, _make_gbm, tabular_features,
)


def _bars(n: int = 120, base: float = 10.0, drift: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    ret = rng.normal(drift, 0.02, n)
    close = base * np.exp(np.cumsum(ret))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame({
        "date": idx, "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
        "amount": volume * close,
    })


def _two_stocks():
    return {"000001": _bars(120, drift=0.001, seed=1),
            "000002": _bars(120, drift=-0.001, seed=2)}


def test_relative_label():
    """相对标签：跑赢当日全池中位数者标 1；未来不足 horizon 处为 NaN。"""
    data = _two_stocks()
    y = relative_label_panel(data, horizon=5)
    assert y.shape == (120, 2)
    assert y.index.is_monotonic_increasing
    # 末日（未来不足 5 天）应为 NaN
    assert y.iloc[-1].isna().all()
    # 中间某日：一只 > 中位数 → 恰一 1 一 0（2 只时中位数即两者均值）
    mid = y.iloc[60]
    assert mid.dropna().sum() == 1.0
    # 两边相等时 > 严格，但合成数据很少等值，可断言总和 ≤ 2
    assert mid.sum() <= 2.0


def test_enhanced_features_columns():
    """增强特征含基础 + 截面 + alpha 列，行数与行情一致且无 NaN。"""
    data = _two_stocks()
    feat, cols = build_enhanced_features(data, window=30)
    for s in data:
        assert s in feat
        assert len(feat[s]) == 120
        # 基础 + 截面(rank_*) + alpha 精选
        assert "rank_close" in cols
        assert any(c.startswith("alpha") for c in cols)
        assert any(c.startswith("ma_dev") for c in cols)
        # 截面列应落在 [0,1] 附近（rank pct）
        rc = feat[s]["rank_close"]
        assert rc.max() <= 1.0 and rc.min() >= 0.0
        assert feat[s].isna().sum().sum() == 0


def test_make_samples_and_split():
    """样本窗口形状正确、按日期切分无跨期泄漏。"""
    data = _two_stocks()
    X, y, rel, dates, syms, cols = make_samples(data, window=30, horizon=5)
    assert X.ndim == 3 and X.shape[2] == len(cols)
    assert len(y) == len(dates) == len(syms) == len(rel)
    # 样本决策日 = date
    assert dates[0] <= dates[-1]
    # 标签只在 0/1 之间（可 NaN-free，因为装配已跳过无标签样本）
    assert set(np.unique(y)).issubset({0.0, 1.0})
    # split：训练日期全部早于验证日期
    tr, va = split_by_date(dates, 0.8)
    assert dates[va].min() > dates[tr].max()
    assert len(tr) + len(va) == len(dates)


def test_tabular_features_shape():
    """GBM 输入 = 末帧+均值+标准差+斜率，4F 维。"""
    X = np.random.randn(10, 30, 5).astype(np.float32)
    X2 = tabular_features(X)
    assert X2.shape == (10, 20)


def test_gbm_fit_predict():
    """GBM 工厂能训练并输出概率（lightgbm 或 sklearn 回退）。"""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = _make_gbm({"n_estimators": 50, "max_depth": 3, "num_leaves": 7})
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    assert p.shape == (200,)
    # 应明显优于随机
    from sklearn.metrics import accuracy_score
    assert accuracy_score(y, p > 0.5) > 0.7


def test_cross_metrics_basic():
    """截面评估返回 RankIC / ICIR / Top-Bottom。"""
    dates = np.repeat(np.array(["2020-01-01", "2020-01-02"]), 10)
    rng = np.random.default_rng(0)
    score = np.tile(rng.normal(size=10), 2)
    ret = score * 0.5 + rng.normal(0, 0.1, 20)
    m = _cross_metrics(dates, score, ret, min_n=5)
    assert m["rankic_mean"] > 0.3     # 强正相关应检出正 IC
    assert m["n_days"] == 2
    assert m["top_bottom"] > 0


if __name__ == "__main__":
    import inspect
    fns = [obj for name, obj in inspect.getmembers(sys.modules[__name__])
           if name.startswith("test_") and callable(obj)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} 通过")
    sys.exit(1 if failed else 0)
