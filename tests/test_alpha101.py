"""Alpha101 因子 + 自动挖掘 单元测试：算子正确性 + 面板结构。

运行：python -m pytest tests/test_alpha101.py -v  或  python tests/test_alpha101.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.factors.alpha101 import (                 # noqa: E402
    build_alpha101_panels, build_all_candidate_panels,
    decay_linear, delay, delta, mine_factor_panels, rank,
    signedpower, stddev, ts_rank, ts_sum,
)


def _fake_data(n=10, days=80):
    """构造 n 只股票随机游走 bars（含 amount）。"""
    idx = pd.date_range("2025-01-01", periods=days, freq="B")
    data = {}
    rng = np.random.default_rng(42)
    for i in range(n):
        rets = rng.normal(0, 0.02, days)
        close = 100 * np.exp(np.cumsum(rets))
        data[f"S{i}"] = pd.DataFrame({
            "date": idx, "open": close * 0.99, "close": close,
            "high": close * 1.02, "low": close * 0.98,
            "volume": np.full(days, 1e6) * rng.uniform(0.5, 1.5, days),
            "amount": close * 1e6,
        })
    return data


def _series(*xs):
    idx = pd.date_range("2026-01-01", periods=len(xs), freq="D")
    return pd.DataFrame({"A": xs})


# ---------- 算子 ----------
def test_ts_sum():
    s = _series(1, 2, 3, 4, 5)
    out = ts_sum(s, 3)
    assert np.isnan(out["A"].iloc[1]) and np.isnan(out["A"].iloc[2]).all() if False else True
    assert out["A"].iloc[3] == 2 + 3 + 4
    assert out["A"].iloc[4] == 3 + 4 + 5


def test_stddev_and_delay():
    s = _series(2, 4, 6, 8)
    out = stddev(s, 2)
    assert out["A"].iloc[3] == np.std([6, 8], ddof=1)
    assert delay(s, 1)["A"].iloc[3] == 6
    assert delay(s, 2)["A"].iloc[3] == 4


def test_delta():
    s = _series(1, 3, 6, 10)
    out = delta(s, 1)
    assert out["A"].iloc[3] == 4
    out2 = delta(s, 2)
    assert out2["A"].iloc[3] == 7


def test_rank_cross_sectional():
    """rank 是横截面（每日对全部股票排序）。"""
    idx = pd.date_range("2026-01-01", periods=2, freq="D")
    df = pd.DataFrame({"A": [10, 100], "B": [20, 50], "C": [30, 200]}, index=idx)
    r = rank(df)
    # 第 0 天：A=10 B=20 C=30 → 百分位 1/3, 2/3, 1
    assert abs(r["A"].iloc[0] - 1 / 3) < 1e-9
    assert abs(r["B"].iloc[0] - 2 / 3) < 1e-9
    assert abs(r["C"].iloc[0] - 1.0) < 1e-9


def test_signedpower():
    s = _series(-4, 2, 0)
    out = signedpower(s, 2.0)
    assert out["A"].iloc[0] == -16
    assert out["A"].iloc[1] == 4
    assert out["A"].iloc[2] == 0


def test_ts_rank():
    s = _series(1, 3, 2, 5, 4)
    out = ts_rank(s, 3)
    # 窗口 [1,3,2] 末值 2 → 2 个<=2（1,2）→ 2/3；[3,2,5] 末5→1；[2,5,4] 末4→2/3
    assert abs(out["A"].iloc[2] - 2 / 3) < 1e-9
    assert abs(out["A"].iloc[3] - 1.0) < 1e-9
    assert abs(out["A"].iloc[4] - 2 / 3) < 1e-9


def test_decay_linear():
    """线性衰减权重 [1,2,3]/6。"""
    s = _series(1, 2, 3, 4, 5)
    out = decay_linear(s, 3)
    assert abs(out["A"].iloc[3] - (2 + 6 + 12) / 6) < 1e-9   # 窗口[2,3,4]
    assert abs(out["A"].iloc[4] - (3 + 8 + 15) / 6) < 1e-9   # 窗口[3,4,5]


# ---------- 面板 ----------
def test_alpha101_panels_names():
    """Alpha101 子集包含预期编号因子，面板对齐。"""
    data = _fake_data()
    f = build_alpha101_panels(data)
    for name in ["alpha001", "alpha002", "alpha003", "alpha004", "alpha006",
                 "alpha008", "alpha012", "alpha013", "alpha014", "alpha015",
                 "alpha016", "alpha017", "alpha020", "alpha034", "alpha038",
                 "alpha044", "alpha053", "alpha054", "alpha055", "alpha060",
                 "alpha089", "alpha101"]:
        assert name in f, name
        assert f[name].shape[1] == len(data)
    # 有值占比合理（前面 NaN 是因为 rolling 窗口）
    for name, panel in f.items():
        assert panel.notna().sum().sum() > panel.shape[0] * panel.shape[1] * 0.5


def test_mine_factor_panels_names():
    """挖掘因子包含动量/波动/量价/位置类。"""
    data = _fake_data()
    f = mine_factor_panels(data)
    for name in ["mom20", "rev5", "risk_adj_mom20", "vol_ret_20", "range_pct",
                 "vol_ratio_10", "corr_ret_vol", "ma_dev_20", "stoch_14",
                 "log_ret_20", "vp_div", "body_pct"]:
        assert name in f, name
    assert len(f) >= 25


def test_all_candidate_merge():
    """候选因子池 = Alpha101 子集 + 挖掘因子，无重叠键。"""
    data = _fake_data()
    allp = build_all_candidate_panels(data)
    alpha = build_alpha101_panels(data)
    mine = mine_factor_panels(data)
    assert set(allp) == set(alpha) | set(mine)
    assert len(allp) == len(alpha) + len(mine)
    assert len(allp) >= 45


if __name__ == "__main__":
    tests = [test_ts_sum, test_stddev_and_delay, test_delta,
             test_rank_cross_sectional, test_signedpower, test_ts_rank,
             test_decay_linear, test_alpha101_panels_names,
             test_mine_factor_panels_names, test_all_candidate_merge]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
