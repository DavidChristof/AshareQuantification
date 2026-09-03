"""因子 IC / ICIR 检验 单元测试：Rank IC 正确性 + 面板/报告结构。

运行：python -m pytest tests/test_factor_ic.py -v  或  python tests/test_factor_ic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quant.factors.analysis import (                    # noqa: E402
    build_factor_panels, factor_ic_report, forward_returns,
    rank_ic_series, summarize_ic,
)


def _fake_data(n=10, days=60):
    """构造 n 只股票随机游走的 bars。"""
    idx = pd.date_range("2025-01-01", periods=days, freq="B")
    data = {}
    rng = np.random.default_rng(42)
    for i in range(n):
        rets = rng.normal(0, 0.02, days)
        close = 100 * np.exp(np.cumsum(rets))
        data[f"S{i}"] = pd.DataFrame({
            "date": idx, "open": close, "close": close,
            "high": close * 1.01, "low": close * 0.99,
            "volume": np.full(days, 1e6),
        })
    return data


def test_rank_ic_perfect_positive():
    """因子值与未来收益完全同序 → Rank IC ≈ +1。"""
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    syms = [f"S{i}" for i in range(10)]
    base = np.arange(10.0)
    f = pd.DataFrame(np.tile(base, (3, 1)), index=idx, columns=syms)
    ics = rank_ic_series(f, f.copy())
    assert abs(ics.iloc[0] - 1.0) < 1e-6


def test_rank_ic_perfect_negative():
    """因子值与未来收益完全反序 → Rank IC ≈ -1。"""
    idx = pd.date_range("2026-01-01", periods=3, freq="B")
    syms = [f"S{i}" for i in range(10)]
    base = np.arange(10.0)
    f = pd.DataFrame(np.tile(base, (3, 1)), index=idx, columns=syms)
    ics = rank_ic_series(f, -f.copy())
    assert abs(ics.iloc[0] - (-1.0)) < 1e-6


def test_summarize_ic():
    """汇总：平均 IC 计算正确；常数序列 std=0 时 ICIR 视为 0。"""
    ics = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1])
    rep = summarize_ic(ics)
    assert rep["mean_ic"] == 0.1
    assert rep["ic_positive"] == 1.0
    assert rep["icir"] == 0.0   # std=0 → ICIR 无法计算，置 0


def test_build_factor_panels_columns():
    """因子面板：包含全部因子名。"""
    data = _fake_data()
    factors = build_factor_panels(data)
    for name in ["mom_5", "mom_20", "mom_60", "rev_1", "rev_5",
                 "vol_20", "volume_ratio_20", "ma_dev_20", "rsi_14"]:
        assert name in factors, name
        assert factors[name].shape[1] == 10


def test_factor_ic_report_structure():
    """报告结构：含判定列；每因子至少有一期数据。"""
    data = _fake_data(days=80)
    rep = factor_ic_report(data, horizons=(5, 20))
    assert {"factor", "horizon", "mean_ic", "icir", "judge"}.issubset(rep.columns)
    assert rep["horizon"].isin([5, 20]).all()
    assert len(rep) >= 9   # 9 个因子 × 至少一期
    assert set(rep["factor"]) == set(
        ["mom_5", "mom_20", "mom_60", "rev_1", "rev_5",
         "vol_20", "volume_ratio_20", "ma_dev_20", "rsi_14"])


if __name__ == "__main__":
    tests = [test_rank_ic_perfect_positive, test_rank_ic_perfect_negative,
             test_summarize_ic, test_build_factor_panels_columns,
             test_factor_ic_report_structure]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
