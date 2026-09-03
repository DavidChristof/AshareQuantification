"""因子中性化 单元测试：风格面板构建 + 残差剔除风格暴露 + 报告结构。

运行：python -m pytest tests/test_factor_neutralize.py -v  或  python tests/test_factor_neutralize.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.factors.analysis import build_factor_panels  # noqa: E402
from quant.factors.neutralize import (               # noqa: E402
    build_style_panels, neutralize_factor, neutralize_report,
)


def _fake_data(n=12, days=80):
    """构造 n 只股票随机游走 bars（含 close×volume 成交额）。"""
    idx = pd.date_range("2025-01-01", periods=days, freq="B")
    data = {}
    rng = np.random.default_rng(42)
    for i in range(n):
        rets = rng.normal(0, 0.02, days)
        close = 100 * np.exp(np.cumsum(rets))
        data[f"S{i}"] = pd.DataFrame({
            "date": idx, "open": close, "close": close,
            "high": close * 1.01, "low": close * 0.99,
            "volume": np.full(days, 1e6) * rng.uniform(0.5, 1.5),
        })
    return data


def test_build_style_panels():
    """风格面板：含 size_proxy/vol_20，列=股票数，index 对齐。"""
    data = _fake_data()
    styles = build_style_panels(data)
    assert set(styles) == {"size_proxy", "vol_20"}
    for name, panel in styles.items():
        assert panel.shape[1] == len(data)
        assert panel.index.is_monotonic_increasing
        # 滚动窗口后前 19 天为 NaN，其余有值
        assert panel.notna().sum().sum() > panel.shape[0] * panel.shape[1] * 0.7


def test_neutralize_removes_style():
    """因子 = 2×size + 噪声（纯风格暴露）→ 中性化后与 size 相关≈0。"""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    syms = [f"S{i}" for i in range(15)]
    size = pd.DataFrame(rng.normal(size=(60, 15)), index=idx, columns=syms)
    noise = rng.normal(0, 0.05, (60, 15))
    factor = 2.0 * size.values + noise
    factor_panel = pd.DataFrame(factor, index=idx, columns=syms)
    neut = neutralize_factor(factor_panel, {"size": size})

    # 中性化后某日残差与 size 横截面相关 ≈ 0
    row = neut.dropna(how="all").iloc[10]
    size_row = size.loc[neut.dropna(how="all").index[10]]
    valid = row.notna() & size_row.notna()
    corr = float(np.corrcoef(row[valid], size_row[valid])[0, 1])
    assert abs(corr) < 0.2
    # 残差方差 << 原始方差（风格部分被移除）
    orig_std = factor_panel.std(axis=0).mean()
    neut_std = neut.std(axis=0).mean()
    assert neut_std < orig_std


def test_neutralize_independent_factor_kept():
    """因子与风格无关（纯随机）→ 中性化后残差与原始高度相关（不被破坏）。"""
    rng = np.random.default_rng(2)
    idx = pd.date_range("2025-01-01", periods=60, freq="B")
    syms = [f"S{i}" for i in range(15)]
    style = pd.DataFrame(rng.normal(size=(60, 15)), index=idx, columns=syms)
    factor_panel = pd.DataFrame(rng.normal(size=(60, 15)), index=idx, columns=syms)
    neut = neutralize_factor(factor_panel, {"size": style})
    # 相关接近 1（随机因子基本不被风格解释；截距中心化使相关略降）
    mask = factor_panel.notna() & neut.notna()
    fv = factor_panel.values[mask.values]
    nv = neut.values[mask.values]
    corr = float(np.corrcoef(fv, nv)[0, 1])
    assert corr > 0.85


def test_neutralize_panel_shape():
    """中性化面板与原始同 shape；样本不足日保持 NaN。"""
    data = _fake_data(days=40)
    factors = build_factor_panels(data)
    styles = build_style_panels(data)
    name = "mom_5"
    neut = neutralize_factor(factors[name], styles)
    assert neut.shape == factors[name].shape
    assert list(neut.index) == list(factors[name].index)
    assert list(neut.columns) == list(factors[name].columns)


def test_neutralize_report_structure():
    """报告：列齐全；vol_20 不作为被中性化因子；含 survive 判定。"""
    data = _fake_data(days=80)
    rep = neutralize_report(data, horizons=(5, 20))
    assert {"factor", "horizon", "ic_raw", "ic_neutralized",
            "delta_ic", "survive"}.issubset(rep.columns)
    assert "vol_20" not in set(rep["factor"])       # 风格变量不自我中性化
    assert rep["horizon"].isin([5, 20]).all()
    # 8 个被检验因子 × 2 期；mom_60(60日窗口) 在 horizon=20 时可能样本不足被跳过
    assert len(rep) >= 14


if __name__ == "__main__":
    tests = [test_build_style_panels, test_neutralize_removes_style,
             test_neutralize_independent_factor_kept,
             test_neutralize_panel_shape, test_neutralize_report_structure]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
