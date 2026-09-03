"""风险调整指标 单元测试：β / Jensen Alpha / 信息比率 / Treynor / Sortino / 捕获率。

运行：python -m pytest tests/test_risk_adjusted.py -v  或  python tests/test_risk_adjusted.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.backtest.metrics import (                 # noqa: E402
    beta, capture_ratio, information_ratio, jensen_alpha,
    sortino_ratio, treynor_ratio,
)


def _series(values, start=0.0):
    """构造带日期的净值序列（起始净值 100）。"""
    idx = pd.date_range("2023-01-01", periods=len(values), freq="D")
    return pd.Series(values, index=idx)


def test_beta_identical_is_one():
    """完全跟随基准 → β=1。"""
    rng = np.random.default_rng(7)
    rets = rng.normal(0.001, 0.02, 200)
    eq = _series(100 * np.cumprod(1 + rets))
    bench = _series(100 * np.cumprod(1 + rets))
    assert abs(beta(eq, bench) - 1.0) < 1e-6


def test_beta_scaled_double():
    """策略收益 = 2×基准收益 → β≈2。"""
    rng = np.random.default_rng(8)
    rb = rng.normal(0.001, 0.02, 200)
    rp = 2.0 * rb
    eq = _series(100 * np.cumprod(1 + rp))
    bench = _series(100 * np.cumprod(1 + rb))
    assert abs(beta(eq, bench) - 2.0) < 0.05


def test_jensen_alpha_positive_when_outperform():
    """恒定跑赢基准（额外 +0.001/日）→ α>0。"""
    rng = np.random.default_rng(9)
    rb = rng.normal(0.001, 0.02, 200)
    rp = rb + 0.001
    eq = _series(100 * np.cumprod(1 + rp))
    bench = _series(100 * np.cumprod(1 + rb))
    assert jensen_alpha(eq, bench, risk_free=0.02) > 0.0


def test_information_ratio_zero_when_same():
    """与基准完全相同 → 主动收益为 0 → IR=0。"""
    rng = np.random.default_rng(10)
    rets = rng.normal(0.001, 0.02, 200)
    eq = _series(100 * np.cumprod(1 + rets))
    bench = _series(100 * np.cumprod(1 + rets))
    assert abs(information_ratio(eq, bench)) < 1e-6


def test_treynor_matches_formula():
    """Treynor = (年化R_p - R_f) / β。"""
    from quant.backtest.metrics import returns_from_equity

    rng = np.random.default_rng(11)
    rb = rng.normal(0.001, 0.02, 200)
    rp = 1.5 * rb
    eq = _series(100 * np.cumprod(1 + rp))
    bench = _series(100 * np.cumprod(1 + rb))
    b = beta(eq, bench)
    # 用与函数一致的收益序列（pct_change 后首值被丢弃）
    rp_eq = returns_from_equity(eq)
    exp = (rp_eq.mean() * 252 - 0.02) / b
    assert abs(treynor_ratio(eq, bench, 0.02) - exp) < 1e-8


def test_sortino_only_downside():
    """只有上涨 → 下行风险为 0 → Sortino 为 0（不崩溃）。"""
    eq = _series([100, 101, 102, 103, 104])
    assert sortino_ratio(eq, 0.02) == 0.0


def test_capture_ratio_sane():
    """上行捕获>1 且 下行捕获<1 → 涨多跌少。"""
    rng = np.random.default_rng(12)
    rb = rng.normal(0.001, 0.02, 200)
    # 涨时放大、跌时缩小
    rp = np.where(rb > 0, rb * 1.5, rb * 0.5)
    eq = _series(100 * np.cumprod(1 + rp))
    bench = _series(100 * np.cumprod(1 + rb))
    up_cap, down_cap = capture_ratio(eq, bench)
    assert up_cap > 1.0
    assert down_cap < 1.0


def test_zero_variance_guard():
    """基准恒定时 β/Treynor 不崩溃（β=0 保护）。"""
    eq = _series([100, 101, 102, 103, 104])
    bench = _series([100, 100, 100, 100, 100])
    assert beta(eq, bench) == 0.0
    assert treynor_ratio(eq, bench) == 0.0
    # IR 在基准恒定（rb=0）时仍有定义：active=rp，正常计算不崩溃
    information_ratio(eq, bench)


if __name__ == "__main__":
    tests = [test_beta_identical_is_one, test_beta_scaled_double,
             test_jensen_alpha_positive_when_outperform,
             test_information_ratio_zero_when_same, test_treynor_matches_formula,
             test_sortino_only_downside, test_capture_ratio_sane,
             test_zero_variance_guard]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
