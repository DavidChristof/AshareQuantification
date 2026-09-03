"""多因子回归 单元测试：Fama-MacBeth 两步法 + Pooling OLS。

运行：python -m pytest tests/test_factor_regression.py -v  或  python tests/test_factor_regression.py
"""
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.factors.regression import (               # noqa: E402
    _normal_pvalue, _ols, drop_collinear, fama_macbeth, pooled_ols,
)


def _synthetic(horizon=5, n_days=100, n_stocks=30, seed=1):
    """构造已知线性关系的面板：未来收益 ≈ 0.5·f1 − 0.3·f2 + 噪声。

    返回 (f1_panel, f2_panel, close_panel, n_stocks)。
    """
    rng = np.random.default_rng(seed)
    symbols = [f"{i:06d}" for i in range(n_stocks)]
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    f1 = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=symbols)
    f2 = pd.DataFrame(rng.normal(0, 1, (n_days, n_stocks)), index=dates, columns=symbols)

    # 按 modulo-horizon 链递推 close，使未来收益 = 0.5*f1 - 0.3*f2 + e
    e = rng.normal(0, 0.05, (n_days, n_stocks))
    close = pd.DataFrame(np.nan, index=dates, columns=symbols)
    for si, s in enumerate(symbols):
        for k in range(horizon):
            chain = np.arange(k, n_days, horizon)
            if not len(chain):
                continue
            close.iloc[chain[0], si] = 100.0 * (1 + rng.uniform(0, 0.5))
            for j in range(len(chain) - 1):
                t = chain[j]
                r = (0.5 * f1.iloc[t, si] - 0.3 * f2.iloc[t, si]
                     + e[t, si])
                close.iloc[chain[j + 1], si] = close.iloc[chain[j], si] * (1 + r)
    return f1, f2, close, symbols


def _patched(f1, f2, close):
    """mock 掉面板构建，注入已知因子。"""
    stack = ExitStack()
    stack.enter_context(mock.patch("quant.factors.regression.build_factor_panels",
                                   return_value={"f1": f1, "f2": f2}))
    stack.enter_context(mock.patch("quant.factors.regression._prepare_panels",
                                   return_value=(close, None)))
    return stack


def test_ols_recovers_known_coefficients():
    """基础 OLS 能精确恢复线性系数。"""
    rng = np.random.default_rng(0)
    x1 = rng.normal(size=200)
    x2 = rng.normal(size=200)
    y = 0.5 * x1 - 0.3 * x2 + rng.normal(0, 0.05, 200)
    X = np.column_stack([np.ones(200), x1, x2])
    betas, r2, _ = _ols(X, y)
    assert abs(betas[0] - 0.0) < 0.1        # 截距≈0
    assert abs(betas[1] - 0.5) < 0.05
    assert abs(betas[2] + 0.3) < 0.05
    assert 0.0 <= r2 <= 1.0


def test_fama_macbeth_recovers_sign():
    """Fama-MacBeth 恢复已知系数（f1>0、f2<0）。"""
    f1, f2, close, _ = _synthetic()
    with _patched(f1, f2, close):
        res = fama_macbeth({}, horizon=5, factor_names=["f1", "f2"])
    assert not res.empty
    assert res.loc["f1", "mean_beta"] > 0
    assert res.loc["f2", "mean_beta"] < 0
    # 量级接近真实系数（含噪声允许误差）
    assert abs(res.loc["f1", "mean_beta"] - 0.5) < 0.15
    assert abs(res.loc["f2", "mean_beta"] + 0.3) < 0.15
    # 统计推断列齐全、无 NaN
    for col in ["mean_beta", "std_beta", "t_stat", "p_value"]:
        assert res[col].notna().all()
    assert res["mean_r2"].iloc[0] > 0


def test_fama_macbeth_columns_and_pvalue():
    """输出结构 + p 值边界。"""
    f1, f2, close, _ = _synthetic()
    with _patched(f1, f2, close):
        res = fama_macbeth({}, horizon=5, factor_names=["f1", "f2"])
    assert list(res.columns) == ["mean_beta", "std_beta", "t_stat",
                                 "p_value", "significant", "mean_r2", "n_days"]
    assert res["n_days"].iloc[0] > 50        # 有效截面天数充足
    assert (res["p_value"] >= 0).all() and (res["p_value"] <= 1).all()
    # 已知强信号 → f1 应显著
    assert res.loc["f1", "significant"]


def test_pooled_ols_recovers_coefficients():
    """Pooling 合并面板恢复系数 + R² 合理。"""
    f1, f2, close, _ = _synthetic()
    with _patched(f1, f2, close):
        p = pooled_ols({}, horizon=5, factor_names=["f1", "f2"])
    df = p["factor"]
    assert abs(df.loc["f1", "coef"] - 0.5) < 0.05
    assert abs(df.loc["f2", "coef"] + 0.3) < 0.05
    assert 0.0 <= p["r2"] <= 1.0
    assert p["adj_r2"] <= p["r2"] + 1e-9
    assert p["n_obs"] > 500
    assert df["t_stat"].notna().all()


def test_min_stocks_guard():
    """股票数少于 MIN_STOCKS → 返回空/跳过不崩溃。"""
    f1, f2, close, _ = _synthetic(n_stocks=5)   # 5 只 < MIN_STOCKS=10
    with _patched(f1, f2, close):
        res = fama_macbeth({}, horizon=5, factor_names=["f1", "f2"])
    assert res.empty or len(res) == 0


def test_normal_pvalue_bounds():
    """正态近似 p 值：t=0 → 1；|t| 越大 p 越小。"""
    assert abs(_normal_pvalue(0.0) - 1.0) < 1e-9
    p1 = _normal_pvalue(1.0)
    p2 = _normal_pvalue(2.0)
    assert 0 < p2 < p1 < 1
    assert abs(_normal_pvalue(float("nan")) - 1.0) < 1e-9


def test_drop_collinear():
    """完全共线因子（b=-a）被剔除，独立因子保留。"""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2023-01-01", periods=60, freq="D")
    cols = [f"{i:06d}" for i in range(20)]
    base = pd.DataFrame(rng.normal(size=(60, 20)), index=idx, columns=cols)
    panels = {"a": base, "b": -base, "c": pd.DataFrame(rng.normal(size=(60, 20)),
                                                          index=idx, columns=cols)}
    kept, dropped = drop_collinear(panels, ["a", "b", "c"])
    assert "b" in dropped          # b = -a 完全共线 → 剔除
    assert "a" in kept and "c" in kept


if __name__ == "__main__":
    tests = [test_ols_recovers_known_coefficients,
             test_fama_macbeth_recovers_sign,
             test_fama_macbeth_columns_and_pvalue,
             test_pooled_ols_recovers_coefficients,
             test_min_stocks_guard,
             test_normal_pvalue_bounds,
             test_drop_collinear]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
