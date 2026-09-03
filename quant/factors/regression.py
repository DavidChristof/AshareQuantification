"""多因子回归：检验多个因子对未来收益的**联合解释力**与**边际显著性**。

与单因子 IC 检验的区别（教材 2.4/因子检验进阶）：
    - 单因子 IC 只能看「某个因子单独预测力」，因子间有共线性时会互相干扰
    - 多因子回归把全部因子放进同一模型，控制其他因子后看**每个因子的边际贡献**
      （系数是否显著 ≠ 0），避免把相关因子误判为独立有效

两种标准做法（均纯 numpy 手写 OLS，无额外依赖）：

1. Fama-MacBeth 两步法（横截面回归）：
   - 第一步：每个截面日期 t，回归 y(t+h) ~ 因子(t)，得到当日系数 β_t
   - 第二步：对 β_t 时间序列求均值 = 因子长期载荷，t 统计量 = mean/(std/√T)
   - 优点：允许系数随时间变化，t 统计量天然反映时变稳定性

2. 面板 Pooling OLS（合并所有日期×股票）：
   - 把所有观测合并成一个长表做一次 OLS，看整体拟合优度 R² 和系数显著性
   - 优点：样本量大、给出 R²（因子集整体解释力）

注意：本模块与 IC 检验一样用「因子值(t) vs 未来收益(t+h)」评估预测力，
是因子检验的学术标准做法（非回测交易）。
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from .analysis import _prepare_panels, build_factor_panels, forward_returns

logger = logging.getLogger(__name__)

MIN_STOCKS = 10    # 单截面回归最少股票数
COLLINEAR_THRESH = 0.999   # 因子对 |相关| 超过此值判为完全共线 → 剔除冗余


# ---------- 共线性处理 ----------
def drop_collinear(factor_panel: dict[str, pd.DataFrame],
                   factor_names: list[str],
                   corr_thresh: float = COLLINEAR_THRESH) -> tuple[list[str], list[str]]:
    """剔除成对完全共线的冗余因子（如 rev_5 = -mom_5）。

    多重共线性会使 X'X 病态/奇异，导致回归系数与标准误不可靠。
    返回 (保留的因子, 被剔除的因子)。
    """
    kept, dropped = [], []
    for name in factor_names:
        new = factor_panel[name]
        for k in kept:
            # 按共同 (date × symbol) 展平成逐观测对，算皮尔逊相关
            ci = new.index.intersection(factor_panel[k].index)
            cc = new.columns.intersection(factor_panel[k].columns)
            a = new.loc[ci, cc].values.ravel()
            b = factor_panel[k].loc[ci, cc].values.ravel()
            mask = np.isfinite(a) & np.isfinite(b)
            if mask.sum() < MIN_STOCKS:
                continue
            c = abs(float(np.corrcoef(a[mask], b[mask])[0, 1]))
            if c > corr_thresh:
                dropped.append(name)
                break
        else:
            kept.append(name)
    return kept, dropped


# ---------- OLS 基础 ----------
def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """最小二乘回归（带截距）。返回 (betas, R², SSE)。"""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    betas, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ betas
    sse = float(((y - y_hat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - sse / sst if sst > 0 else 0.0
    return betas, r2, sse


def _normal_pvalue(t: float) -> float:
    """双尾正态近似 p 值（用 erf，不依赖 scipy）。"""
    if not np.isfinite(t):
        return 1.0
    return float(1.0 - math.erf(abs(t) / math.sqrt(2.0)))


# ---------- Fama-MacBeth ----------
def _cross_sectional_fit(factor_panel: dict[str, pd.DataFrame], ret_panel: pd.DataFrame,
                         date, factor_names: list[str]) -> tuple[np.ndarray | None, float]:
    """单日横截面回归：y(未来收益) ~ 各因子 + 截距。返回 (betas, R²) 或 (None, 0)。"""
    cols = {"ret": ret_panel.loc[date]}
    for name in factor_names:
        if date in factor_panel[name].index:
            cols[name] = factor_panel[name].loc[date]
    df = pd.DataFrame(cols).dropna()
    if len(df) < MIN_STOCKS:
        return None, 0.0
    y = df["ret"].values
    X = df[factor_names].values
    X = np.column_stack([np.ones(len(X)), X])
    betas, r2, _ = _ols(X, y)
    return betas, r2


def fama_macbeth(data: dict, horizon: int = 5,
                 factor_names: list[str] | None = None,
                 factor_panels: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Fama-MacBeth 两步法多因子回归。

    Args:
        data: {symbol: bars}
        horizon: 未来收益预测期（日）
        factor_names: 参与回归的因子名（默认用全部 build_factor_panels 因子）
        factor_panels: 可选的外部因子面板（如 Alpha101/挖掘因子池）；
                       不传则内部用 build_factor_panels

    Returns:
        DataFrame（index=截距+各因子）列：mean_beta / std_beta / t_stat /
        p_value / significant(α=0.05) / 整体 mean_r2 / n_days
    """
    factor_panel = (factor_panels if factor_panels is not None
                    else build_factor_panels(data))
    close_panel, _ = _prepare_panels(data)
    if factor_names is None:
        factor_names = list(factor_panel)
    # 剔除完全共线因子（如 rev_5 = -mom_5），避免 X'X 病态
    factor_names, dropped = drop_collinear(factor_panel, factor_names)
    ret_panel = forward_returns(close_panel, horizon)

    # 第一步：逐截面日期回归，收集 β_t 序列 + R²_t 序列
    beta_rows, r2_rows, dates = [], [], []
    for date in ret_panel.index:
        betas, r2 = _cross_sectional_fit(factor_panel, ret_panel, date, factor_names)
        if betas is None:
            continue
        beta_rows.append(betas)
        r2_rows.append(r2)
        dates.append(date)
    if not beta_rows:
        return pd.DataFrame()

    beta_mat = np.array(beta_rows)          # (T × k+1)
    r2_arr = np.array(r2_rows)
    n_days = len(beta_mat)

    # 第二步：对 β_t 序列做统计推断
    rows = []
    labels = ["intercept"] + factor_names
    for j, label in enumerate(labels):
        col = beta_mat[:, j]
        mean_b = float(col.mean())
        std_b = float(col.std(ddof=1)) if n_days > 1 else 0.0
        t_stat = mean_b / (std_b / math.sqrt(n_days)) if std_b > 0 else 0.0
        rows.append({
            "term": label,
            "mean_beta": round(mean_b, 4),
            "std_beta": round(std_b, 4),
            "t_stat": round(t_stat, 3),
            "p_value": round(_normal_pvalue(t_stat), 4),
            "significant": bool(t_stat != 0 and _normal_pvalue(t_stat) < 0.05),
        })
    df = pd.DataFrame(rows).set_index("term")
    df["mean_r2"] = round(float(r2_arr.mean()), 4)
    df["n_days"] = n_days
    df.attrs["dropped"] = dropped
    return df


# ---------- 面板 Pooling OLS ----------
def pooled_ols(data: dict, horizon: int = 5,
               factor_names: list[str] | None = None,
               factor_panels: dict[str, pd.DataFrame] | None = None) -> dict:
    """合并所有日期×股票观测做一次 OLS。

    Args:
        data: {symbol: bars}
        horizon: 未来收益预测期（日）
        factor_names: 参与回归的因子名
        factor_panels: 可选的外部因子面板

    Returns:
        dict: factor(DataFrame: term/coef/se/t_stat/p_value/significant)、
              r2、adj_r2、n_obs、n_factors
    """
    factor_panel = (factor_panels if factor_panels is not None
                    else build_factor_panels(data))
    close_panel, _ = _prepare_panels(data)
    if factor_names is None:
        factor_names = list(factor_panel)
    # 剔除完全共线因子，避免 X'X 病态
    factor_names, dropped = drop_collinear(factor_panel, factor_names)
    ret_panel = forward_returns(close_panel, horizon)

    frames = []
    for date in ret_panel.index:
        cols = {"ret": ret_panel.loc[date]}
        for name in factor_names:
            if date in factor_panel[name].index:
                cols[name] = factor_panel[name].loc[date]
        df = pd.DataFrame(cols).dropna()
        if len(df) >= MIN_STOCKS:
            frames.append(df)
    if not frames:
        return {}
    long = pd.concat(frames)

    y = long["ret"].values
    X = np.column_stack([np.ones(len(long)), long[factor_names].values])
    n, k = X.shape
    betas, r2, sse = _ols(X, y)

    # 系数标准误：se(β) = sqrt(σ² · (X'X)^{-1})，用伪逆兜底秩亏
    dof = n - k
    sigma2 = sse / dof if dof > 0 else 0.0
    try:
        cov = sigma2 * np.linalg.pinv(X.T @ X)
        # 数值误差可能使对角线出现极小的负值 → clip 后 sqrt
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    t_stats = np.where(np.isfinite(se) & (se > 0), betas / np.where(se > 0, se, 1.0), 0.0)

    rows = []
    labels = ["intercept"] + factor_names
    for j, label in enumerate(labels):
        t = float(t_stats[j])
        rows.append({
            "term": label,
            "coef": round(float(betas[j]), 4),
            "se": round(float(se[j]), 4) if np.isfinite(se[j]) else float("nan"),
            "t_stat": round(t, 3),
            "p_value": round(_normal_pvalue(t), 4),
            "significant": bool(t != 0 and _normal_pvalue(t) < 0.05),
        })
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k) if n > k else r2
    return {
        "factor": pd.DataFrame(rows).set_index("term"),
        "r2": round(float(r2), 4),
        "adj_r2": round(float(adj_r2), 4),
        "n_obs": n,
        "n_factors": len(factor_names),
        "dropped_factors": dropped,
    }


# ---------- 综合报告 ----------
def multi_factor_report(data: dict, horizons: tuple[int, ...] = (5, 20),
                        factor_names: list[str] | None = None) -> dict:
    """对多个预测期分别输出 Fama-MacBeth 与 Pooling 报告。

    Returns:
        {"fama_macbeth": {horizon: DataFrame}, "pooled": {horizon: dict}}
    """
    return {
        "fama_macbeth": {h: fama_macbeth(data, h, factor_names) for h in horizons},
        "pooled": {h: pooled_ols(data, h, factor_names) for h in horizons},
    }


def _format_report(report: dict) -> str:
    """把 multi_factor_report 结果格式化成可读文本。"""
    lines = []
    for h, fm in report["fama_macbeth"].items():
        lines.append(f"\n===== Fama-MacBeth 多因子回归（预测期 {h} 日） =====")
        if fm.empty:
            lines.append("（有效截面样本不足）")
            continue
        dropped = fm.attrs.get("dropped", [])
        if dropped:
            lines.append(f"!! 已剔除完全共线因子：{dropped}")
        lines.append(f"平均 R^2 = {fm['mean_r2'].iloc[0]} · 有效截面 {fm['n_days'].iloc[0]} 天")
        sig = fm[fm["significant"]]
        lines.append(fm.reset_index().to_string(index=False))
        lines.append(f"\nα=0.05 显著因子：{list(sig.index)}" if len(sig)
                     else "\n无显著因子")
    for h, p in report["pooled"].items():
        lines.append(f"\n===== Pooling OLS（预测期 {h} 日，合并面板） =====")
        if not p:
            lines.append("（样本不足）")
            continue
        dropped = p.get("dropped_factors", [])
        if dropped:
            lines.append(f"!! 已剔除完全共线因子：{dropped}")
        lines.append(f"R^2={p['r2']} · adj_R^2={p['adj_r2']} · 观测 {p['n_obs']} 个")
        sig = p["factor"][p["factor"]["significant"]]
        lines.append(p["factor"].reset_index().to_string(index=False))
        lines.append(f"\nα=0.05 显著因子：{list(sig.index)}" if len(sig)
                     else "\n无显著因子")
    return "\n".join(lines)
