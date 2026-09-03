"""因子中性化：剔除因子中的风格暴露（规模/波动率），检验残差的「纯」预测力。

背景（教材「因子中性化」）：
    一个因子的 IC 显著，可能只是因为它是某个风格的代理——
    比如「高动量股」同时「市值小/波动高」，动量因子的预测力
    其实是暴露在规模/波动风格上。中性化 = 把因子对风格变量做
    横截面回归，**取残差**作为「纯因子」，再检验其 IC。

    - 若中性化后 IC 大幅下降 → 原预测力来自风格暴露（伪因子）
    - 若中性化后 IC 基本保留 → 因子有独立于风格的 alpha

风格变量（本机数据可得，纯价量计算）：
    - size_proxy：ln(20 日均成交额) = ln(close×volume 的 20 日均值) —— 规模/流动性代理
    - vol_20：20 日对数收益波动率 —— 波动率风格

注意：vol_20 既是风格变量也是候选因子，故不作为「被中性化因子」
（对它自己回归会得到 0 残差，无意义）；它作为风格被保留。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .analysis import (_prepare_panels, build_factor_panels, forward_returns,
                       rank_ic_series, summarize_ic)

logger = logging.getLogger(__name__)

MIN_STOCKS = 10
# 中性化后 IC 存活阈值：|IC| 仍 ≥ 0.03 视为保留一定独立预测力
SURVIVE_IC = 0.03


# ---------- 风格变量 ----------
def build_style_panels(data: dict, size_window: int = 20) -> dict[str, pd.DataFrame]:
    """构建风格变量面板 {size_proxy, vol_20}（date × symbol）。"""
    close_panel, volume_panel = _prepare_panels(data)
    amount = close_panel * volume_panel                     # 近似成交额（无 amount 列也可用）
    size_proxy = np.log(amount.rolling(size_window).mean())
    logret = np.log(close_panel / close_panel.shift(1))
    vol_20 = logret.rolling(size_window).std()
    return {"size_proxy": size_proxy, "vol_20": vol_20}


# ---------- 中性化 ----------
def neutralize_factor(factor_panel: pd.DataFrame,
                      style_panels: dict[str, pd.DataFrame],
                      min_stocks: int = MIN_STOCKS) -> pd.DataFrame:
    """逐日横截面回归 factor ~ 风格变量，返回残差面板（中性化后的纯因子）。

    Args:
        factor_panel: 待中性化因子面板（date × symbol）
        style_panels: 风格变量面板 {name: DataFrame(date × symbol)}

    Returns:
        同 shape 的残差面板；某日样本不足则该日全 NaN。
    """
    styles = list(style_panels)
    out = pd.DataFrame(index=factor_panel.index, columns=factor_panel.columns,
                       dtype=float)
    for date in factor_panel.index:
        cols = {"f": factor_panel.loc[date]}
        ok = True
        for s in styles:
            if date in style_panels[s].index:
                cols[s] = style_panels[s].loc[date]
            else:
                ok = False
                break
        if not ok:
            continue
        df = pd.DataFrame(cols).dropna()
        if len(df) < min_stocks:
            continue
        X = np.column_stack([np.ones(len(df))] + [df[s].values for s in styles])
        y = df["f"].values
        betas, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ betas
        out.loc[date, df.index] = resid
    return out


# ---------- 报告 ----------
def neutralize_report(data: dict, horizons: tuple[int, ...] = (5, 20)
                      ) -> pd.DataFrame:
    """每个因子 × 预测期：原始 IC vs 中性化后 IC + 变化判定。

    Returns:
        DataFrame（factor/horizon/ic_raw/ic_neutralized/icir_raw/
        icir_neutralized/delta_ic/survive）
    """
    factors = build_factor_panels(data)
    styles = build_style_panels(data)
    close_panel, _ = _prepare_panels(data)
    rows = []
    for name, fpanel in factors.items():
        if name in styles:          # vol_20 是风格变量，不自我中性化
            continue
        neut = neutralize_factor(fpanel, styles)
        for h in horizons:
            rets = forward_returns(close_panel, h)
            raw = summarize_ic(rank_ic_series(fpanel, rets))
            nt = summarize_ic(rank_ic_series(neut, rets))
            if raw is None or nt is None:
                continue
            rows.append({
                "factor": name,
                "horizon": h,
                "ic_raw": raw["mean_ic"],
                "ic_neutralized": nt["mean_ic"],
                "icir_raw": raw["icir"],
                "icir_neutralized": nt["icir"],
                "delta_ic": round(nt["mean_ic"] - raw["mean_ic"], 4),
                "survive": bool(abs(nt["mean_ic"]) >= SURVIVE_IC),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["horizon", "ic_raw"],
                            ascending=[True, False]).reset_index(drop=True)
    return df


def _format_report(report: pd.DataFrame) -> str:
    if report.empty:
        return "（无有效数据）"
    lines = []
    for h, grp in report.groupby("horizon"):
        lines.append(f"\n===== 因子中性化（预测期 {h} 日） =====")
        lines.append("列：ic_raw=原始IC · ic_neutralized=剔除风格后IC · "
                     "delta=变化 · survive=是否保留独立预测力")
        lines.append(grp.to_string(index=False))
        survived = grp[grp["survive"]]
        lines.append(f"\n中性化后仍有效：{list(survived['factor'])}" if len(survived)
                     else "\n中性化后全部失效（预测力来自风格暴露）")
    return "\n".join(lines)
