"""因子有效性检验：Rank IC / 平均 IC / ICIR（教材 1.3.2-5）。

核心概念：
    IC = Spearman(Rank(因子值), Rank(未来收益))     # 截面秩相关
    平均 IC = 各期 IC 的均值                          # 因子整体预测力
    ICIR  = 平均 IC / IC 标准差                       # 因子稳定性

判定阈值（教材）：
    |IC| > 0.05  有效因子；  IC > 0.1  优秀因子；  IC < 0.03  弱
    ICIR > 0.5   高质量（预测强且稳定）

注意：IC 检验用「因子值(t) vs 未来收益(t+h)」，是评估预测力的标准做法（非回测交易）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..features.technical import compute_technical

logger = logging.getLogger(__name__)

# 教材判定阈值
IC_EFFECTIVE = 0.05     # |IC| > 0.05 视为有效
IC_EXCELLENT = 0.10     # IC > 0.10 优秀
ICIR_HIGH = 0.50        # ICIR > 0.5 高质量


# ---------- 因子面板构建 ----------
def _prepare_panels(data: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """{symbol: bars} → (close_panel, volume_panel) 宽表（date × symbol）。"""
    closes, volumes = {}, {}
    for s, b in data.items():
        idx = pd.to_datetime(b["date"])
        closes[s] = pd.Series(b["close"].values, index=idx)
        volumes[s] = pd.Series(b["volume"].values, index=idx)
    return (pd.DataFrame(closes).sort_index(),
            pd.DataFrame(volumes).sort_index())


def build_factor_panels(data: dict) -> dict[str, pd.DataFrame]:
    """从日线数据构建多个因子的逐日截面面板 {因子名: DataFrame(date × symbol)}。

    第一版因子集（全部可由价量直接计算，无需额外数据源）：
        动量类：mom_5 / mom_20 / mom_60    过去 N 日涨幅
        反转类：rev_1 / rev_5              过去 N 日涨幅取负（短期反转）
        波动类：vol_20                     20 日对数收益标准差
        量能类：volume_ratio_20            20 日量比
        技术类：ma_dev_20                  收盘价偏离 MA20、rsi_14
    """
    close_panel, vol_panel = _prepare_panels(data)
    logret = np.log(close_panel / close_panel.shift(1))

    f: dict[str, pd.DataFrame] = {}
    f["mom_5"] = close_panel.pct_change(5)
    f["mom_20"] = close_panel.pct_change(20)
    f["mom_60"] = close_panel.pct_change(60)
    f["rev_1"] = -close_panel.pct_change(1)
    f["rev_5"] = -close_panel.pct_change(5)
    f["vol_20"] = logret.rolling(20).std()
    f["volume_ratio_20"] = vol_panel / vol_panel.rolling(20).mean()
    f["ma_dev_20"] = close_panel / close_panel.rolling(20).mean() - 1

    # RSI(14) 逐股计算
    rsi = {}
    for s, b in data.items():
        bars = b.copy()
        bars.index = pd.to_datetime(bars["date"])
        feat = compute_technical(bars, ["rsi"])
        rsi[s] = feat["rsi"]
    f["rsi_14"] = pd.DataFrame(rsi).sort_index()
    return f


def forward_returns(close_panel: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """未来 horizon 日收益面板：close.shift(-h)/close - 1。"""
    return close_panel.shift(-horizon) / close_panel - 1


# ---------- Rank IC 计算 ----------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.sqrt((x ** 2).sum() * (y ** 2).sum()))
    return float((x * y).sum() / denom) if denom else 0.0


def rank_ic_series(factor_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                   min_stocks: int = 10) -> pd.Series:
    """逐日横截面 Rank IC（Spearman 秩相关）序列。

    对每个交易日：取因子值与未来收益都有效的股票，
    分别求秩后算皮尔逊相关（即 Spearman）。
    """
    ics, dates = [], []
    for date in factor_panel.index:
        if date not in ret_panel.index:
            continue
        df = pd.concat([factor_panel.loc[date], ret_panel.loc[date]], axis=1).dropna()
        df.columns = ["f", "r"]
        if len(df) < min_stocks:
            continue
        ic = _pearson(df["f"].rank().values, df["r"].rank().values)
        ics.append(ic)
        dates.append(date)
    return pd.Series(ics, index=dates, name="rank_ic")


def summarize_ic(ic_series: pd.Series) -> dict:
    """汇总单个因子的 IC 统计：平均 IC / ICIR / 正占比 / |IC| 均值。"""
    if ic_series.empty:
        return None
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    icir = mean_ic / std_ic if std_ic else 0.0
    return {
        "mean_ic": round(mean_ic, 4),
        "icir": round(icir, 3),
        "ic_positive": round(float((ic_series > 0).mean()), 3),
        "abs_ic": round(float(ic_series.abs().mean()), 4),
        "n_days": len(ic_series),
    }


def judge_factor(rep: dict) -> str:
    """按教材阈值给因子评级。"""
    if rep is None:
        return "无数据"
    abs_ic = abs(rep["mean_ic"])
    if rep["icir"] > ICIR_HIGH and abs_ic > IC_EFFECTIVE:
        return "高质量有效"
    if abs_ic > IC_EXCELLENT:
        return "优秀"
    if abs_ic > IC_EFFECTIVE:
        return "有效"
    return "弱/无效"


def factor_ic_report(data: dict, horizons: tuple[int, ...] = (5, 20)
                     ) -> pd.DataFrame:
    """对全部因子、多个预测期计算 IC 报告（DataFrame）。

    index = 因子名，列 = {horizon, mean_ic, icir, ic_positive, abs_ic, 判定}
    """
    factors = build_factor_panels(data)
    close_panel, _ = _prepare_panels(data)
    rows = []
    for name, fpanel in factors.items():
        for h in horizons:
            rets = forward_returns(close_panel, h)
            ics = rank_ic_series(fpanel, rets)
            rep = summarize_ic(ics)
            if rep is None:
                continue
            rep["factor"] = name
            rep["horizon"] = h
            rep["judge"] = judge_factor(rep)
            rows.append(rep)
    df = pd.DataFrame(rows)
    df = df[["factor", "horizon", "mean_ic", "icir", "ic_positive",
             "abs_ic", "n_days", "judge"]]
    return df.sort_values(["horizon", "abs_ic"], ascending=[True, False]).reset_index(drop=True)
