"""横截面增强数据装配（模型 v2 数据端）。

为什么需要（对照旧 FeaturePipeline）：
    旧模型学"单股未来绝对涨跌"(acc≈0.5)，但真实任务是"选股 = 横截面相对强弱排序"。
    v2 把三件事对齐到这个任务：

    1. **相对标签** `relative`：y = 未来 horizon 收益是否 > 当日全池中位数（跑赢半数）。
    2. **截面特征**：当日该股在全池的分位（rank_close/rank_volume/rank_ret1/rank_ret5），
       让模型看到"同一天里它相对其他股票的位置"。
    3. **Alpha101/挖掘因子特征**：把 factors.alpha101 的候选面板按精选列并入输入，
       让序列模型直接用上因子研究里证明有效的信号。

防泄漏：滚动 z-score 只用过去 window；截面 rank 只用到当日（决策日 t）横截面；
未来收益仅用于标签（评估）。预测端 build 特征时不产生标签。

约定：所有返回的 feature 表 index = date（datetime），列 = feature_columns（顺序固定）。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..factors.alpha101 import build_all_candidate_panels
from ..features.technical import compute_technical

logger = logging.getLogger(__name__)

# ---------- 基础技术特征（沿用旧 pipeline.feature_columns，去掉 date） ----------
BASE_COLS = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "ma20",
    "close_ma5_ratio", "close_ma10_ratio", "close_ma20_ratio",
    "rsi", "macd_dif", "macd_dea", "macd_hist",
    "volatility", "volume_ratio",
    "ret_1d", "ret_3d", "ret_5d",
]
_INDICATORS = ["ma", "rsi", "macd", "volatility", "volume_ratio", "returns"]

# ---------- 当日截面分位特征（0~1，不做滚动标准化，保留横截面语义） ----------
CROSS_COLS = ["rank_close", "rank_volume", "rank_ret1", "rank_ret5"]

# ---------- Alpha101 + 自动挖掘精选列（覆盖动量/反转/波动/量价/位置） ----------
# 与 17_factor_mine 共享算子库；按已验证显著 + 覆盖度选取，避免维度爆炸。
ALPHA_PICK = [
    "alpha001", "alpha002", "alpha003", "alpha004", "alpha006", "alpha008",
    "alpha012", "alpha014", "alpha034", "alpha038", "alpha053", "alpha101",
]
MINED_PICK = [
    "mom5", "mom20", "mom60", "risk_adj_mom20",
    "vol_ret_5", "vol_ret_20", "range_pct", "range_20",
    "vol_pct_20", "vp_div", "corr_ret_vol",
    "ma_dev_5", "ma_dev_20", "ma_dev_60", "stoch_14",
]


def _to_date_index(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df.index = pd.to_datetime(df["date"])
    df = df.sort_index()
    return df


def _cross_sectional_rank_features(data: dict) -> dict[str, pd.DataFrame]:
    """计算每日横截面分位面板 → {symbol: DataFrame(date×cross_cols)}。"""
    closes = {s: pd.Series(b["close"].values, index=pd.to_datetime(b["date"]))
              for s, b in data.items()}
    vols = {s: pd.Series(b["volume"].values, index=pd.to_datetime(b["date"]))
            for s, b in data.items()}
    close_panel = pd.DataFrame(closes).sort_index()
    vol_panel = pd.DataFrame(vols).sort_index()
    ret1 = close_panel.pct_change(1)
    ret5 = close_panel.pct_change(5)

    def _pct(panel: pd.DataFrame) -> pd.DataFrame:
        return panel.rank(axis=1, pct=True)

    rank_map = {
        "rank_close": _pct(close_panel),
        "rank_volume": _pct(vol_panel),
        "rank_ret1": _pct(ret1),
        "rank_ret5": _pct(ret5),
    }
    out: dict[str, pd.DataFrame] = {}
    for s in data:
        cols = {}
        for col, panel in rank_map.items():
            cols[col] = panel[s]
        out[s] = pd.DataFrame(cols).sort_index()
    return out


def _alpha_features(data: dict) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """计算 Alpha101+挖掘因子面板，精选列按日期并入每股。"""
    panels = build_all_candidate_panels(data)
    want = [c for c in ALPHA_PICK + MINED_PICK if c in panels]
    out: dict[str, pd.DataFrame] = {}
    for s in data:
        cols = {}
        for name in want:
            panel = panels[name]
            if s in panel.columns:
                cols[name] = panel[s]
        out[s] = pd.DataFrame(cols).sort_index()
    return out, want


def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """对每列做滚动 z-score（只用过去 window，防泄漏）。"""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        mean = df[col].rolling(window, min_periods=1).mean()
        std = df[col].rolling(window, min_periods=1).std()
        z = (df[col] - mean) / std.replace(0, np.nan)
        out[col] = z.fillna(0.0)
    return out


def build_enhanced_features(
        data: dict,
        window: int = 30,
        use_cross: bool = True,
        use_alpha: bool = True,
        feature_columns: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """全池增强特征装配。

    Returns:
        (feat_dict, feature_columns)
        feat_dict: {symbol: DataFrame(date index × feature_columns)}，已标准化
        feature_columns: 实际使用的有序特征列
    """
    # 1. 每股基础技术指标（原始值）
    tech = {}
    for s, b in data.items():
        df = _to_date_index(b)
        feat = compute_technical(df, _INDICATORS)
        tech[s] = feat[BASE_COLS]
    logger.info("基础技术特征完成 (%d 只)", len(tech))

    # 2. 截面分位特征（一次算全池）
    cross_map = _cross_sectional_rank_features(data) if use_cross else {}
    # 3. Alpha101/挖掘因子（一次算全池）
    alpha_map, alpha_used = (_alpha_features(data) if use_alpha else ({}, []))

    cross_used = list(cross_map[next(iter(cross_map))].columns) if cross_map else []
    raw_cols = cross_used

    feat_dict: dict[str, pd.DataFrame] = {}
    all_columns = None
    for s in data:
        parts = []
        # base 列滚动标准化
        parts.append(_rolling_zscore(tech[s][BASE_COLS], window))
        # 截面分位列：不做滚动标准化（保留 0~1 横截面语义）
        if raw_cols:
            parts.append(cross_map[s][raw_cols])
        merged = pd.concat(parts, axis=1)
        # Alpha101/挖掘因子（join 后单独滚动标准化）
        if alpha_map and s in alpha_map:
            merged = merged.join(alpha_map[s][alpha_used], how="left")
        for col in alpha_used:
            m = merged[col].rolling(window, min_periods=1).mean()
            st = merged[col].rolling(window, min_periods=1).std()
            merged[col] = ((merged[col] - m) / st.replace(0, np.nan)).fillna(0.0)
        merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        cols = BASE_COLS + raw_cols + alpha_used
        if feature_columns is not None:
            # 预测端：强制与训练列一致（缺失列补 0，多余列丢弃）
            merged = merged.reindex(columns=feature_columns, fill_value=0.0)
            cols = feature_columns
        feat_dict[s] = merged
        if all_columns is None:
            all_columns = list(cols)
    return feat_dict, all_columns or feature_columns or []


# ============================================================
# 相对标签
# ============================================================
def relative_label_panel(data: dict, horizon: int) -> pd.DataFrame:
    """相对标签面板：y(t,s) = 1 若未来 horizon 收益 > 当日全池中位数。

    Returns:
        DataFrame(date × symbol)，未来不足 horizon 或停牌日 = NaN（不产生样本）。
    """
    closes = {s: pd.Series(b["close"].values, index=pd.to_datetime(b["date"]))
              for s, b in data.items()}
    close_panel = pd.DataFrame(closes).sort_index()
    fwd = close_panel.shift(-horizon) / close_panel - 1.0
    med = fwd.median(axis=1)                     # 当日全池未来收益中位数
    y = fwd.gt(med, axis=0).astype(np.float32)   # 每行与当日中位数比较
    y = y.where(fwd.notna())                     # fwd 为 NaN 的位置 → NaN
    return y


def fwd_return_panel(data: dict, horizon: int) -> pd.DataFrame:
    """未来 horizon 连续收益面板（用于 RankIC / 分层评估，非标签）。"""
    closes = {s: pd.Series(b["close"].values, index=pd.to_datetime(b["date"]))
              for s, b in data.items()}
    close_panel = pd.DataFrame(closes).sort_index()
    return close_panel.shift(-horizon) / close_panel - 1.0


def absolute_label_panel(data: dict, horizon: int) -> pd.DataFrame:
    """绝对标签面板（对照用，旧模型语义）：y = 未来 horizon 收益是否 > 0。"""
    fwd = fwd_return_panel(data, horizon)
    y = (fwd > 0).astype(np.float32)
    return y.where(fwd.notna())


# ============================================================
# 样本装配
# ============================================================
def make_samples(
        data: dict, window: int = 30, horizon: int = 5,
        use_cross: bool = True, use_alpha: bool = True,
        feature_columns: list[str] | None = None,
        label_mode: str = "relative",
):
    """全池 → 训练样本 (X, y, rel_ret, dates, symbols)。

    X:      (N, window, F) float32
    y:      (N,) 标签：relative=跑赢全池中位数 / absolute=未来收益>0（对照）
    rel_ret:(N,) 样本未来 horizon 连续收益（评估用，与标签语义无关）
    dates:  (N,) 每个样本的决策日
    symbols:(N,) 每样本对应股票
    """
    feat_dict, fcols = build_enhanced_features(
        data, window, use_cross, use_alpha, feature_columns)
    if label_mode == "absolute":
        y_panel = absolute_label_panel(data, horizon)
    else:
        y_panel = relative_label_panel(data, horizon)
    ret_panel = fwd_return_panel(data, horizon)

    Xs, ys, rs, ds, ss = [], [], [], [], []
    for s, feat in feat_dict.items():
        if s not in y_panel.columns:
            continue
        dates = feat.index
        vals = feat.values.astype(np.float32)   # (T, F)
        yy = y_panel[s].reindex(dates).values
        rr = ret_panel[s].reindex(dates).values
        n = len(vals)
        for i in range(window, n):
            y_i = yy[i]
            if not np.isfinite(y_i):             # 未来不足 horizon / 停牌 → 跳过
                continue
            Xs.append(vals[i - window:i])
            ys.append(y_i)
            rs.append(rr[i] if np.isfinite(rr[i]) else 0.0)
            ds.append(dates[i])
            ss.append(s)

    if not Xs:
        raise RuntimeError("样本装配为空：请检查数据起止日期是否覆盖 window+horizon")
    logger.info("样本装配: X=%s 窗口=%d 特征=%d, 正样本 %.1f%%",
                len(Xs), window, len(fcols), np.mean(ys) * 100)
    return (np.asarray(Xs, dtype=np.float32),
            np.asarray(ys, dtype=np.float32),
            np.asarray(rs, dtype=np.float32),
            np.asarray(ds, dtype="datetime64[ns]"),
            np.asarray(ss), fcols)


def split_by_date(dates: np.ndarray, ratio: float = 0.8):
    """按日期切分（val 全在 train 之后），避免跨期泄漏。

    Returns:
        (train_idx, val_idx)
    """
    uniq = np.unique(dates)
    cut = uniq[int(len(uniq) * ratio)]
    train_idx = np.where(dates <= cut)[0]
    val_idx = np.where(dates > cut)[0]
    return train_idx, val_idx
