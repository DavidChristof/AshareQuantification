"""Alpha101 因子库：WorldQuant 经典 101 个 alpha 因子（A 股子集 + 算子库）。

受 FinHack（github.com/FinHackCN/finhack）启发——它内置 Alpha101/Alpha191
因子集。我们实现：

    1. **算子库**：Alpha101 的核心是算子组合（rank/ts_rank/delay/delta/
       correlation/decay_linear/signedpower/ts_argmax...），全部纯 pandas/numpy 实现。
       - 时序算子对 DataFrame 逐列（每只股票）沿时间轴滚动
       - 横截面算子 `rank` 沿 axis=1（每交易日截面排序）

    2. **Alpha101 因子子集**：从原 101 个中选出仅依赖 OHLCV/amount 的
       代表性因子（约 20 个），标注原编号。vwap = amount/volume（真实成交均价）。

约定：因子面板均为 DataFrame(date × symbol)，与 analysis.build_factor_panels 一致，
可直接复用现有的 rank_ic / 多因子回归 / 中性化流水线。

参考：WorldQuant Alpha101 因子公式（Kahler & Maslov, 2016）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import _prepare_panels


# ============================================================
# 数据面板
# ============================================================
def _ohlcv_panels(data: dict) -> dict[str, pd.DataFrame]:
    """{symbol: bars} → {open, high, low, close, volume, amount, vwap} 面板。"""
    cols = {k: {} for k in ("open", "high", "low", "close", "volume", "amount")}
    for s, b in data.items():
        idx = pd.to_datetime(b["date"])
        for c in cols:
            cols[c][s] = pd.Series(b[c].values, index=idx)
    panels = {k: pd.DataFrame(v).sort_index() for k, v in cols.items()}
    panels["vwap"] = (panels["amount"] / panels["volume"].replace(0, np.nan))
    return panels


# ============================================================
# 算子库（Alpha101 核心）
# ============================================================
def ts_sum(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).sum()


def sma(df: pd.DataFrame, n: int, m: int) -> pd.DataFrame:
    return df.ewm(alpha=m / n, adjust=False).mean()


def stddev(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).std()


def ts_max(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).max()


def ts_min(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.rolling(n).min()


def ts_argmax(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 窗口内最大值出现的位置（0-based，最右优先）。"""
    return df.rolling(n).apply(lambda x: np.argmax(x), raw=True)


def ts_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 窗口内最新值的百分位排名 [0,1]（等价 rolling rank pct 末值）。"""
    return df.rolling(n).apply(lambda x: (x <= x[-1]).mean(), raw=True)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).corr(y)


def ts_cov(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n).cov(y)


def delay(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.shift(n)


def delta(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.diff(n)


def returns(df: pd.DataFrame) -> pd.DataFrame:
    return df.pct_change()


def rank(df: pd.DataFrame) -> pd.DataFrame:
    """横截面排名（每日对全部股票），返回 [0,1] 百分位。"""
    return df.rank(axis=1, pct=True)


def scale(df: pd.DataFrame, a: float = 1.0) -> pd.DataFrame:
    """横截面缩放到绝对值和为 a（de-mean 后再 scale）。"""
    demeaned = df.sub(df.mean(axis=1), axis=0)
    denom = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return demeaned.div(denom, axis=0) * a


def sign(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素符号：正→1、负→-1、零/NaN 保留。"""
    s = df.copy()
    s[s > 0] = 1.0
    s[s < 0] = -1.0
    return s


def signedpower(df: pd.DataFrame, a: float) -> pd.DataFrame:
    """带符号幂：sign(x)*|x|^a（NaN 保留）。"""
    return sign(df) * df.abs().pow(a)


def decay_linear(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """线性衰减加权移动平均（权重 1..n 归一化）。"""
    w = np.arange(1, n + 1, dtype=float)
    w /= w.sum()
    return df.rolling(n).apply(lambda x: np.dot(x, w), raw=True)


def ts_sum_returns(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """n 日累乘收益（约等于 sum of returns）。"""
    return (1 + df.pct_change()).rolling(n).apply(np.prod, raw=True) - 1


# ============================================================
# Alpha101 因子子集（仅依赖 OHLCV/amount）
# ============================================================
def build_alpha101_panels(data: dict) -> dict[str, pd.DataFrame]:
    """从日线数据计算 Alpha101 因子子集面板 {alpha#: DataFrame(date×symbol)}。"""
    P = _ohlcv_panels(data)
    open_, high, low, close = P["open"], P["high"], P["low"], P["close"]
    vol, vwap = P["volume"], P["vwap"]
    ret = close.pct_change()
    adv20 = vol.rolling(20).mean()

    f: dict[str, pd.DataFrame] = {}
    # Alpha#001: rank(ts_argmax(signedpower(((returns<0)?stddev(returns,20):close),2),5)) - 0.5
    cond = ret < 0
    std20 = ret.rolling(20).std()
    base = cond.where(cond, std20)          # returns<0 → stddev(returns,20)，否则 close
    f["alpha001"] = rank(ts_argmax(signedpower(base, 2.0), 5)) - 0.5

    # Alpha#002: -1*cov? 原式 correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)
    f["alpha002"] = -1 * ts_corr(
        rank(delta(np.log(vol.clip(lower=1e-9)), 2)),
        rank((close - open_) / open_), 6)

    # Alpha#003: -1*correlation(rank(open), rank(volume), 10)
    f["alpha003"] = -1 * ts_corr(rank(open_), rank(vol), 10)

    # Alpha#004: -1*ts_rank(rank(low), 9)
    f["alpha004"] = -1 * ts_rank(rank(low), 9)

    # Alpha#006: -1*correlation(open, volume, 10)
    f["alpha006"] = -1 * ts_corr(open_, vol, 10)

    # Alpha#008: -1*rank(ts_sum(open,5)*ts_sum(returns,5) - delay(ts_sum(open,5)*ts_sum(returns,5),10))
    t5 = ts_sum(open_, 5) * ts_sum(ret, 5)
    f["alpha008"] = -1 * rank(t5 - delay(t5, 10))

    # Alpha#012: sign(delta(volume,1)) * (-1*delta(close,1))
    f["alpha012"] = sign(delta(vol, 1)) * (-1 * delta(close, 1))

    # Alpha#013: -1*rank(covariance(rank(close), rank(volume), 5))
    f["alpha013"] = -1 * rank(ts_cov(rank(close), rank(vol), 5))

    # Alpha#014: -1*rank(delta(returns,3))*correlation(open,volume,10)
    f["alpha014"] = -1 * rank(delta(ret, 3)) * ts_corr(open_, vol, 10)

    # Alpha#015: -1*sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    f["alpha015"] = -1 * (rank(ts_corr(rank(high), rank(vol), 3))
                          + rank(ts_corr(rank(high), rank(vol), 3)).shift(1)
                          + rank(ts_corr(rank(high), rank(vol), 3)).shift(2))

    # Alpha#016: -1*rank(covariance(rank(high), rank(volume), 5))
    f["alpha016"] = -1 * rank(ts_cov(rank(high), rank(vol), 5))

    # Alpha#017: -1*rank(ts_rank(close,10))*rank(delta(delta(close,1),1))*rank(ts_rank(volume/adv20,5))
    f["alpha017"] = -1 * rank(ts_rank(close, 10)) * rank(delta(delta(close, 1), 1)) \
        * rank(ts_rank(vol / adv20, 5))

    # Alpha#020: -1*(rank(open-delay(high,1))*rank(open-delay(close,1))*rank(open-delay(low,1)))
    f["alpha020"] = -1 * (rank(open_ - delay(high, 1)) * rank(open_ - delay(close, 1))
                          * rank(open_ - delay(low, 1)))

    # Alpha#034: rank(2*rank(rank(1)+ts_rank(rank(close-delay(close,1)/close),1)) + rank(rank(close-delay(close,1)/close))) - 1
    dclose = (close - delay(close, 1)) / close
    f["alpha034"] = rank(2 * (rank(1 + ts_rank(rank(dclose), 1))) + rank(rank(dclose))) - 1

    # Alpha#038: -1*rank(ts_rank(close,10))*rank(close/open)
    f["alpha038"] = -1 * rank(ts_rank(close, 10)) * rank(close / open_)

    # Alpha#044: -1*rank((ts_rank((close-delay(close,1))/close,3)*ts_rank(close,2))*rank(volume/adv20))
    f["alpha044"] = -1 * rank(
        ts_rank((close - delay(close, 1)) / close, 3) * ts_rank(close, 2)
        * rank(vol / adv20))

    # Alpha#053: -1*delta(((close-low)-(high-close))/(close-low), 9)
    inner = ((close - low) - (high - close)) / (close - low).replace(0, np.nan)
    f["alpha053"] = -1 * delta(inner, 9)

    # Alpha#054: -1*rank(delta(1/(open/close-1),3))/rank((close-delay(close,1))/close)
    f["alpha054"] = -1 * rank(delta(1 / (open_ / close - 1), 3)) \
        / rank((close - delay(close, 1)) / close)

    # Alpha#055: -1*correlation(rank((close-ts_min(low,12))/(ts_max(high,12)-ts_min(low,12))), rank(volume), 6)
    stoch = (close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)).replace(0, np.nan)
    f["alpha055"] = -1 * ts_corr(rank(stoch), rank(vol), 6)

    # Alpha#060: -1*rank(ts_sum(delta(volume,1),5)/adv20)*rank(close/delay(close,1)-1)
    f["alpha060"] = -1 * rank(ts_sum(delta(vol, 1), 5) / adv20) \
        * rank(close / delay(close, 1) - 1)

    # Alpha#089: sma(close,13,1)-sma(close,27,1)-sma(sma(close,13,1)-sma(close,27,1),10,1)
    s1, s2 = sma(close, 13, 1), sma(close, 27, 1)
    f["alpha089"] = s1 - s2 - sma(s1 - s2, 10, 1)

    # Alpha#101: (close-open)/((high-low)+0.001)
    f["alpha101"] = (close - open_) / ((high - low) + 0.001)

    # 去无限值
    for name, panel in f.items():
        f[name] = panel.replace([np.inf, -np.inf], np.nan)
    return f


# ============================================================
# 自动因子挖掘：从 OHLCV 用算子自动组合候选因子
# ============================================================
def mine_factor_panels(data: dict) -> dict[str, pd.DataFrame]:
    """自动挖掘候选因子：动量/反转/波动/量价/位置/乖离等组合。"""
    P = _ohlcv_panels(data)
    open_, high, low, close = P["open"], P["high"], P["low"], P["close"]
    vol, amount, vwap = P["volume"], P["amount"], P["vwap"]
    ret = close.pct_change()

    f: dict[str, pd.DataFrame] = {}
    # ---- 动量 / 反转 ----
    for n in (5, 10, 20, 60):
        f[f"mom{n}"] = close / delay(close, n) - 1
        f[f"rev{n}"] = delay(close, n) / close - 1           # 反转 = -mom
    # 风险调整动量
    f["risk_adj_mom20"] = (close / delay(close, 20) - 1) / ret.rolling(20).std()

    # ---- 波动 / 振幅 ----
    for n in (5, 10, 20):
        f[f"vol_ret_{n}"] = ret.rolling(n).std()
    f["range_pct"] = (high - low) / close                     # 当日振幅
    f["range_20"] = (ts_max(high, 20) - ts_min(low, 20)) / close   # 20日区间
    f["body_pct"] = (close - open_).abs() / open_             # 实体占比
    f["upper_shadow"] = (high - pd.concat([open_, close], axis=1).max(axis=1)) / close
    f["lower_shadow"] = (pd.concat([open_, close], axis=1).min(axis=1) - low) / close

    # ---- 量价 ----
    adv10 = vol.rolling(10).mean()
    f["vol_ratio_10"] = vol / adv10                           # 量比
    f["amount_ratio_20"] = amount / amount.rolling(20).mean()  # 额比
    f["corr_ret_vol"] = ts_corr(ret, vol.pct_change(), 10)     # 量价相关性
    f["corr_ret_amount"] = ts_corr(ret, amount.pct_change(), 10)
    f["vol_pct_20"] = vol / vol.rolling(20).mean() - 1
    f["vp_div"] = close / vwap - 1                            # 收盘 vs 成交均价

    # ---- 位置 / 乖离 ----
    for n in (5, 20, 60):
        f[f"ma_dev_{n}"] = close / sma(close, n, 1) - 1        # 均线乖离
    f["stoch_14"] = (close - ts_min(low, 14)) / (ts_max(high, 14) - ts_min(low, 14))
    f["log_ret_20"] = np.log(close / delay(close, 20))

    for name, panel in f.items():
        f[name] = panel.replace([np.inf, -np.inf], np.nan)
    return f


def build_all_candidate_panels(data: dict) -> dict[str, pd.DataFrame]:
    """合并 Alpha101 子集 + 自动挖掘因子，作为候选因子池。"""
    panels = {}
    panels.update(build_alpha101_panels(data))
    panels.update(mine_factor_panels(data))
    return panels
