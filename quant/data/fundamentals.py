"""时点(point-in-time)财务指标：把按**报告期**的季度财务数据，变成**某一天真正看得到**的 PE/ROE。

## 为什么不能直接用

`data/fundamentals.db` 里的数据有两个陷阱，直接用会**偷看未来**：

1. **数值是「年内累计」的**。600519 2020 年：Q1 11.04 → H1 19.05 → Q3 28.54 → 全年 39.42。
   所以 `eps_diluted` 是**年初至今累计 EPS**，不是单季 EPS。算 TTM 必须先去累计再滚动加总。
   （ROE 同理，也是累计口径。）
2. **报告期 ≠ 公告日**。2020-06-30 的半年报，8 月底才公告。若在 7 月就用它，就是未来函数。

## 本模块的做法

- **去累计**：Q1 用原值；Q2/Q3/Q4 用它减去上一期（同年同季报口径）。
- **TTM**：最近 4 个单季之和。
- **披露滞后**：按**法定披露截止日**把报告期映射到"最早可见日"——
  Q1→04-30、H1→08-31、Q3→10-31、年报→次年 04-30。
  用截止日（而非实际公告日）是**保守**做法：只会晚用、绝不会早用 ⇒ 结构上不可能有未来函数。
- **`asof(date)`**：取 `可见日 <= date` 的**最新**一期。**绝不**先 ffill 到每日再 shift。

## 用法

    from quant.data.fundamentals import load_financials, pit_pe_roe_panels
    fin = load_financials(con)                       # 去累计后的单季 + 可见日
    pe, roe = pit_pe_roe_panels(fin, dates, symbols) # date × symbol 面板（无未来函数）
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime

import numpy as np
import pandas as pd

# 报告期 (月, 日) → 法定披露截止日（月, 日, 跨年偏移）
_DEADLINE = {
    (3, 31): (4, 30, 0),      # 一季报：4/30 前
    (6, 30): (8, 31, 0),      # 半年报：8/31 前
    (9, 30): (10, 31, 0),     # 三季报：10/31 前
    (12, 31): (4, 30, 1),     # 年报：次年 4/30 前
}


def disclosure_date(report_date) -> date | None:
    """报告期 → 该报告**最早可以合法看到**的日期（法定披露截止日）。"""
    d = _as_date(report_date)
    if d is None:
        return None
    key = (d.month, d.day)
    if key not in _DEADLINE:
        return None
    m, dd, yoff = _DEADLINE[key]
    return date(d.year + yoff, m, dd)


def _as_date(v) -> date | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def prepare_financials(df: pd.DataFrame) -> pd.DataFrame:
    """**纯函数**：原始财务表 → 加 `avail_date`（可见日）+ 去累计后的单季值。

    输入列需含：symbol, report_date, eps_diluted, eps_weighted, roe, roe_weighted
    Returns: symbol, report_date, avail_date, eps_q, roe_q, eps_ytd, roe_ytd
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    df["report_date"] = pd.to_datetime(df["report_date"])
    df = df.sort_values(["symbol", "report_date"]).reset_index(drop=True)
    # 优先用摊薄 EPS（拿不到时回退加权）
    df["eps_ytd"] = df["eps_diluted"].where(df["eps_diluted"].notna(), df["eps_weighted"])
    df["roe_ytd"] = df["roe"].where(df["roe"].notna(), df["roe_weighted"])

    # ---- 去累计：年内按季度顺序做差分；跨年（Q1）重置 ----
    df["_yr"] = df["report_date"].dt.year
    df["_q"] = df["report_date"].dt.quarter
    for src, dst in (("eps_ytd", "eps_q"), ("roe_ytd", "roe_q")):
        prev = df.groupby(["symbol", "_yr"])[src].shift(1)
        same_year = df.groupby(["symbol", "_yr"]).cumcount() > 0
        df[dst] = np.where(same_year, df[src] - prev, df[src])
    df["avail_date"] = df["report_date"].map(disclosure_date)
    out = df[["symbol", "report_date", "avail_date", "eps_q", "roe_q",
              "eps_ytd", "roe_ytd"]]
    return out.dropna(subset=["avail_date"]).reset_index(drop=True)


def load_financials(con: sqlite3.Connection) -> pd.DataFrame:
    """读 `data/fundamentals.db` → `prepare_financials` 的结果。"""
    df = pd.read_sql_query(
        "SELECT symbol, report_date, eps_diluted, eps_weighted, roe, roe_weighted "
        "FROM financials", con)
    return prepare_financials(df)


def ttm_panel(fin: pd.DataFrame, col: str, dates, symbols,
              window: int = 4) -> pd.DataFrame:
    """把单季值滚动 `window` 期求和 → 面板（date × symbol），按**可见日**对齐。

    - `window=4`（默认）→ TTM（滚动四个单季之和）
    - `window=1` → **最近可见的单期值**。线上 `_fetch_roe` 取的就是这个
      （`stock_financial_analysis_indicator` 的最后一期「净资产收益率」，
      **是年内累计口径、不是 TTM**）⇒ 忠实复刻线上选股时必须用 `window=1`。

    ⚠️ 对齐方式：对每个交易日 t，取 `avail_date <= t` 的**最新一期**，
    再取该期往前 `window` 期的和。绝不把整表先 ffill 到每日再 shift。
    """
    if fin.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(dates), columns=symbols, dtype="float64")
    sub = fin[fin["symbol"].isin(symbols)][["symbol", "report_date", "avail_date", col]]
    sub = sub.sort_values(["symbol", "report_date"])
    sub["ttm"] = (sub.groupby("symbol")[col]
                     .rolling(window, min_periods=window).sum()
                     .reset_index(level=0, drop=True))
    sub = sub.dropna(subset=["ttm"])
    if sub.empty:
        return pd.DataFrame(index=pd.DatetimeIndex(dates), columns=symbols, dtype="float64")

    out = np.full((len(dates), len(symbols)), np.nan)
    idx = {s: i for i, s in enumerate(symbols)}
    # 统一成 datetime.date 再比 —— 注意 pd.Timestamp **也是** datetime.date 的子类，
    # 用 isinstance 分流会漏掉它，随后 Timestamp 与 date 比较会抛 TypeError（踩过一次）。
    dts = [pd.Timestamp(d).date() for d in dates]
    for sym, g in sub.groupby("symbol"):
        j = idx.get(sym)
        if j is None:
            continue
        av = list(g["avail_date"])
        tv = list(g["ttm"])
        k = -1
        for i, t in enumerate(dts):
            while k + 1 < len(av) and av[k + 1] <= t:
                k += 1
            if k >= 0:
                out[i, j] = tv[k]
    return pd.DataFrame(out, index=pd.DatetimeIndex(dates), columns=symbols)


def pit_pe_roe_panels(fin: pd.DataFrame, close: pd.DataFrame,
                      symbols: list[str] | None = None):
    """→ (pe, roe) 两个 date × symbol 面板（**无未来函数**）。

    pe(t)  = close(t) / EPS_ttm(t)
    roe(t) = 最近可见的 TTM ROE（单季 ROE 滚动 4 期之和；A股口径下 ≈ 年化 ROE）

    ⚠️ **`close` 必须传不复权价**（`universe_pit.load_raw_close`），不能传 qfq 前复权价。

    原因：EPS 是 as-reported 的（报告期当时的股本口径，未经追溯重述），而 **qfq 价
    已被"今天之后发生的所有送转与分红"折算过**。两者相除，历史 PE 系统性偏低，
    送转前可差到 4 倍。实测 2020-04-30 截面：两种口径的 PE 排序相关仅 0.938，
    按送转分层低到 0.862 —— 而 PE 是选股主因子，排序错 = 选股错。

    （`roe` 不受影响：它是比值，不含价格项。）
    """
    dates = close.index
    syms = symbols or list(close.columns)
    eps_ttm = ttm_panel(fin, "eps_q", dates, syms)
    roe_ttm = ttm_panel(fin, "roe_q", dates, syms)
    pe = close[syms].div(eps_ttm.reindex(columns=syms).replace(0, np.nan))
    pe = pe.where(pe > 0)                       # 亏损（EPS<=0）→ NaN，与 selector 一致（MIN_PE）
    return pe, roe_ttm
