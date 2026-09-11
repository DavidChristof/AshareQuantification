"""时点(point-in-time)宇宙重建：按中证指数公开编制规则，消掉回测的幸存者偏差。

## 为什么需要它

`docs/2026-09-11-survivorship-bias.md` 证实：老的 559 池是用**今天的**中证500/1000
成分名单回填历史的，池内流通市值加权 +107.7% 而真实中证500 只有 +12.9%（超额 +97.9pp），
且偏差**正好挂在流动性轴上**（2021 年冷门股平均 +124.7% vs 已活跃 +26.8%）。
⇒ 回测要可信，宇宙必须是"**在那个时点可观察到的**"。

## 编制规则（中证指数官网口径）

**中证500(000905)**：① 剔除沪深300样本及过去一年日均总市值前 300；
② 剩余按过去一年日均成交金额由高到低**剔除后 20%**；③ 剩余按日均总市值取前 500。
**中证1000(000852)**：① 剔除**中证800**(沪深300+中证500)及日均总市值前 300；
② 同样剔除成交额后 20%；③ 剩余按日均总市值取前 1000。
**调整**：每半年（6/12 月**第二个星期五的下一交易日**）；缓冲区：成交额前 90% 老样本
可参与市值排名、市值 800 名前新样本优先进入、1200 名前老样本优先保留。

## 本实现的近似（**已在 docs 写明**）

| 项 | 处理 |
|---|---|
| 日均**总**市值 | 用**流通市值**代理（`full_daily.float_mcap = amount/turnover`）。总股本拿不到（东财接口本机不通）。实测用前复权价算市值会偏 3%~11.5%，故必须用 amount/turnover。 |
| 缓冲区 | **v1 未实现**（接口已留 `PitRules.buffer_*`）。 |
| ST 过滤 | **v1 未实现**（无历史 ST 数据）。ST 股换手低，多会被"成交额后 20%"自动剔掉。 |
| 上市满 1 年 | 用"上市以来交易日数 ≥ `min_list_days`"代理。 |
| 沪深300 | 重建近似（成交额剔后20% → 市值前 300），仅供 500/1000 剔除用。 |

## 无未来函数的两条硬规则

1. **窗口端点 `date <= cutoff`**，cutoff = 第二个星期五；生效日 = **下一交易日**。
   cutoff < 生效日，所以用截止日收盘的数据不算偷看未来。
2. **快照只往前看**：任一时点 t 用的是 `review_date <= t` 的**最新**快照，
   **绝不**把整库快照先 ffill 到每日再 shift。

## 用法

    from quant.data.universe_pit import PitRules, build_snapshots, build_mask
    rules = PitRules()
    snaps = build_snapshots(con, trading_days, "2020-01-01", "2026-09-11", rules)
    mask  = build_mask(con, ("csi500", "csi1000"), dates, symbols)
"""
from __future__ import annotations

import calendar
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# 三个指数的标识（同时是 pit_members.index_code 的取值）
HS300, CSI500, CSI1000 = "hs300", "csi500", "csi1000"

# 北交所前缀：中证500/1000 的样本空间只含沪深两市
_BJ_PREFIX = ("92", "8", "4")


@dataclass(frozen=True)
class PitRules:
    """编制规则的参数（全部可按敏感性扫描调整）。"""
    window: int = 250                  # "过去一年" ≈ 250 个交易日
    min_days_in_window: int = 120      # 窗口内至少要有多少天数据（停牌/新股）
    min_list_days: int = 60            # "上市满一个季度"的交易日数
    exclude_top_mcap: int = 300        # 剔除日均市值前 N
    cut_bottom_pct: float = 0.20       # 剔除日均成交额后 N%
    hs300_size: int = 300
    csi500_size: int = 500
    csi1000_size: int = 1000
    exclude_bj: bool = True
    # ---- 预留（v1 默认不启用，接口先留好）----
    buffer_new_top: int | None = None       # 新样本优先进入的市值排名阈值
    buffer_old_top: int | None = None       # 老样本优先保留的市值排名阈值


# ============================================================
# 纯函数：调仓日
# ============================================================
def second_friday(year: int, month: int) -> date:
    """某年某月的第二个星期五。"""
    c = calendar.Calendar()
    fridays = [d for d in c.itermonthdates(year, month)
               if d.month == month and d.weekday() == 4]
    if len(fridays) < 2:
        raise ValueError(f"{year}-{month} 不足两个星期五")
    return fridays[1]


def semi_annual_cutoffs(years: Iterable[int]) -> list[date]:
    """各年 6 月与 12 月的第二个星期五（= 计算用的**数据截止日**）。"""
    out: list[date] = []
    for y in years:
        out.append(second_friday(y, 6))
        out.append(second_friday(y, 12))
    return sorted(out)


def next_trading_day(d: date, trading_days: Sequence) -> date | None:
    """**严格大于** d 的第一个交易日。

    trading_days 来自数据库里实际出现过的日期（含节假日顺延），
    绝不要用 `d + timedelta(days=1)` —— 会撞上周末/端午/国庆。
    """
    for t in trading_days:
        td = t.date() if hasattr(t, "date") else (t if isinstance(t, date) else None)
        if td is None:
            td = datetime.strptime(str(t)[:10], "%Y-%m-%d").date()
        if td > d:
            return td
    return None


def review_effective_dates(cutoffs: Sequence[date], trading_days: Sequence) -> dict[date, date]:
    """{截止日(第二个周五): 生效日(下一交易日)}。"""
    out: dict[date, date] = {}
    for c in cutoffs:
        nxt = next_trading_day(c, trading_days)
        if nxt is not None:
            out[c] = nxt
    return out


# ============================================================
# 纯函数：选样
# ============================================================
def cut_bottom_by_amount(amt: pd.Series, pct: float) -> pd.Index:
    """按成交额**分位**剔除后 pct（返回**保留**的索引）。

    用 rank(pct=True) 而不是分位数阈值，抗并列值。
    注意分母是**传进来的这个 Series**——调用方必须先做前一步的剔除，
    再调用本函数（用全集分母是常见错误）。
    """
    if not len(amt):
        return amt.index
    return amt.index[amt.rank(pct=True) > pct]


def pick_members(mcap: pd.Series, amt: pd.Series, rules: PitRules,
                 exclude: set[str], size: int) -> list[str]:
    """在 `exclude` 之外，按规则选 `size` 只（成交额剔后 20% → 市值取前 size）。"""
    cand = [s for s in mcap.index if s not in exclude]
    if not cand:
        return []
    sub_mcap, sub_amt = mcap[cand], amt[cand]
    keep = cut_bottom_by_amount(sub_amt, rules.cut_bottom_pct)
    if not len(keep):
        return []
    ranked = sub_mcap[keep].sort_values(ascending=False)
    return list(ranked.index[:size])


def build_snapshot(stats: pd.DataFrame, rules: PitRules,
                   list_dates: dict[str, str] | None = None,
                   cutoff: date | None = None,
                   trading_days: Sequence | None = None) -> dict[str, list[str]]:
    """窗口统计 → {hs300, csi500, csi1000} 三个名单（**纯函数**）。

    Args:
        stats: index=symbol，列至少含 `mcap_avg` / `amt_avg` / `n_days`。
        list_dates: {symbol: 首个交易日 'YYYY-MM-DD'}，用于"上市满 N 个交易日"。
        cutoff / trading_days: 给定则启用上市时长过滤（否则只用 n_days）。
    """
    need = {"mcap_avg", "amt_avg", "n_days"}
    if stats is None or not len(stats) or not need.issubset(stats.columns):
        return {HS300: [], CSI500: [], CSI1000: []}

    df = stats.copy()
    df = df[(df["n_days"] >= rules.min_days_in_window) & df["mcap_avg"].notna()
            & df["amt_avg"].notna()]
    if not len(df):
        return {HS300: [], CSI500: [], CSI1000: []}

    if rules.exclude_bj:
        df = df[[not str(s).startswith(_BJ_PREFIX) for s in df.index]]

    # 上市时长：要求该股的上市日 ≤ 截止日往前第 min_list_days 个交易日
    if list_dates and cutoff is not None and trading_days is not None:
        days = [_as_date(t) for t in trading_days]
        days = [d for d in days if d is not None]
        i = sum(1 for d in days if d <= cutoff) - 1      # cutoff 在交易日序列中的下标
        if i >= rules.min_list_days:
            threshold = days[i - rules.min_list_days]
            keep = [s for s in df.index
                    if _as_date(list_dates.get(s)) is not None
                    and _as_date(list_dates[s]) <= threshold]
            df = df.loc[keep]

    mcap, amt = df["mcap_avg"], df["amt_avg"]

    # ① 沪深300 近似（成交额剔后20% → 市值前 300），仅供 500/1000 剔除
    top300 = set(mcap.sort_values(ascending=False).index[: rules.exclude_top_mcap])
    hs300 = pick_members(mcap, amt, rules, exclude=set(), size=rules.hs300_size)

    # ② 中证500：剔除 (沪深300 ∪ 市值前300) → 成交额剔后20% → 市值前 500
    csi500 = pick_members(mcap, amt, rules, exclude=set(hs300) | top300,
                          size=rules.csi500_size)

    # ③ 中证1000：剔除 (中证800 ∪ 市值前300) → 成交额剔后20% → 市值前 1000
    csi1000 = pick_members(mcap, amt, rules, exclude=set(hs300) | set(csi500) | top300,
                           size=rules.csi1000_size)

    return {HS300: hs300, CSI500: csi500, CSI1000: csi1000}


def _as_date(v) -> date | None:
    """把 'YYYY-MM-DD' / datetime / date 统一成 date。"""
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


# ============================================================
# IO 层
# ============================================================
def window_stats(con: sqlite3.Connection, cutoff: str, win_start: str
                 ) -> pd.DataFrame:
    """单个截止日的窗口统计（用 SQL 聚合，**不建大面板**——内存恒定）。

    Returns: index=symbol，列 = mcap_avg / amt_avg / n_days
    """
    df = pd.read_sql_query(
        "SELECT symbol, AVG(float_mcap) AS mcap_avg, AVG(amount) AS amt_avg, "
        "       COUNT(*) AS n_days "
        "FROM full_daily "
        "WHERE date > ? AND date <= ? AND amount IS NOT NULL "
        "  AND float_mcap IS NOT NULL AND float_mcap > 0 "
        "GROUP BY symbol", con, params=(win_start, cutoff))
    return df.set_index("symbol") if len(df) else \
        pd.DataFrame(columns=["mcap_avg", "amt_avg", "n_days"])


def list_dates(con: sqlite3.Connection) -> dict[str, str]:
    """{symbol: 首个交易日}（上市日的代理）。"""
    return {r[0]: r[1] for r in
            con.execute("SELECT symbol, MIN(date) FROM full_daily GROUP BY symbol")}


def build_snapshots(con: sqlite3.Connection, start: str, end: str,
                    rules: PitRules | None = None,
                    progress: bool = True) -> dict[date, dict[str, list[str]]]:
    """在 [start, end] 内逐期重建宇宙。

    Returns: {生效日: {hs300: [...], csi500: [...], csi1000: [...]}}
    """
    rules = rules or PitRules()
    days = [r[0] for r in con.execute(
        "SELECT DISTINCT date FROM full_daily WHERE date <= ? ORDER BY date", (end,))]
    days = [d for d in days if d >= start]
    if not days:
        raise RuntimeError("库内无数据：请先跑 scripts/35_fetch_full_market.py")

    cutoffs = [c for c in semi_annual_cutoffs(range(
        int(start[:4]), int(end[:4]) + 1)) if start <= c.isoformat() <= end]
    eff = review_effective_dates(cutoffs, days)
    ldates = list_dates(con)

    out: dict[date, dict[str, list[str]]] = {}
    for c in cutoffs:
        eff_day = eff.get(c)
        if eff_day is None:
            continue
        cstr = c.isoformat()
        idx = days.index(next(d for d in days if d >= cstr))
        if idx < rules.window:                      # 回看窗口不足 → 跳过该期
            if progress:
                print(f"  [跳过] 截止 {c}：回看窗口不足（{idx} < {rules.window} 个交易日）")
            continue
        win_start = days[idx - rules.window]
        st = window_stats(con, cstr, win_start)
        snap = build_snapshot(st, rules, ldates, c, days)
        out[eff_day] = snap
        if progress:
            print(f"  截止 {c} → 生效 {eff_day}：HS300 {len(snap[HS300])} · "
                  f"CSI500 {len(snap[CSI500])} · CSI1000 {len(snap[CSI1000])}"
                  f"（窗口 {win_start}~{cstr}，候选 {len(st)}）")
    return out


def save_snapshots(con: sqlite3.Connection,
                   snaps: dict[date, dict[str, list[str]]]) -> int:
    """写入 `pit_members`（稀疏：一期一行一票）。返回写入行数。"""
    con.executescript("""
        CREATE TABLE IF NOT EXISTS pit_members (
            review_date TEXT NOT NULL, index_code TEXT NOT NULL, symbol TEXT NOT NULL,
            PRIMARY KEY (review_date, index_code, symbol));
    """)
    # 先清掉这些期次的旧行再写：否则改了规则/数据后重跑，**上一版多出来的成员会残留**
    # （INSERT OR REPLACE 只覆盖同主键的行，删不掉上一版有、这一版没有的票）。
    days = [str(d) for d in snaps]
    if days:
        con.executemany("DELETE FROM pit_members WHERE review_date = ?",
                        [(d,) for d in days])
    rows = [(str(d), code, s)
            for d, snap in snaps.items() for code, syms in snap.items() for s in syms]
    con.executemany("INSERT OR REPLACE INTO pit_members VALUES (?,?,?)", rows)
    con.commit()
    return len(rows)


def load_snapshots(con: sqlite3.Connection) -> dict[date, dict[str, list[str]]]:
    """从 `pit_members` 读回快照结构。"""
    out: dict[date, dict[str, list[str]]] = {}
    for rd, code, sym in con.execute(
            "SELECT review_date, index_code, symbol FROM pit_members ORDER BY review_date"):
        out.setdefault(_as_date(rd), {}).setdefault(code, []).append(sym)
    return out


def build_mask(con: sqlite3.Connection, index_codes: Iterable[str] = (CSI500, CSI1000),
               dates: Sequence | None = None, symbols: Iterable[str] | None = None
               ) -> pd.DataFrame:
    """date × symbol 的 bool 掩码：任一时点 t 取 `review_date <= t` 的**最新**快照。

    [!]️ 必须是"往前取最新快照"，**不能**先 ffill 整库再 shift（那会在调仓日当天偷看未来）。
    """
    codes = set(index_codes)
    snaps: dict[date, set[str]] = {}
    for rd, code, sym in con.execute("SELECT review_date, index_code, symbol FROM pit_members"):
        if code in codes:
            snaps.setdefault(_as_date(rd), set()).add(sym)
    if not snaps:
        return pd.DataFrame()

    rev = sorted(snaps)
    if dates is None:
        dates = [r[0] for r in con.execute("SELECT DISTINCT date FROM full_daily ORDER BY date")]
    dts = [d if isinstance(d, pd.Timestamp) else pd.Timestamp(_as_date(d)) for d in dates]

    # 每个日期 → 生效的最新一期（严格 <= t）
    assign: list[set[str]] = []
    j = -1
    for t in dts:
        while j + 1 < len(rev) and rev[j + 1] <= t.date():
            j += 1
        assign.append(snaps[rev[j]] if j >= 0 else set())

    cols = sorted(symbols) if symbols is not None else sorted({s for v in snaps.values() for s in v})
    idx = {s: i for i, s in enumerate(cols)}
    arr = np.zeros((len(dts), len(cols)), dtype=bool)
    for r, members in enumerate(assign):
        for s in members:
            c = idx.get(s)
            if c is not None:
                arr[r, c] = True
    return pd.DataFrame(arr, index=pd.DatetimeIndex(dts), columns=cols)


def load_panels(con: sqlite3.Connection, cols: Sequence[str], dates: Sequence,
                symbols: Sequence[str]) -> dict[str, pd.DataFrame]:
    """按需取面板（只取给定 symbol，float32）——避免 5000×1700 的大面板吃内存。"""
    if not len(symbols):
        return {}
    q = ",".join("?" * len(symbols))
    d0, d1 = str(dates[0])[:10], str(dates[-1])[:10]
    sel = ",".join(["date", "symbol"] + list(cols))
    df = pd.read_sql_query(
        f"SELECT {sel} FROM full_daily WHERE date >= ? AND date <= ? AND symbol IN ({q})",
        con, params=[d0, d1, *symbols])
    out: dict[str, pd.DataFrame] = {}
    for c in cols:
        p = df.pivot_table(index="date", columns="symbol", values=c).sort_index()
        p.index = pd.to_datetime(p.index)
        out[c] = p.astype("float32")
    return out


def load_raw_close(con: sqlite3.Connection, dates: Sequence, symbols: Sequence[str],
                   ) -> pd.DataFrame:
    """**不复权**收盘价面板 = `amount / volume`（成交额 / 成交量 = 当日元/股）。

    ## 为什么必须有这个函数：qfq 价不能用来算历史 PE

    `full_daily.close` 是 **qfq 前复权**，锚定在**今天**。也就是说历史价已经被
    「今天之后发生的所有送转与分红」折算过了，而财务数据里的 EPS 是 **as-reported**
    （报告期当时的股本口径，未经追溯重述）。两者相除，历史 PE 会被系统性低估：

        送转前真实 PE = 100/1.0 = 100
        qfq 价已被 10送10 折半 => qfq价/EPS = 50/1.0 = 50   （低估一半）

    实测（`scripts/39_validate_fundamentals.py` E 段，2020-04-30 截面）：
    两种口径的 PE **排序相关只有 0.938**；按「日后是否送转」分层，
    有送转的票低到 **0.862**、比值 p5 = 0.235（失真 4 倍）。
    PE 是线上选股 60% 的权重 => 回测用 qfq 口径会得到错误的选股结果。

    ⚠️ 对照：**线上选股不受影响**——`selector._fetch_pe` 取的是百度估值接口的
    PE(TTM) 当日值，qfq 在「今天」无折算差。本函数只服务历史回测。

    Returns: date × symbol 的 float64 面板（volume=0 的停牌行 -> NaN）。
    """
    if not len(symbols):
        return pd.DataFrame(index=pd.DatetimeIndex(dates), dtype="float64")
    q = ",".join("?" * len(symbols))
    ds = ",".join("?" * len(dates))
    df = pd.read_sql_query(
        f"SELECT date, symbol, amount, volume FROM full_daily "
        f"WHERE date IN ({ds}) AND symbol IN ({q})",
        con, params=[*[str(d)[:10] for d in dates], *symbols])
    df["raw"] = df["amount"] / df["volume"].replace(0, np.nan)
    p = df.pivot_table(index="date", columns="symbol", values="raw").sort_index()
    p.index = pd.to_datetime(p.index)
    return p.reindex(index=pd.DatetimeIndex(dates), columns=list(symbols)).astype("float64")
