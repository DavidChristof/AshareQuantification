"""第 40 步：**线上 `select_daily` 的忠实复刻**，在 PIT 无偏宇宙上回测。

## 为什么做这个

`docs/2026-09-11-pit-universe.md` 的结论是：此前所有回测的**绝对水平不可信**，因为
559 池是「用今天的成分名单回填历史」。而线上选股有 60% 权重压在 PE/ROE 上，其历史
时点值此前拿不到 => **线上策略从来没被严格回测过**。

前两步补齐了材料：PIT 无偏宇宙（`quant/data/universe_pit.py`）+ 时点 PE/ROE
（`quant/data/fundamentals.py`，已在 `docs/2026-09-12-fundamentals-pit.md` 校验）。
现在把 `selector.select_daily` 的**每一层逐条复刻**上来。

## 复刻对照表（逐条对应 selector.py）

| 线上 | 本脚本 | 说明 |
|---|---|---|
| `selection.universe=large`（现池40∪中小盘~560） | **PIT 成员**（中证500∪1000） | **唯一有意不同的一处**：换掉有偏的池子正是本脚本的目的 |
| 第1层 排除 ST（按 name 匹配） | **未实现** | 无历史 ST 名单，见「已知局限」 |
| 第2层 `liq >= 1e8` | `amount >= 1e8` | 线上取实时快照额、这里取当日全天额 |
| 流动性 top `BASIC_TOPK=80` | 同 | |
| `_fetch_pe` 百度 PE(TTM)，`<MIN_PE` 视为缺失 | `不复权价 / EPS_ttm`，`<1.0` 视为缺失 | 口径见 docs/2026-09-12（**不能用 qfq 价**） |
| `_fetch_roe` 最新一期累计 ROE | `latest_panel(roe_ytd)` | **不是 TTM** —— 线上取的就是最新一期报告值 |
| `score_stock` PE/ROE 各 50 分 | 同（同公式同常数） | |
| 基本面 top `TECH_TOPK=40` 算技术面 | 同 | |
| `_daily_tech` 取 120 自然日 qfq 日线 | 同（预计算面板） | 技术面**必须用 qfq**（比值口径，锚点无关） |
| `_score_tech` 4 因子归一 + 权重 | 同 | |
| `fetch_market_regime` 沪深300 日线 | 同（`MarketRegime(20,20)`） | 用 `close<=d` 切片，无未来函数 |
| `total = fund*0.6 + tech*0.4` | 同 | |
| **`tech is None` → `total = fund_score`** | 同（变体 A） | [!] 这是线上代码的**真实行为**，见下 |

## [!] 复刻中发现的一处线上选股缺陷（变体 A vs B）

```python
tech_score = _score_tech(ts, weights)
if tech_score is None:              # 无技术面则只靠基本面
    total = fund_score              # <-- 直接拿满权重的基本面分
else:
    total = fund_score * 0.6 + tech_score * 0.4
```

技术面只对**基本面 top 40** 计算，而排序是对**全部 ~80 只**做的。于是**排在 41~80 名的票
（没算技术面）拿到 `total = fund_score`（0~100）**，而 top40 里的票被封顶为
`0.6*fund + 0.4*tech <= 100`。举例：fund=83 的落选者得 83 分，而 top40 里一只
fund=83/tech=30 的票只有 61.8 分 => **缺数据的票被系统性优待，技术面（40% 权重）大半被架空**。

变体 A 如实复刻该行为，变体 B 把缺失技术面按 0 处理（`total = fund*0.6`）以量化其代价。

## 变体

    A 忠实复刻（含上面那处缺陷）
    B A + 修正「技术面缺失拿满分」-> total = fund*0.6
    C B + 关闭 regime 门控（恒用默认权重 mom25/trd15/vol30/rev30）
    D 基准：同宇宙同流动性门槛下**随机**取 12 只（种子固定），看选股是否真有 alpha
    E 基准：PIT 全宇宙等权（就是"不选股"）

用法：
    .venv/Scripts/python.exe scripts/40_pit_select_backtest.py
    .venv/Scripts/python.exe scripts/40_pit_select_backtest.py --start 2021-07-01 --topn 12
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from quant.config import load_config                                # noqa: E402
from quant.data.fundamentals import load_financials, ttm_panel      # noqa: E402
from quant.data.universe_pit import CSI1000, CSI500, build_mask    # noqa: E402
from quant.timing.regime import MarketRegime                        # noqa: E402

# ---- 与 selector.py 逐字一致的常数 ----
MIN_AMOUNT = 1e8
MIN_PE = 1.0
PE_BASE = 30.0
ROE_BASE = 15.0
BASIC_TOPK = 80
TECH_TOPK = 40
DEFAULT_TECH_WEIGHTS = {"mom": 25, "trd": 15, "vol": 30, "rev": 30}
TECH_REGIME_WEIGHTS = {
    "uptrend": {"mom": 45, "trd": 25, "vol": 15, "rev": 15},
    "range": {"mom": 20, "trd": 10, "vol": 35, "rev": 35},
    "downtrend": {"mom": 10, "trd": 10, "vol": 35, "rev": 45},
}

# ---- 回测参数（与 scripts/37 口径一致，便于跨结论对照）----
REBAL = 5
COST = 0.0016
TOPN = 12
RANDOM_SEED = 42
N_SEEDS = 20        # 随机基准取多少个种子平均（单种子噪声太大，曾把 t 值从 -0.5 抖到 -4）


def _pivot(df, col, fdtype="float32"):
    p = df.pivot_table(index="date", columns="symbol", values=col).sort_index()
    p.index = pd.to_datetime(p.index)
    return p.astype(fdtype)


def pick_tech_weights(regime):
    w = dict(TECH_REGIME_WEIGHTS.get(regime or "", DEFAULT_TECH_WEIGHTS))
    total = float(sum(w.values())) or 1.0
    return {k: round(v / total * 100, 1) for k, v in w.items()}


# ============================================================
# 数据
# ============================================================
def load_pit_panels(cfg, start):
    """PIT 宇宙面板：close(qfq, 技术面用) / amount(流动性) / raw(不复权价, PE 用)。"""
    con = sqlite3.connect(str(Path(cfg.resolve("data")) / "full_market.db"))
    dates = pd.DatetimeIndex([r[0] for r in con.execute(
        "SELECT DISTINCT date FROM full_daily WHERE date >= ? ORDER BY date", (start,))])
    mask = build_mask(con, (CSI500, CSI1000), dates)
    syms = list(mask.columns)
    ph = ",".join("?" * len(syms))
    df = pd.read_sql_query(
        f"SELECT symbol,date,close,amount,volume FROM full_daily "
        f"WHERE date >= ? AND symbol IN ({ph})", con, params=[start, *syms])
    con.close()
    close = _pivot(df, "close")
    amount = _pivot(df, "amount")
    volume = _pivot(df, "volume")
    raw = amount / volume.replace(0, np.nan)
    m = mask.reindex(close.index).fillna(False).astype(bool)
    m = m.reindex(columns=close.columns, fill_value=False)
    for p in (close, amount, raw):
        p[~m] = np.nan
    return close, amount, raw, m


def load_hs300(cfg):
    """沪深300 日线（regime 用）。落库缓存，避免每次重跑都打网络。"""
    db = Path(cfg.resolve("data")) / "index_daily.db"
    if db.exists():
        con = sqlite3.connect(db)
        try:
            df = pd.read_sql_query("SELECT date,close FROM hs300_daily ORDER BY date", con)
            con.close()
            if len(df) > 100:
                s = pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
                return s
        except Exception:  # noqa: BLE001
            con.close()
    from quant.realtime.indices import fetch_index_daily       # noqa: PLC0415
    df = fetch_index_daily("sh000300")
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    con = sqlite3.connect(db)
    df.to_sql("hs300_daily", con, if_exists="replace", index=False)
    con.close()
    return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))


def tech_factor_panels(close):
    """4 个技术面因子面板（qexec 与 selector._daily_tech 同式）。"""
    ret1 = close.pct_change(fill_method=None)
    mom = close.pct_change(20, fill_method=None)
    ma20 = close.rolling(20).mean()
    trd = close / ma20 - 1
    vol = ret1.rolling(20).std()
    rev = close.shift(60) / close - 1
    # 线上：len<21 时 mom20=0.0、ma20 无效时 trend=0.0、len<61 时 rev60=0.0
    return mom.fillna(0.0), trd.fillna(0.0), vol, rev.fillna(0.0)


def score_parts(pe, roe):
    """基本面两半的计算值，分开返回 —— 归因要知道 PE / ROE 各自贡献了什么。

    线上 `selector.score_stock` 把它们熔成一个数就再也拆不开，所以这里单独算一份。
    公式与线上逐字一致。
    """
    pe_score = max(0.0, min(50.0, 50.0 * (PE_BASE / max(pe, 1.0))))
    roe_score = max(0.0, min(50.0, 50.0 * (roe / ROE_BASE)))
    return pe_score, roe_score


def score_stock(pe, roe):
    """与 selector.score_stock 同式。"""
    p, r = score_parts(pe, roe)
    return round(p + r, 1)


def score_tech(mom, trd, vol, rev, w):
    """与 selector._score_tech 同式。"""
    m = max(0.0, min(1.0, (mom + 0.05) / 0.25))
    t = max(0.0, min(1.0, (trd + 0.05) / 0.10))
    v = max(0.0, min(1.0, 1.0 - vol / 0.03))
    r = max(0.0, min(1.0, (rev + 0.20) / 0.30))
    return round(m * w["mom"] + t * w["trd"] + v * w["vol"] + r * w["rev"], 1)


# ============================================================
# 单日选股（复刻 select_daily）
# ============================================================
def candidates_at(d, close, amount, pe, roe):
    """复刻第 1~2 层 + 基本面：流动性 >=1e8 -> top80 -> PE/ROE 打分。

    返回 (funded, n_liq)；funded = [(code, fund_score, pe_score, roe_score)] 已按 fund_score 降序。

    [!] 为什么多返回 pe_score/roe_score：归因要回答「PE 和 ROE 谁在赚」，
    而线上把两者熔成一个 fund_score，拆不开。多返回两个数不改变任何排序行为。
    """
    cl, amt = close.loc[d], amount.loc[d]
    liq = amt[amt >= MIN_AMOUNT].dropna()
    if len(liq) == 0:
        return [], 0
    liq_rank = list(liq.nlargest(BASIC_TOPK).index)

    # PE/ROE 缺失或 PE<MIN_PE -> 整只剔除（线上 fetch_fundamental 返回 None）
    ped, roed = pe.loc[d], roe.loc[d]
    funded = []
    for c in liq_rank:
        p, r = ped.get(c, np.nan), roed.get(c, np.nan)
        if not np.isfinite(p) or p < MIN_PE or not np.isfinite(r):
            continue
        if not np.isfinite(cl.get(c, np.nan)):
            continue
        pes, roes = score_parts(p, r)
        funded.append((c, round(pes + roes, 1), pes, roes))
    funded.sort(key=lambda x: -x[1])
    return funded, len(liq)


def tech_scores_at(d, tech, codes, weights):
    """给定代码，算技术面分；数据不足（vol20 为 NaN）跳过。"""
    mom, trd, vol, rev = tech
    out = {}
    for c in codes:
        v = vol.at[d, c]
        if not np.isfinite(v):
            continue
        out[c] = score_tech(mom.at[d, c], trd.at[d, c], float(v), rev.at[d, c], weights)
    return out


def select_at(d, close, amount, pe, roe, tech, weights, mode="A", topn=None):
    """按 `mode` 产出该日 top-N。

    mode: A 忠实复刻 / B 修正 tech 缺失 / F 只用基本面 / G 只用技术面
          P 只用 PE 分 / R 只用 ROE 分（归因用，仿 G 走全 80 只候选）
    """
    topn = topn or TOPN
    funded, n_liq = candidates_at(d, close, amount, pe, roe)
    if not funded:
        return [], {}

    if mode == "G":                     # 技术面单因子：对全部 80 只候选算技术面
        ts_map = tech_scores_at(d, tech, [r[0] for r in funded], weights)
        if not ts_map:
            return [], {}
        rows = sorted(ts_map.items(), key=lambda x: -x[1])
        return [c for c, _ in rows[:topn]], {"funded": len(funded),
                                             "tech": len(ts_map), "liq": n_liq}

    if mode == "F":                     # 基本面单因子
        return [r[0] for r in funded[:topn]], {"funded": len(funded),
                                               "tech": 0, "liq": n_liq}

    if mode in ("P", "R"):              # 基本面**半**因子：只按 PE 分 / ROE 分排序
        col = 2 if mode == "P" else 3
        rows = sorted(funded, key=lambda x: -x[col])
        return [r[0] for r in rows[:topn]], {"funded": len(funded),
                                             "tech": 0, "liq": n_liq}

    # A/B：技术面只对基本面 top TECH_TOPK 计算（与线上一致）
    ts_map = tech_scores_at(d, tech, [r[0] for r in funded[:TECH_TOPK]], weights)
    rows = []
    for c, fs, _pes, _roes in funded:
        ts = ts_map.get(c)
        if ts is None:
            total = fs * 0.6 if mode == "B" else fs
        else:
            total = fs * 0.6 + ts * 0.4
        rows.append((c, total))
    rows.sort(key=lambda x: -x[1])
    diag = {"funded": len(funded), "tech": len(ts_map), "liq": n_liq,
            "picked_from_top40": sum(1 for c, _ in rows[:topn] if c in ts_map)}
    return [c for c, _ in rows[:topn]], diag


# ============================================================
# 回测
# ============================================================
# 单因子消融（--attrib 用）：只保留一个技术面子因子，权重 100、其余 0。
# `select_at` 的 G 分支本来就吃任意权重 —— 所以这四个臂**不需要新的打分代码**。
# 方向沿用 score_tech 的方向：mom/trd/rev 越大越好，vol 越小越好（score_tech 里是 1 - vol/0.03）。
TECH_ONLY_WEIGHTS = {
    "M_mom": {"mom": 100, "trd": 0, "vol": 0, "rev": 0},
    "M_trd": {"mom": 0, "trd": 100, "vol": 0, "rev": 0},
    "M_vol": {"mom": 0, "trd": 0, "vol": 100, "rev": 0},
    "M_rev": {"mom": 0, "trd": 0, "vol": 0, "rev": 100},
}
# 基本面**半**因子：只按 PE 分 / 只按 ROE 分排序
FUND_HALF_MODE = {"P_pe": "P", "R_roe": "R"}


def run(variant, close, amount, pe, roe, tech, regime_by_date, topn, dates):
    sel_dates = list(dates[::REBAL])
    nav, curve = 1.0, []
    diags, alphas = [], []
    for d in sel_dates[:-1]:
        i = dates.get_loc(d)
        if i + REBAL >= len(dates):
            break
        d2 = dates[i + REBAL]

        if variant in ("D", "H"):                           # 随机基准（多种子取平均）
            # D 从「通过 1 亿流动性门槛」的集合抽；H 从全宇宙抽（隔离门槛本身的影响）
            uni_r = _uni_ret(close, d, d2)
            cand = amount.loc[d].dropna()
            if variant == "D":
                cand = cand[cand >= MIN_AMOUNT]
            ok = cand.index.intersection(close.loc[d].dropna().index
                                         ).intersection(close.loc[d2].dropna().index)
            if len(ok) < topn:
                continue
            arr = np.array(list(ok))
            rs = []
            for k in range(N_SEEDS):
                rng = np.random.default_rng(RANDOM_SEED + i * 1000 + k)
                sub = rng.choice(arr, size=topn, replace=False)
                seg = (close.loc[d2, sub] / close.loc[d, sub] - 1).dropna()
                if len(seg):
                    rs.append(float(seg.mean()))
            if not rs:
                continue
            r = float(np.mean(rs)) - COST
            nav *= (1 + r)
            curve.append((d2, nav))
            if not np.isnan(uni_r):
                alphas.append((d2, r - uni_r))
            continue
        if variant == "E":                                  # 全宇宙等权
            picks = list(close.loc[d].dropna().index)
        elif variant in TECH_ONLY_WEIGHTS:                   # 单技术因子（G 分支 + 单位权重）
            picks, diag = select_at(d, close, amount, pe, roe, tech,
                                    TECH_ONLY_WEIGHTS[variant], "G", topn)
            if diag:
                diags.append({**diag, "date": d})
        elif variant in FUND_HALF_MODE:                      # 只按 PE 分 / 只按 ROE 分
            picks, diag = select_at(d, close, amount, pe, roe, tech,
                                    DEFAULT_TECH_WEIGHTS, FUND_HALF_MODE[variant], topn)
            if diag:
                diags.append({**diag, "date": d})
        else:
            mode = variant if variant in ("A", "B", "F", "G") else "B"
            w = (DEFAULT_TECH_WEIGHTS if variant == "C"
                 else regime_by_date.get(d, DEFAULT_TECH_WEIGHTS))
            picks, diag = select_at(d, close, amount, pe, roe, tech, w, mode, topn)
            if diag:
                diags.append({**diag, "date": d})
        if len(picks) < max(1, topn // 2):
            continue

        p0 = close.loc[d, picks]
        p1 = close.loc[d2, picks]
        seg = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(seg):
            continue
        r = float(seg.mean()) - (COST if variant != "E" else 0.0)
        nav *= (1 + r)
        curve.append((d2, nav))
        uni_r = _uni_ret(close, d, d2)
        if not np.isnan(uni_r):
            alphas.append((d2, r - uni_r))
    return curve, diags, alphas


def _uni_ret(close, d, d2):
    """PIT 全宇宙等权收益（同期基准）。"""
    uni = (close.loc[d2] / close.loc[d] - 1).replace([np.inf, -np.inf], np.nan).dropna()
    return float(uni.mean()) if len(uni) else np.nan


def metrics(curve):
    if not curve:
        return {}
    ser = pd.Series(dict(curve))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    tot = ser.iloc[-1] - 1
    per = ser.pct_change().dropna()
    return {"total": round(tot * 100, 1),
            "annual": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else 0.0,
            "maxdd": round(float((ser / ser.cummax() - 1).min()) * 100, 1),
            "sharpe": round(float(per.mean() / per.std() * np.sqrt(252 / REBAL)), 2)
            if per.std() else 0.0,
            "_ser": ser}


def yearly(ser):
    return {str(y): round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
            for y, g in ser.groupby(ser.index.year) if len(g) > 1}


# ============================================================
# 因子归因（--attrib）
# ============================================================
# 六个子因子，方向统一成「越大越好」——这样价差的正负可直接读成「越大越赚」。
# vol 取负：score_tech 里是 `1 - vol/0.03`，低波得高分，所以「低波」= 好。
FACTOR_LABELS = {
    "pe_score": "PE 分(低PE=高分)",
    "roe_score": "ROE 分(高ROE=高分)",
    "mom": "动量 mom20",
    "trd": "趋势 close/MA20-1",
    "vol": "低波 -vol20",
    "rev": "反转 rev60",
}


def _factor_panels(pe, roe, tech):
    """6 个子因子面板，方向统一为「越大越好」。"""
    mom, trd, vol, rev = tech
    pe_s = (50.0 * (PE_BASE / pe.clip(lower=1.0))).clip(0.0, 50.0)
    roe_s = (50.0 * (roe / ROE_BASE)).clip(0.0, 50.0)
    return {"pe_score": pe_s, "roe_score": roe_s,
            "mom": mom, "trd": trd, "vol": -vol, "rev": rev}


def _tstat(vals, since=None, pct=True):
    """逐期值 -> {n, mean_pct, t}；since='2023-01-01' 时只取该日之后（子区间稳健性）。

    pct=True 把均值乘 100（收益类）；**IC 必须传 pct=False** ——
    Spearman IC 的取值在 [-1, 1]，乘 100 会打出「IC=7.15」这种不可能的数（踩过一次）。
    t 值与是否乘 100 无关（分子分母同比例）。
    """
    a = np.asarray([v for d, v in vals if since is None or str(d)[:10] >= since], dtype=float)
    a = a[np.isfinite(a)]
    if len(a) < 3:
        return {"n": int(len(a)), "mean_pct": None, "t": None}
    se = float(a.std(ddof=1)) / np.sqrt(len(a))
    scale = 100.0 if pct else 1.0
    return {"n": int(len(a)), "mean_pct": round(float(a.mean()) * scale, 4),
            "t": round(float(a.mean()) / se, 2) if se else None}


def factor_quantile_table(close, pe, roe, tech, dates):
    """第 1 层：因子分位归因 —— 不经组合构建、不经费用、不被那个缺陷污染。

    每个调仓期，对每个子因子：
      价差 = 当日截面 top1/3 减 bottom1/3 的前向 5 日收益
      超额 = 当日截面 top1/3 减 全宇宙均值（「超配头部」实际能拿到多少）
      IC   = 该因子与前瞻收益的截面 Spearman
    再对逐期值做单样本 t（与 scripts/45 同一手法）。
    """
    from scipy.stats import spearmanr
    panels = _factor_panels(pe, roe, tech)
    spread = {k: [] for k in FACTOR_LABELS}
    excess = {k: [] for k in FACTOR_LABELS}
    ic = {k: [] for k in FACTOR_LABELS}
    sel = list(dates[::REBAL])
    for d in sel[:-1]:
        i = dates.get_loc(d)
        if i + REBAL >= len(dates):
            break
        d2 = dates[i + REBAL]
        fwd = (close.loc[d2] / close.loc[d] - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(fwd) < 30:
            continue
        uni = float(fwd.mean())
        for k, p in panels.items():
            if d not in p.index:
                continue
            f = p.loc[d].dropna()
            b = f.index.intersection(fwd.index)
            if len(b) < 30:
                continue
            ff = f[b].to_numpy(dtype=float)
            rr = fwd[b].to_numpy(dtype=float)
            ok = np.isfinite(ff) & np.isfinite(rr)
            if ok.sum() < 30:
                continue
            ff, rr = ff[ok], rr[ok]
            kk = max(1, len(ff) // 3)
            o = np.argsort(ff)
            top, bot = float(rr[o[-kk:]].mean()), float(rr[o[:kk]].mean())
            spread[k].append((d2, top - bot))
            excess[k].append((d2, top - uni))
            rho, _ = spearmanr(ff, rr)
            if np.isfinite(rho):
                ic[k].append((d2, float(rho)))
    return spread, excess, ic


def run_attribution(close, amount, pe, roe, tech, dates, regime_by_date, topn):
    """跑两层归因，写 results/factor_attribution.json（**不碰** pit_select_backtest.json）。"""
    from scipy.stats import norm

    out = {}
    spread, excess, ic = factor_quantile_table(close, pe, roe, tech, dates)

    # ---- 第 1 层表 ----
    n_tests = len(FACTOR_LABELS) * 3          # 6 因子 x (价差/超额/IC)
    thr = float(norm.ppf(1 - 0.025 / n_tests))
    print()
    print("=" * 96)
    print("第 1 层：因子分位归因（PIT 宇宙，逐调仓期；top1/3 - bottom1/3，或 top1/3 - 全宇宙）")
    print(f"  多重比较：{n_tests} 个检验 -> Bonferroni 阈值 |t| > {thr:.2f} 才算显著")
    print("=" * 96)
    print(f"{'因子':<20}{'价差%/期':>10}{'t':>7}   {'超额%/期':>10}{'t':>7}   "
          f"{'IC':>8}{'t':>7}   {'2023+价差t':>11}")
    print("-" * 96)
    out["quantile"] = {}
    for k, lab in FACTOR_LABELS.items():
        sp, ex, icv = _tstat(spread[k]), _tstat(excess[k]), _tstat(ic[k], pct=False)
        sp23 = _tstat(spread[k], since="2023-01-01")
        out["quantile"][k] = {"spread": sp, "excess": ex, "ic": icv, "spread_2023": sp23,
                              "n_periods": sp["n"]}
        mark = lambda t: ("*" if t is not None and abs(t) > thr else " ")  # noqa: E731
        print(f"{lab:<20}{(sp['mean_pct'] or 0):>10.3f}{sp['t'] or 0:>6.2f}{mark(sp['t'])}  "
              f"{(ex['mean_pct'] or 0):>10.3f}{ex['t'] or 0:>6.2f}{mark(ex['t'])}  "
              f"{(icv['mean_pct'] or 0):>8.4f}{icv['t'] or 0:>6.2f}{mark(icv['t'])}  "
              f"{(sp23['t'] or 0):>10.2f}")

    # ---- 第 2 层：单因子消融回测（对照 = D，同流动性门槛的随机 12 只）----
    print()
    print("=" * 96)
    print("第 2 层：单因子消融回测（能不能活着走过 top12 组合构建 + 逐期成本）")
    print("  对照 D = 同流动性门槛随机 12 只 —— 本框架里唯一有合法零假设的基准")
    print("=" * 96)
    arms = [("A", "A 真分数（含缺陷）"), ("B", "B 真分数（修缺陷）"),
            ("M_mom", "只 mom"), ("M_trd", "只 trd"), ("M_vol", "只低波"), ("M_rev", "只反转"),
            ("F", "只基本面(PE+ROE)"), ("P_pe", "只 PE 分"), ("R_roe", "只 ROE 分"),
            ("G", "只技术面(四合一)"), ("D", "D 随机 12(零假设)")]
    alpha_series = {}
    out["arms"] = {}
    print(f"{'臂':<24}{'总收益':>9}{'年化':>8}{'最大回撤':>10}{'Sharpe':>8}{'逐期alpha':>11}{'t':>7}")
    print("-" * 96)
    for v, lab in arms:
        curve, _diags, alphas = run(v, close, amount, pe, roe, tech,
                                    regime_by_date, topn, dates)
        m = metrics(curve)
        if not m:
            print(f"{lab:<24}{'无结果':>9}")
            continue
        ser = m.pop("_ser")
        alpha_series[v] = pd.Series(dict(alphas)).sort_index()
        a = np.asarray([x for _, x in alphas], dtype=float)
        a_mean = float(a.mean()) * 100 if len(a) else 0.0
        a_t = (float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))
               if len(a) > 2 and a.std() else 0.0)
        out["arms"][v] = {**m, "label": lab, "alpha_mean_pct": round(a_mean, 3),
                          "alpha_t": round(a_t, 2), "n_periods": len(a),
                          "yearly": yearly(ser)}
        print(f"{lab:<24}{m['total']:>9.1f}{m['annual']:>8.2f}{m['maxdd']:>10.1f}"
              f"{m['sharpe']:>8.2f}{a_mean:>11.3f}{a_t:>7.2f}")

    # ---- 各臂 vs D 的配对检验 ----
    print()
    print("各臂 减 D（配对，同一天、只差一个打分规则）:")
    print(f"{'对比':<30}{'alpha差/期':>12}{'t':>8}")
    print("-" * 60)
    out["pair_vs_D"] = {}
    dser = alpha_series.get("D")
    for v, lab in arms:
        if v == "D" or v not in alpha_series or dser is None:
            continue
        s = pd.concat([alpha_series[v].rename("x"), dser.rename("y")],
                      axis=1, sort=False).dropna()
        if len(s) < 10:
            continue
        dd = s["x"] - s["y"]
        t = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std() else 0.0
        out["pair_vs_D"][v] = {"diff_pct": round(float(dd.mean()) * 100, 3),
                               "t": round(t, 2), "n": int(len(dd))}
        print(f"{lab + ' 减 D':<30}{float(dd.mean()) * 100:>12.3f}{t:>8.2f}")

    print()
    print("[!] A 与 B 要一起看：A 保留「排名 41~80 的票 total=fund_score 未减半」这个缺陷，")
    print("    它会**机械放大基本面分量的实际权重**（超出名义 0.6）=> 任何「基本面 vs 技术面」")
    print("    的比较在 A 口径下都偏向基本面。只看 A 会被它骗。")

    Path("results").mkdir(exist_ok=True)
    p = Path("results/factor_attribution.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n[out] {p}（既有 results/pit_select_backtest.json 未被触碰）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--topn", type=int, default=TOPN)
    ap.add_argument("--attrib", action="store_true",
                    help="改跑**因子归因**（第1层分位 + 第2层单因子消融），"
                         "写 results/factor_attribution.json，不跑 A-H、不碰既有 JSON")
    args = ap.parse_args()

    cfg = load_config()
    print(f"[in] 载入 PIT 面板 (start={args.start}) ...")
    close, amount, raw, mask = load_pit_panels(cfg, args.start)
    dates = close.index
    print(f"[in] PIT 面板: {close.shape[0]} 交易日 x {close.shape[1]} 只")

    conf = sqlite3.connect(str(Path(cfg.resolve("data")) / "fundamentals.db"))
    fin = load_financials(conf)
    conf.close()
    syms = list(close.columns)
    eps_ttm = ttm_panel(fin, "eps_q", dates, syms, window=4)
    roe = ttm_panel(fin, "roe_ytd", dates, syms, window=1)
    pe = (raw[syms] / eps_ttm.reindex(columns=syms).replace(0, np.nan))
    pe = pe.where(pe > 0)
    m = mask.reindex(index=pe.index, columns=pe.columns, fill_value=False).astype(bool)
    for p in (pe, roe):                       # 只在 PIT 成员内统计/使用
        p[~m] = np.nan
    n_cell = int(m.values.sum())
    print(f"[in] 时点面板: PE 有值率 {pe.notna().sum().sum() / max(n_cell, 1):.1%}"
          f" / ROE 有值率 {roe.notna().sum().sum() / max(n_cell, 1):.1%}")

    tech = tech_factor_panels(close)
    idx = load_hs300(cfg)
    mr = MarketRegime()
    regime_by_date = {}
    for d in dates:
        sub = idx[idx.index <= d]
        if len(sub) < 30:
            continue
        regime_by_date[d] = pick_tech_weights(mr.detect(sub)["regime"])
    n_up = sum(1 for w in regime_by_date.values() if w["mom"] > 30)
    print(f"[in] regime 判定: {len(regime_by_date)} 天（其中 {n_up} 天为上涨权重）")

    if args.attrib:                      # 归因模式：跑完就返回，不碰 A-H 与既有 JSON
        run_attribution(close, amount, pe, roe, tech, dates, regime_by_date, args.topn)
        return

    names = {"A": "A 忠实复刻（含 tech 缺失拿满分）",
             "B": "B 修正 tech 缺失 -> fund*0.6",
             "C": "C = B + 关闭 regime 门控",
             "D": "D 随机 12 只（同流动性门槛）",
             "E": "E PIT 全宇宙等权（不选股）",
             "F": "F 只用基本面（无技术面）",
             "G": "G 只用技术面（无基本面）",
             "H": "H 随机 12 只（全宇宙，不过流动性门槛）"}
    out = {}
    print()
    print(f"{'变体':<34}{'总收益':>9}{'年化':>8}{'最大回撤':>10}{'Sharpe':>8}"
          f"{'逐期alpha':>11}{'t值':>7}")
    print("-" * 90)
    alpha_series = {}
    for v in ("A", "B", "C", "D", "E", "F", "G", "H"):
        curve, diags, alphas = run(v, close, amount, pe, roe, tech,
                                   regime_by_date, args.topn, dates)
        m = metrics(curve)
        if not m:
            print(f"{names[v]:<34}{'无结果':>9}")
            continue
        ser = m.pop("_ser")
        alpha_series[v] = pd.Series(dict(alphas)).sort_index()
        a = np.array([x for _, x in alphas])
        a_mean = float(a.mean()) * 100 if len(a) else 0.0
        a_t = float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a)))) if len(a) > 2 and a.std() else 0.0
        out[v] = {**m, "alpha_mean_pct": round(a_mean, 3), "alpha_t": round(a_t, 2),
                  "n_periods": len(a), "yearly": yearly(ser),
                  "curve": {str(k)[:10]: round(x, 4) for k, x in ser.items()}}
        if diags:
            df = pd.DataFrame(diags)
            out[v]["diag_avg_funded"] = round(float(df["funded"].mean()), 1)
            out[v]["diag_avg_tech"] = round(float(df["tech"].mean()), 1)
            if "picked_from_top40" in df:
                out[v]["diag_avg_picked_from_top40"] = round(
                    float(df["picked_from_top40"].mean()), 2)
        print(f"{names[v]:<34}{m['total']:>9.1f}{m['annual']:>8.2f}"
              f"{m['maxdd']:>10.1f}{m['sharpe']:>8.2f}{a_mean:>11.3f}{a_t:>7.2f}")

    # ---- 配对比较：把「选股本身」的贡献从「流动性门槛+费用」里剥出来 ----
    print()
    print("配对比较（同期的 alpha 之差，逐期配对，t 值用配对标准误）:")
    print(f"{'对比':<30}{'alpha差/期':>12}{'t值':>8}{'解读':>34}")
    print("-" * 84)
    pairs = [("A", "D", "A 选股 减 同门槛随机 = 打分自身的贡献"),
             ("A", "E", "A 减 宇宙等权 = 门槛+费用+打分的总代价"),
             ("D", "E", "随机(过门槛) 减 宇宙 = 门槛+费用的代价"),
             ("H", "E", "随机(全宇宙) 减 宇宙 = 纯费用"),
             ("D", "H", "两者之差 = 流动性门槛本身的代价"),
             ("F", "G", "基本面 减 技术面 = 哪一边更拖累"),
             ("A", "F", "加技术面 相对 只用基本面"),
             ("A", "C", "开/关 regime 门控")]
    for x, y, note in pairs:
        if x not in alpha_series or y not in alpha_series:
            continue
        s = pd.concat([alpha_series[x].rename("x"), alpha_series[y].rename("y")],
                      axis=1).dropna()
        if len(s) < 10:
            continue
        d_ = s["x"] - s["y"]
        t = float(d_.mean() / (d_.std(ddof=1) / np.sqrt(len(d_)))) if d_.std() else 0.0
        out[f"pair_{x}_{y}"] = {"diff_pct": round(float(d_.mean()) * 100, 3),
                                "t": round(t, 2), "n": len(d_)}
        print(f"{x + ' 减 ' + y:<30}{float(d_.mean()) * 100:>12.3f}{t:>8.2f}{note:>34}")

    print()
    print("（逐期 alpha = 该期组合收益 减 同期 PIT 全宇宙等权收益，含费；正=选股有信息）")
    for v in ("A", "F", "G", "D"):
        if v in out:
            print(f"  {v} 逐年收益（%）: {out[v]['yearly']}")
    if "A" in out:
        print()
        print(f"A 诊断: 平均基本面票数 {out['A'].get('diag_avg_funded')} / "
              f"平均算出技术面 {out['A'].get('diag_avg_tech')} / "
              f"平均入选票中来自 top40 的只数 {out['A'].get('diag_avg_picked_from_top40')}"
              f"（共 {args.topn} 只）")

    Path("results").mkdir(exist_ok=True)
    with open("results/pit_select_backtest.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[out] results/pit_select_backtest.json")


if __name__ == "__main__":
    main()
