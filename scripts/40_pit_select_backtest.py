"""第 40 步：**线上 `select_daily` 的忠实复刻**，在 PIT 无偏宇宙上回测。

## 为什么做这个

`docs/2026-09-11-pit-universe.md` 的结论是：此前所有回测的**绝对水平不可信**，因为
559 池是「用今天的成分名单回填历史」。而线上选股有 60% 权重压在 PE/ROE 上，其历史
时点值此前拿不到 ⇒ **线上策略从来没被严格回测过**。

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
| **`tech is None` → `total = fund_score`** | 同（变体 A） | ⚠️ 这是线上代码的**真实行为**，见下 |

## ⚠️ 复刻中发现的一处线上选股缺陷（变体 A vs B）

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
fund=83/tech=30 的票只有 61.8 分 ⇒ **缺数据的票被系统性优待，技术面（40% 权重）大半被架空**。

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


def score_stock(pe, roe):
    """与 selector.score_stock 同式。"""
    pe_score = max(0.0, min(50.0, 50.0 * (PE_BASE / max(pe, 1.0))))
    roe_score = max(0.0, min(50.0, 50.0 * (roe / ROE_BASE)))
    return round(pe_score + roe_score, 1)


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

    返回 (funded, n_liq)；funded = [(code, fund_score)] 已按 fund_score 降序。
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
        funded.append((c, score_stock(p, r)))
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
    """
    topn = topn or TOPN
    funded, n_liq = candidates_at(d, close, amount, pe, roe)
    if not funded:
        return [], {}

    if mode == "G":                     # 技术面单因子：对全部 80 只候选算技术面
        ts_map = tech_scores_at(d, tech, [c for c, _ in funded], weights)
        if not ts_map:
            return [], {}
        rows = sorted(ts_map.items(), key=lambda x: -x[1])
        return [c for c, _ in rows[:topn]], {"funded": len(funded),
                                             "tech": len(ts_map), "liq": n_liq}

    if mode == "F":                     # 基本面单因子
        return [c for c, _ in funded[:topn]], {"funded": len(funded),
                                               "tech": 0, "liq": n_liq}

    # A/B：技术面只对基本面 top TECH_TOPK 计算（与线上一致）
    ts_map = tech_scores_at(d, tech, [c for c, _ in funded[:TECH_TOPK]], weights)
    rows = []
    for c, fs in funded:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--topn", type=int, default=TOPN)
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
