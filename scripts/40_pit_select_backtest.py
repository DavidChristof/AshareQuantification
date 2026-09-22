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

# 一次**完整往返**的费用（占成交额），按纸面/手动账户的费用口径（config: commission 0.0003 /
# stamp_tax 0.0005 / slippage 0.0002，均无最低佣金）：
#     买入  佣金 0.03% + 滑点 0.02%                 = 0.05%
#     卖出  佣金 0.03% + 印花税 0.05% + 滑点 0.02%   = 0.10%
#     合计                                           = 0.15%
# [!] 由此可知 `COST = 0.16%` **等价于「每期 100% 全额换手」的一次往返**。
# 而实测实际换手只有约 61%（每期换掉 7.4/12 只）=> 旧的 COST 口径**高估**了约 1.7 倍。
# 「按换手计费」模式用 ROUND_TRIP x 实际换手，才是自洽的。
ROUND_TRIP = 0.0015

# 实盘 3000 元账户口径（5 元最低佣金主导）：往返约 = 10/名义额 + 0.112%。仅作参照，未用于回测。
REAL_MIN_COMMISSION = 5.0


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


def score_tech(mom, trd, vol, rev, w, mode="clamp", flip=()):
    """与 selector._score_tech 同式。

    mode="clamp"（默认 = 线上原式）：把每个因子压到 [0,1] 后再加权，末尾 round 到 0.1。
    mode="raw"（**仅供归因**，不改线上）：不压、不 round —— 保留因子的连续排序信息。
    flip：要**反向**的因子名集合（如 {"mom","trd"}）。翻转用 `1 - x` 而不是负权重 ——
          这样分数仍在同一尺度上，只有方向变，不改各分量的相对幅度。

    [!] 为什么要 mode 这个开关：clamp 会让大量候选并列在 1.0，而 `sorted` 是稳定的 =>
    并列组内的顺序由插入顺序（= fund_score 降序）决定，**因子不参与排序**。
    实测 trd/mom 有 ~84%/81% 的候选并列在满分，其 top-12 **100% 取自并列组**。
    """
    m = (mom + 0.05) / 0.25
    t = (trd + 0.05) / 0.10
    v = 1.0 - vol / 0.03
    r = (rev + 0.20) / 0.30
    m = 1.0 - m if "mom" in flip else m
    t = 1.0 - t if "trd" in flip else t
    v = 1.0 - v if "vol" in flip else v
    r = 1.0 - r if "rev" in flip else r
    if mode == "clamp":
        m = max(0.0, min(1.0, m))
        t = max(0.0, min(1.0, t))
        v = max(0.0, min(1.0, v))
        r = max(0.0, min(1.0, r))
    val = m * w["mom"] + t * w["trd"] + v * w["vol"] + r * w["rev"]
    return round(val, 1) if mode == "clamp" else val


def _norm_panels(tech, how):
    """四个原始技术面板 -> 归一化到可比尺度的面板。

    how: clamp（线上原式，会并列）/ raw（不裁，保序，但尾部可能被离群值主导）
         / rank（**截面百分位**：严格保序、有界、且不会有并列）
    """
    mom, trd, vol, rev = tech
    m, t = (mom + 0.05) / 0.25, (trd + 0.05) / 0.10
    v, r = 1.0 - vol / 0.03, (rev + 0.20) / 0.30
    if how == "clamp":
        return tuple(x.clip(0.0, 1.0) for x in (m, t, v, r))
    if how == "raw":
        return (m, t, v, r)
    if how == "rank":
        return tuple(x.rank(axis=1, pct=True) for x in (m, t, v, r))
    raise ValueError(f"未知的归一化方式: {how}")


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


def tech_scores_at(d, tech, codes, weights, mode="clamp", flip=()):
    """给定代码，算技术面分；数据不足（vol20 为 NaN）跳过。mode/flip 见 score_tech。"""
    mom, trd, vol, rev = tech
    out = {}
    for c in codes:
        v = vol.at[d, c]
        if not np.isfinite(v):
            continue
        out[c] = score_tech(mom.at[d, c], trd.at[d, c], float(v), rev.at[d, c],
                            weights, mode, flip)
    return out


def select_at(d, close, amount, pe, roe, tech, weights, mode="A", topn=None,
              tech_mode="clamp", tech_flip=(), keep=None, buffer=0):
    """按 `mode` 产出该日 top-N。

    mode: A 忠实复刻 / B 修正 tech 缺失 / F 只用基本面 / G 只用技术面
          P 只用 PE 分 / R 只用 ROE 分（归因用，仿 G 走全 80 只候选）
    tech_mode: clamp（线上原式）/ raw（不 clamp，保序）—— 见 score_tech。
    tech_flip: 要反向的因子集合 —— 见 score_tech。
    keep/buffer: **持有缓冲区**（降换手用）。`keep` = 上一期持有的票；
        掉出前 `topn + buffer` 名的才卖，还在缓冲区内的继续持有（保持分数序），
        空缺用缓冲区里按分最高的新票补足。buffer=0 时行为与原先完全一致。
        线上手动/实盘路径**没有**这个机制（只在「今天 top-12 里还有它」时才留）。
    """
    topn = topn or TOPN
    funded, n_liq = candidates_at(d, close, amount, pe, roe)
    if not funded:
        return [], {}

    if mode == "G":                     # 技术面单因子：对全部 80 只候选算技术面
        ts_map = tech_scores_at(d, tech, [r[0] for r in funded], weights,
                                tech_mode, tech_flip)
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
    ts_map = tech_scores_at(d, tech, [r[0] for r in funded[:TECH_TOPK]], weights,
                            tech_mode, tech_flip)
    rows = []
    for c, fs, _pes, _roes in funded:
        ts = ts_map.get(c)
        if ts is None:
            total = fs * 0.6 if mode == "B" else fs
        else:
            total = fs * 0.6 + ts * 0.4
        rows.append((c, total))
    rows.sort(key=lambda x: -x[1])
    ranked = [c for c, _ in rows]
    n_keep = 0
    if keep and buffer:
        band = ranked[:topn + buffer]                  # 缓冲区：前 topn+buffer 名
        kept = [c for c in band if c in keep]          # 其中仍持有的 -> 继续持有（保持分数序）
        fresh = [c for c in band if c not in keep]     # 其余按分补足
        n_keep = len(kept)
        picks = kept + fresh[:max(0, topn - len(kept))]
    else:
        picks = ranked[:topn]
    diag = {"funded": len(funded), "tech": len(ts_map), "liq": n_liq,
            "picked_from_top40": sum(1 for c in picks if c in ts_map),
            "kept": n_keep}
    return picks, diag


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


def run(variant, close, amount, pe, roe, tech, regime_by_date, topn, dates,
        tech_use=None, tech_mode="clamp", tech_flip=(), tech_w=None,
        rebal=REBAL, cost_mode="flat", buffer=0, stats=None):
    """tech_use/tech_mode/tech_flip/tech_w：**归因与改造实验专用**（默认 = 原行为，A-H 不受影响）。

    rebal     : 调仓间隔（默认 REBAL=5）。原先写死，现可扫频。
    cost_mode : "flat"（原行为，每期固定 COST）| "turnover"（**按实际换手计费**：
                fee = 换手率 x ROUND_TRIP）。原口径等价于假设每期 100% 全额换手，
                而实测只有约 61% => 旧口径高估成本。
    buffer    : 持有缓冲区宽度（见 select_at）。0 = 原行为（掉出 topN 即卖）。
    stats     : 可选 dict；会填入 "turnover"（逐期换手率）与 "n_periods"。
    """
    tech_use = tech if tech_use is None else tech_use
    sel_dates = list(dates[::rebal])
    nav, curve = 1.0, []
    diags, alphas = [], []
    cur: set = set()                      # 持仓台账（原先没有 —— 无法算换手）
    if stats is not None:
        stats.setdefault("turnover", [])
    for d in sel_dates[:-1]:
        i = dates.get_loc(d)
        if i + rebal >= len(dates):
            break
        d2 = dates[i + rebal]

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
            picks, diag = select_at(d, close, amount, pe, roe, tech_use,
                                    TECH_ONLY_WEIGHTS[variant], "G", topn,
                                    tech_mode, tech_flip, cur, buffer)
            if diag:
                diags.append({**diag, "date": d})
        elif variant in FUND_HALF_MODE:                      # 只按 PE 分 / 只按 ROE 分
            picks, diag = select_at(d, close, amount, pe, roe, tech_use,
                                    DEFAULT_TECH_WEIGHTS, FUND_HALF_MODE[variant], topn,
                                    tech_mode, tech_flip, cur, buffer)
            if diag:
                diags.append({**diag, "date": d})
        else:
            mode = variant if variant in ("A", "B", "F", "G") else "B"
            w = (tech_w if tech_w is not None else
                 (DEFAULT_TECH_WEIGHTS if variant == "C"
                  else regime_by_date.get(d, DEFAULT_TECH_WEIGHTS)))
            picks, diag = select_at(d, close, amount, pe, roe, tech_use, w, mode, topn,
                                    tech_mode, tech_flip, cur, buffer)
            if diag:
                diags.append({**diag, "date": d})
        if len(picks) < max(1, topn // 2):
            continue

        p0 = close.loc[d, picks]
        p1 = close.loc[d2, picks]
        seg = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(seg):
            continue
        # 换手率 = 这一期**新买进**的比例（等权下即需要重新建仓的仓位占比）
        turn = (len(set(picks) - cur) / len(picks)) if cur else 1.0
        if stats is not None:
            stats["turnover"].append(turn)
        fee = (turn * ROUND_TRIP) if cost_mode == "turnover" else COST
        r = float(seg.mean()) - (fee if variant != "E" else 0.0)
        nav *= (1 + r)
        cur = set(picks)
        curve.append((d2, nav))
        uni_r = _uni_ret(close, d, d2)
        if not np.isnan(uni_r):
            alphas.append((d2, r - uni_r))
    if stats is not None:
        stats["n_periods"] = len(stats["turnover"])
    return curve, diags, alphas


def _uni_ret(close, d, d2):
    """PIT 全宇宙等权收益（同期基准）。"""
    uni = (close.loc[d2] / close.loc[d] - 1).replace([np.inf, -np.inf], np.nan).dropna()
    return float(uni.mean()) if len(uni) else np.nan


def metrics(curve, rebal=REBAL):
    if not curve:
        return {}
    ser = pd.Series(dict(curve))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    tot = ser.iloc[-1] - 1
    per = ser.pct_change().dropna()
    return {"total": round(tot * 100, 1),
            "annual": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else 0.0,
            "maxdd": round(float((ser / ser.cummax() - 1).min()) * 100, 1),
            "sharpe": round(float(per.mean() / per.std() * np.sqrt(252 / rebal)), 2)
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


def _clamped_panels(tech):
    """`score_tech` 里**归一化 + clamp 之后**的四个面板（单位权重下就是该子因子的分）。

    [!] 为什么要单独看它：第 1 层的 IC 用的是**原始因子**，而实际选票用的是
    **clamp 之后的分**。`score_tech` 把每个因子 clamp 到 [0,1]：

        v = clamp(1 - vol/0.03)        vol <= 0      -> 全部满分 1.0（并列）
        m = clamp((mom + 0.05)/0.25)   mom >= 0.20   -> 全部满分 1.0（并列）
        t = clamp((trd + 0.05)/0.10)   trd >= 0.05   -> 全部满分 1.0（并列）
        r = clamp((rev + 0.20)/0.30)   rev >= 0.10   -> 全部满分 1.0（并列）

    而 top-12 恰恰取自尾部 —— 若尾部被压成并列，从并列里挑 12 只等于**随机挑**。
    这能同时解释「IC 显著为正却打不过随机」和「动量的 IC 为负、组合显著更差」。
    """
    mom, trd, vol, rev = tech
    return {
        "mom": ((mom + 0.05) / 0.25).clip(0.0, 1.0),
        "trd": ((trd + 0.05) / 0.10).clip(0.0, 1.0),
        "vol": (1.0 - vol / 0.03).clip(0.0, 1.0),
        "rev": ((rev + 0.20) / 0.30).clip(0.0, 1.0),
    }


def _funded_ic_from_panels(close, amount, pe, roe, dates, panels):
    """在**预筛后的 funded 集**上算给定面板的逐期 IC。"""
    from scipy.stats import spearmanr
    out = {k: [] for k in panels}
    sel = list(dates[::REBAL])
    for d in sel[:-1]:
        i = dates.get_loc(d)
        if i + REBAL >= len(dates):
            break
        d2 = dates[i + REBAL]
        fwd = (close.loc[d2] / close.loc[d] - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(fwd) < 30:
            continue
        funded, _n = candidates_at(d, close, amount, pe, roe)
        if not funded:
            continue
        r_sub = fwd.reindex([r[0] for r in funded]).dropna()
        if len(r_sub) < 20:
            continue
        for k, p in panels.items():
            if d not in p.index:
                continue
            f = p.loc[d].reindex(r_sub.index).dropna()
            b = f.index.intersection(r_sub.index)
            if len(b) < 20:
                continue
            ff = f[b].to_numpy(dtype=float)
            rr = r_sub[b].to_numpy(dtype=float)
            ok = np.isfinite(ff) & np.isfinite(rr)
            if ok.sum() < 20:
                continue
            rho, _ = spearmanr(ff[ok], rr[ok])
            if np.isfinite(rho):
                out[k].append((d2, float(rho)))
    return out


def factor_ic_subset(close, amount, pe, roe, tech, dates):
    """在**预筛后的 funded 集**上算**原始因子**的逐期 IC（逻辑见 `_funded_ic_from_panels`）。"""
    return _funded_ic_from_panels(close, amount, pe, roe, dates,
                                  _factor_panels(pe, roe, tech))


def factor_layer_profile(close, amount, pe, roe, tech, dates, layers=5):
    """在 funded 集内按因子**分层**，画每层的前向收益 —— 直接检验「非单调」。

    为什么必须看这个：低波/反转在 funded 集上的 IC 显著为正（+0.055 / +0.062），
    但按它**正确排序**的 top-12 打不过随机（低波甚至显著更差 -0.736, t=-2.19）。
    IC 是全截面的**平均**排序信息，**不保证头部那一段也单调** —— 本函数把每层画出来。

    返回 {因子: [每期的各层「相对 funded 均值」的超额]}；层序 = 因子值**从小到大**，
    所以**最后一层就是选票会取的那一端**。
    """
    panels = _factor_panels(pe, roe, tech)
    prof = {k: [] for k in panels}
    sel = list(dates[::REBAL])
    for d in sel[:-1]:
        i = dates.get_loc(d)
        if i + REBAL >= len(dates):
            break
        d2 = dates[i + REBAL]
        fwd = (close.loc[d2] / close.loc[d] - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(fwd) < 30:
            continue
        funded, _n = candidates_at(d, close, amount, pe, roe)
        if not funded:
            continue
        r_sub = fwd.reindex([r[0] for r in funded]).dropna()
        if len(r_sub) < layers * 4:
            continue
        base = float(r_sub.mean())
        for k, p in panels.items():
            if d not in p.index:
                continue
            f = p.loc[d].reindex(r_sub.index).dropna()
            b = f.index.intersection(r_sub.index)
            if len(b) < layers * 4:
                continue
            order = f[b].sort_values(ascending=True).index     # 因子值 低 -> 高
            rr = r_sub[order].to_numpy(dtype=float)
            groups = np.array_split(np.arange(len(rr)), layers)
            prof[k].append([float(rr[g].mean()) - base for g in groups])
    return prof


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

    # ---- 分辨 (a)/(b)：把 IC 改在**预筛后的候选集**上重算 ----
    sub_ic = factor_ic_subset(close, amount, pe, roe, tech, dates)
    out["funded_ic"] = {}
    print()
    print("=" * 96)
    print("分辨：把 IC 改在**预筛后的 funded 集**上重算（第 1 层的 IC 是在全宇宙上算的）")
    print("  判据：全宇宙显著、funded 上塌掉 => 信息被**预筛**砍掉；")
    print("        funded 上仍显著且同号 => 问题在**切点**（top-12 太极端）")
    print("=" * 96)
    print(f"{'因子':<20}{'全宇宙IC':>10}{'t':>7}{'fundedIC':>10}{'t':>7}{'n':>6}   判定")
    print("-" * 96)
    n_funded_cut = 0
    for k, lab in FACTOR_LABELS.items():
        full = out["quantile"][k]["ic"]
        sub = _tstat(sub_ic[k], pct=False)
        out["funded_ic"][k] = sub
        ft, st = full["t"], sub["t"]
        if ft is None or st is None:
            verdict = "-"
        elif abs(ft) > thr and abs(st) < 2.0:
            verdict = "塌掉 -> 预筛砍掉了信息"
            n_funded_cut += 1
        elif abs(ft) > thr and abs(st) > 2.0 and full["mean_pct"] * sub["mean_pct"] > 0:
            verdict = "仍成立 -> 问题在切点"
        elif abs(ft) < 2.0 and abs(st) < 2.0:
            verdict = "两边都测不出"
        else:
            verdict = "符号翻转/边缘"
        print(f"{lab:<20}{full['mean_pct']:>10.4f}{ft or 0:>7.2f}"
              f"{sub['mean_pct']:>10.4f}{st or 0:>7.2f}{sub['n']:>6}   {verdict}")
    out["verdict_counts"] = {"prefilter_killed": n_funded_cut}

    # ---- 关键一步：原始因子 IC  vs  **clamp 之后的分数**的 IC ----
    # 选票用的是 clamp 后的分，不是原始因子。若 clamp 把尾部压成并列，
    # 从尾部挑 top-12 就等于在并列里随机挑 —— 这能解释「原始因子 IC 显著为正、
    # 组合却打不过随机」，也能解释「动量的 IC 为负、组合显著更差」。
    clamp_ic = _funded_ic_from_panels(close, amount, pe, roe, dates, _clamped_panels(tech))
    out["clamped_ic"] = {}
    print()
    print("=" * 96)
    print("关键：选票用的是 clamp 之后的**分**，不是原始因子 —— 两者的 IC 差多少？")
    print("  score_tech 把每个因子压到 [0,1]，尾部会**并列**（如 vol<=0 一律满分 1.0）")
    print("=" * 96)
    print(f"{'因子':<20}{'原始因子IC':>12}{'t':>7}{'clamp后IC':>11}{'t':>7}   判定")
    print("-" * 96)
    for k, lab in FACTOR_LABELS.items():
        if k not in clamp_ic:
            continue
        raw = out["funded_ic"][k]
        cl = _tstat(clamp_ic[k], pct=False)
        out["clamped_ic"][k] = cl
        if raw["t"] is None or cl["t"] is None:
            verdict = "-"
        elif abs(raw["t"]) > 2.0 and abs(cl["t"]) < 2.0:
            verdict = "clamp 把信息压平了 <== 就是这个"
        elif raw["t"] * cl["t"] < 0:
            verdict = "符号被 clamp 翻转"
        else:
            verdict = "clamp 后仍在"
        print(f"{lab:<20}{raw['mean_pct']:>12.4f}{raw['t']:>7.2f}"
              f"{cl['mean_pct']:>11.4f}{cl['t']:>7.2f}   {verdict}")

    # ---- 修法验证：把排序从 clamp 换成**保序**的归一化，技术面是否开始起作用？----
    print()
    print("=" * 96)
    print("修法验证：排序改用**保序**的归一化，技术面是否开始起作用？")
    print("  关键读数 = B_norm 减 F（F = 只用基本面）。线上 clamp 下这个差是 +0.020 (t=0.18)")
    print("  若去 clamp 后它显著偏离 0 => 技术面开始参与排序（且能看出方向是好是坏）")
    print("=" * 96)
    cF, _dF, aF = run("F", close, amount, pe, roe, tech, regime_by_date, topn, dates)
    sF = pd.Series(dict(aF)).sort_index()
    out["norm_fix"] = {}
    print(f"{'归一化':<26}{'B 总收益':>10}{'B 年化':>9}{'B alpha':>10}{'B t':>7}"
          f"{'B 减 F':>10}{'t':>7}")
    print("-" * 96)
    for name, lab, how, tmode in (("clamp", "线上原式(会并列)", None, "clamp"),
                                  ("raw", "不裁(保序)", "raw", "raw"),
                                  ("rank", "截面百分位(保序、无并列)", "rank", "raw")):
        tu = None if how is None else _norm_panels(tech, how)
        cB, _dB, aB = run("B", close, amount, pe, roe, tech, regime_by_date, topn, dates,
                          tech_use=tu, tech_mode=tmode)
        mB = metrics(cB)
        if not mB:
            continue
        mB.pop("_ser", None)
        a = np.asarray([x for _, x in aB], dtype=float)
        a_t = (float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))
               if len(a) > 2 and a.std() else 0.0)
        s = pd.concat([pd.Series(dict(aB)).sort_index().rename("x"), sF.rename("y")],
                      axis=1, sort=False).dropna()
        dd = s["x"] - s["y"]
        t = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std() else 0.0
        out["norm_fix"][name] = {"B_total": mB["total"], "B_annual": mB["annual"],
                                 "B_sharpe": mB["sharpe"],
                                 "B_alpha_mean_pct": round(float(a.mean()) * 100, 3),
                                 "B_alpha_t": round(a_t, 2),
                                 "B_minus_F_pct": round(float(dd.mean()) * 100, 3),
                                 "B_minus_F_t": round(t, 2)}
        print(f"{lab:<26}{mB['total']:>10.1f}{mB['annual']:>9.2f}"
              f"{float(a.mean()) * 100:>10.3f}{a_t:>7.2f}{float(dd.mean()) * 100:>10.3f}{t:>7.2f}")

    # ---- 单因子臂在两种归一化下 vs D ----
    print()
    print("单因子臂 vs D（同 topN 配对）：换归一化后能不能与随机区分开？")
    print(f"{'臂':<16}{'归一化':<12}{'总收益':>10}{'逐期alpha':>11}{'vs D':>10}{'t':>7}")
    print("-" * 66)
    cD, _dD, aD = run("D", close, amount, pe, roe, tech, regime_by_date, topn, dates)
    sD = pd.Series(dict(aD)).sort_index()
    out["arms_norm"] = {}
    for v in ("M_vol", "M_rev", "M_mom"):
        for name, how, tmode in (("clamp", None, "clamp"), ("rank", "rank", "raw")):
            tu = None if how is None else _norm_panels(tech, how)
            curve, _d2, alphas = run(v, close, amount, pe, roe, tech, regime_by_date,
                                     topn, dates, tech_use=tu, tech_mode=tmode)
            mm = metrics(curve)
            if not mm:
                continue
            mm.pop("_ser", None)
            a = np.asarray([x for _, x in alphas], dtype=float)
            s = pd.concat([pd.Series(dict(alphas)).sort_index().rename("x"), sD.rename("y")],
                          axis=1, sort=False).dropna()
            dd = s["x"] - s["y"]
            t = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std() else 0.0
            out["arms_norm"][f"{v}_{name}"] = {
                "total": mm["total"], "alpha_mean_pct": round(float(a.mean()) * 100, 3),
                "vs_D_pct": round(float(dd.mean()) * 100, 3), "vs_D_t": round(t, 2)}
            print(f"{v:<16}{name:<12}{mm['total']:>10.1f}{float(a.mean()) * 100:>11.3f}"
                  f"{float(dd.mean()) * 100:>10.3f}{t:>7.2f}")

    # ---- 分层收益曲线：IC 显著为正，但**头部那一层**抬起来了吗？----
    from scipy.stats import spearmanr
    LAYERS = 5
    prof = factor_layer_profile(close, amount, pe, roe, tech, dates, LAYERS)
    out["layer_profile"] = {}
    print()
    print("=" * 96)
    print(f"分层收益曲线（funded 集内按因子分 {LAYERS} 层，每层相对 funded 均值的超额 %）")
    print("  层序 = 因子值**从小到大**；**最后一层就是选票会取的那一端**")
    print("  IC 显著为正、但最后一层不抬 => **非单调**：信息在中段，头部没有")
    print("=" * 96)
    hdr = "".join(f"{'层' + str(j + 1):>9}" for j in range(LAYERS))
    print(f"{'因子':<18}{hdr}{'头-底':>9}{'t':>7}{'层序相关':>10}")
    print("-" * 96)
    for k, lab in FACTOR_LABELS.items():
        if k not in prof or not prof[k]:
            continue
        arr = np.asarray(prof[k], dtype=float)          # (n_period, layers)
        mean_layer = arr.mean(axis=0) * 100
        d_tb = arr[:, -1] - arr[:, 0]
        se = float(d_tb.std(ddof=1)) / np.sqrt(len(d_tb)) if len(d_tb) > 2 else 0.0
        t_tb = float(d_tb.mean()) / se if se else 0.0
        mono, _ = spearmanr(np.arange(1, LAYERS + 1), mean_layer)
        out["layer_profile"][k] = {
            "layers_pct": [round(float(x), 3) for x in mean_layer],
            "top_minus_bottom_pct": round(float(d_tb.mean()) * 100, 3),
            "top_minus_bottom_t": round(t_tb, 2),
            "layer_order_spearman": round(float(mono), 3) if np.isfinite(mono) else None,
            "n_periods": int(len(arr))}
        cells = "".join(f"{x:>9.3f}" for x in mean_layer)
        print(f"{lab:<18}{cells}{float(d_tb.mean()) * 100:>9.3f}{t_tb:>7.2f}"
              f"{mono:>10.2f}")

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

    # ---- topN 扫描：直接测「切点太极端」这个解释 ----
    print()
    print("=" * 96)
    print("topN 扫描：若「切点太极端」成立，放宽 topN 应让因子臂追平或超过随机")
    print("=" * 96)
    print(f"{'臂':<16}{'topN':>6}{'总收益':>10}{'逐期alpha':>11}{'vs D 同N':>11}{'t':>7}")
    print("-" * 96)
    out["topn_sweep"] = {}
    for n in (12, 20, 30):
        cD, _dD, aD = run("D", close, amount, pe, roe, tech, regime_by_date, n, dates)
        sD = pd.Series(dict(aD)).sort_index()
        for v in ("M_vol", "M_rev", "M_mom"):
            curve, _d2, alphas = run(v, close, amount, pe, roe, tech,
                                     regime_by_date, n, dates)
            m = metrics(curve)
            if not m:
                continue
            m.pop("_ser", None)
            a = np.asarray([x for _, x in alphas], dtype=float)
            s = pd.concat([pd.Series(dict(alphas)).sort_index().rename("x"),
                           sD.rename("y")], axis=1, sort=False).dropna()
            dd = s["x"] - s["y"]
            t = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std() else 0.0
            out["topn_sweep"][f"{v}_{n}"] = {
                "total": m["total"], "annual": m["annual"], "sharpe": m["sharpe"],
                "alpha_mean_pct": round(float(a.mean()) * 100, 3),
                "vs_D_pct": round(float(dd.mean()) * 100, 3), "vs_D_t": round(t, 2)}
            print(f"{v:<16}{n:>6}{m['total']:>10.1f}{float(a.mean()) * 100:>11.3f}"
                  f"{float(dd.mean()) * 100:>11.3f}{t:>7.2f}")

    # ---- 改造实验：按 §6.7 的方向改 mom/trd，**判定标准事先写死** ----
    from scipy.stats import norm as _norm

    rank_tech = _norm_panels(tech, "rank")
    drop_mt = {"mom": 0, "trd": 0, "vol": 50, "rev": 50}
    treatments = [
        ("T1 反转 mom+trd(clamp)", {"tech_flip": ("mom", "trd")}),
        ("T2 去掉 mom+trd(clamp)", {"tech_w": drop_mt}),
        ("T3 rank+去掉 mom+trd", {"tech_use": rank_tech, "tech_mode": "raw",
                                  "tech_w": drop_mt}),
        ("T4 rank+反转 mom+trd", {"tech_use": rank_tech, "tech_mode": "raw",
                                  "tech_flip": ("mom", "trd")}),
    ]
    # [!] 每加一个处理组，门槛必须跟着抬高 —— 否则就是「多试几个直到有一个显著」。
    n_tests = 2 * len(treatments)
    t_bar = float(_norm.ppf(1 - 0.025 / n_tests))

    print()
    print("=" * 96)
    print("改造实验：mom/trd 反转 or 去掉 —— 判定标准**事先声明**，不是跑完再挑")
    print("  对照 = B（当前权重 25/15/30/30，修掉缺陷，clamp）")
    print(f"  本次 {len(treatments)} 个处理 x 2 个检验 = {n_tests} 个 "
          f"=> Bonferroni 阈值 |t| > {t_bar:.2f}")
    print("  通过需**同时**满足：")
    print(f"    (1) 处理 vs 对照 配对 t > +{t_bar:.2f}")
    print(f"    (2) 处理 vs D（同门槛随机 12） 配对 t > +{t_bar:.2f}")
    print("    (3) 2023+ 子区间与全期**同号**")
    print("  任一不满足 => **不碰 selector.py**")
    print("=" * 96)

    def _paired(x: pd.Series, y: pd.Series, since=None):
        s = pd.concat([x.rename("x"), y.rename("y")], axis=1, sort=False).dropna()
        if since:
            s = s[s.index >= pd.Timestamp(since)]
        if len(s) < 10:
            return None, None, 0
        dd = s["x"] - s["y"]
        t = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if dd.std() else 0.0
        return round(float(dd.mean()) * 100, 3), round(t, 2), len(dd)

    _cc, _dd0, a_ctrl = run("B", close, amount, pe, roe, tech, regime_by_date, topn, dates)
    s_ctrl = pd.Series(dict(a_ctrl)).sort_index()
    _cd, _dd1, a_d0 = run("D", close, amount, pe, roe, tech, regime_by_date, topn, dates)
    s_d = pd.Series(dict(a_d0)).sort_index()

    out["treatment"] = {"criterion": {"n_tests": n_tests, "t_bar": round(t_bar, 3),
                                      "needs_2023_same_sign": True}}
    print(f"{'处理组':<22}{'总收益':>9}{'年化':>8}{'Sharpe':>8}{'vs对照':>9}{'t':>7}"
          f"{'2023+t':>9}{'vs D':>9}{'t':>7}   判定")
    print("-" * 100)
    for lab, kw in treatments:
        c, _d, al = run("B", close, amount, pe, roe, tech, regime_by_date, topn,
                        dates, **kw)
        m = metrics(c)
        if not m:
            continue
        m.pop("_ser", None)
        s = pd.Series(dict(al)).sort_index()
        d_c, t_c, n_c = _paired(s, s_ctrl)
        _d23, t_23, _n23 = _paired(s, s_ctrl, since="2023-01-01")
        d_d, t_d, _nd = _paired(s, s_d)
        ok = (t_c is not None and t_c > t_bar and t_d is not None and t_d > t_bar
              and t_23 is not None and (t_23 > 0) == (t_c > 0))
        out["treatment"][lab] = {
            "total": m["total"], "annual": m["annual"], "sharpe": m["sharpe"],
            "vs_ctrl_pct": d_c, "vs_ctrl_t": t_c, "vs_ctrl_2023_t": t_23,
            "vs_D_pct": d_d, "vs_D_t": t_d, "passed": bool(ok), "n": n_c}
        print(f"{lab:<22}{m['total']:>9.1f}{m['annual']:>8.2f}{m['sharpe']:>8.2f}"
              f"{(d_c or 0):>9.3f}{(t_c or 0):>7.2f}{(t_23 or 0):>9.2f}"
              f"{(d_d or 0):>9.3f}{(t_d or 0):>7.2f}   {'通过' if ok else '未通过'}")
    print()
    print("[!] 判定标准在跑之前就写死了。未通过 => 不碰 selector.py、不改线上权重。")

    # ---- 成本与换手：旧口径高估了多少？降换手值多少？----
    print()
    print("=" * 96)
    print("成本与换手：旧口径 COST=0.16%/期 **假设了 100% 全额换手**，实际换手多少？降换手值多少？")
    print(f"  自洽口径：一次完整往返 ROUND_TRIP = {ROUND_TRIP:.4%}"
          "（买 佣金0.03+滑点0.02 / 卖 佣金0.03+印花0.05+滑点0.02，纸面账户费率）")
    print("  按换手计费：fee = **实际换手率** x ROUND_TRIP")
    print("=" * 96)
    out["turnover"] = {"round_trip": ROUND_TRIP, "rebal_sweep": {}, "buffer": {}}

    def _one(rb, buf, mode):
        st: dict = {}
        c, _d, _a = run("B", close, amount, pe, roe, tech, regime_by_date, topn, dates,
                        rebal=rb, cost_mode=mode, buffer=buf, stats=st)
        mm = metrics(c, rb)
        to = float(np.mean(st.get("turnover") or [0.0]))
        mm.pop("_ser", None)
        return mm, to, len(st.get("turnover") or [])

    print()
    print("① 调仓间隔（缓冲=0）：拉长周期能省多少？")
    print(f"  {'间隔':<8}{'期数':>6}{'平均换手':>10}{'旧口径总收益':>15}{'按换手计费':>13}")
    print("  " + "-" * 62)
    for rb in (5, 10, 20):
        mf, to, n = _one(rb, 0, "flat")
        mt, _, _ = _one(rb, 0, "turnover")
        out["turnover"]["rebal_sweep"][str(rb)] = {
            "n": n, "turnover": round(to, 4), "flat_total": mf["total"],
            "turn_total": mt["total"]}
        print(f"  {rb:>2} 日{'':<3}{n:>6}{to:>10.1%}{mf['total']:>14.1f}%{mt['total']:>12.1f}%")

    print()
    print("② 持有缓冲区（间隔=5）：掉出前 topN+buffer 名才卖 —— 不牺牲信号新鲜度的降换手")
    print(f"  {'缓冲':<8}{'期数':>6}{'平均换手':>10}{'旧口径总收益':>15}{'按换手计费':>13}")
    print("  " + "-" * 62)
    for buf in (0, 6, 12, 24):
        mf, to, n = _one(5, buf, "flat")
        mt, _, _ = _one(5, buf, "turnover")
        out["turnover"]["buffer"][str(buf)] = {
            "n": n, "turnover": round(to, 4), "flat_total": mf["total"],
            "turn_total": mt["total"]}
        print(f"  {buf:>4} 只{'':<2}{n:>6}{to:>10.1%}{mf['total']:>14.1f}%{mt['total']:>12.1f}%")

    print()
    print("[!] 两列之差 = **成本口径**的影响（旧口径高估多少）；同行内跨 buffer/间隔 = 该杠杆的效果。")
    print("[!] 缓冲区的代价：它会**多持有已经掉出榜单的票**，改变的是信号暴露而不只是成本 ——")
    print("    所以 buffer 变大时收益若更差，不能只读成「成本没省下来」。")

    # ---- 配对检验：5 日 vs 20 日 ----
    # 两种频率的**期数不同**（252 vs 63），不能逐期配对。做法：把 5 日策略在
    # **每个 20 日区块**上的复合收益，与 20 日策略在同一区块的收益配对 —— 20 = 4x5，
    # 所以 20 日的调仓日必然是 5 日调仓日的子集，两者的区块边界天然对齐。
    print()
    print("配对检验 5 日 vs 20 日（按**共同的 20 日区块**配对，消除期数差异）:")
    st5b, st20b = {}, {}
    c5, _x1, _y1 = run("B", close, amount, pe, roe, tech, regime_by_date, topn, dates,
                       rebal=5, cost_mode="turnover", stats=st5b)
    c20, _x2, _y2 = run("B", close, amount, pe, roe, tech, regime_by_date, topn, dates,
                        rebal=20, cost_mode="turnover", stats=st20b)
    s5, s20 = pd.Series(dict(c5)), pd.Series(dict(c20))
    common = [t for t in s20.index if t in s5.index]
    diffs = [(s5[b] / s5[a] - 1) - (s20[b] / s20[a] - 1)
             for a, b in zip(common[:-1], common[1:])]
    if len(diffs) >= 10:
        arr = np.asarray(diffs, dtype=float)
        se = float(arr.std(ddof=1)) / np.sqrt(len(arr))
        tv = float(arr.mean()) / se if se else 0.0
        out["turnover"]["pair_5_vs_20"] = {
            "n_blocks": int(len(arr)), "diff_pct": round(float(arr.mean()) * 100, 3),
            "t": round(tv, 2)}
        print(f"  区块数 {len(arr)}   逐区块差均值 {arr.mean() * 100:+.3f}%   t = {tv:+.2f}")
        print("  负 = 5 日更差。**|t|>2 才能说「拉长周期确实更好」**；否则点估计好看也不算数。")
    else:
        print(f"  共同区块只有 {len(diffs)} 个，无法配对。")

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
