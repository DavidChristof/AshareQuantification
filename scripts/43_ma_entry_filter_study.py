"""第 43 步：个股 MA 入场过滤器对照回测 —— 「该不该用个股均线当闸门？」

## 起因（2026-09-18）

用户问：「以大盘趋势作为闸门是不是不够准确？是否应该以个股的 ma5 和 ma20 作为闸门？」

两件事需要分开：

1. **大盘闸门**已在同日关闭（`config market_trend_gate.enabled: false`）。理由与证据见
   `docs/2026-09-16-trend-gate-index-and-alignment.md`：口径对齐到「09:31 可执行」之后，
   闸门的改善消失（pit 宇宙 熔断 -31.4% -> 加闸门 -32.9%），配对检验 6/6 不显著（t 在
   -0.51 ~ +0.15），MA20 阻断率 51.1%（一枚硬币）。

2. **但个股均线不是大盘闸门的替代品** —— 它们管的是不同的东西：
     大盘闸门 = 总仓位/系统性风险（全账户一个开关）
     个股均线 = 选谁 / 何时买（每股一个开关）
   大盘跌时持仓相关性趋近 1、分散化失效，此时指数信号恰恰是**对**的工具。
   换成个股均线等于**放弃仓位控制**（崩盘时只要某只票在 MA5 上方就照样满仓）。

   所以本脚本回答的是**另一半**问题：把个股 MA 当成**入场过滤器**，到底有没有用。

## 既有研究留下的空白（本脚本要补的）

`docs/2026-09-11-score-lag-and-entry-gate.md` 在大池 top12 上测过入场闸门：

| 入场闸门 | 平均5日 | 每期通过 |
|---|---|---|
| 不加闸门 | 0.06% | 12.0 |
| 当日涨幅<=2%（= 现有 chase_guard） | 0.21% | 9.3 |
| **等回踩：收盘<=MA5** | **0.50%** | 2.8 |

两个空白：
  (a) 那是**有幸存者偏差**的池子（+97.9pp，见 docs/2026-09-11-survivorship-bias.md）；
  (b) 只测了**回踩**方向，没测「站上均线才买」—— 而后者才是用户这次提的方向。

## 设计（三个关键选择，都是为了不让结论被混淆骗掉）

### 1. 两个方向都测，不预设结论
    站上方向：close > MA5 / close > MA20 / MA5 > MA20   （趋势/动量）
    回踩方向：close <= MA5 / close <= MA20              （不追高，与现有 entry_gate 同向）

### 2. 三种「过滤之后怎么持仓」，因为它们的混淆**不同**
    cash   : topN 不变，未通过**且未持有**的 -> 该仓位留现金（**线上闸门「不开新仓、保留持仓」
             的原义**）；已持有的继续持有。这是「只改何时买、不改选谁」的**真隔离**。
    filter : 先按分取 topN，再把未通过的剔掉 -> **只数变少**（同时改变了分散度，
             与上面那份旧研究的「每期通过 2.8」是同一口径）。
    refill : 先在**通过的集合**里排名再取 topN -> 只数基本保持，但**成员变了**
             （会补进排名更靠后的票）。
    三种都报，并给出**平均实际只数** —— 让「分散度被改变」这个混淆**可见**。
    结论只有在多种口径下同向才算数。

### 3. 两个宇宙都跑 + trd 控制组
    large（有偏，与既往结论可比） / pit（无偏，本项目的判据口径）。

    [!] **已存在的重叠**：`build_score` 里 `trd = z(close/MA20 - 1)` 权重 0.15 ——
    「close > MA20」这个过滤器与 alpha 本身**部分同向**，不控制的话会把「重复计分」
    当成「过滤器有效」。故提供 `--no-trd`：把 trd 权重置 0、其余按比例归一后重跑。

## 口径（与 scripts/37 逐字对齐，便于跨脚本对照）

- 每 `REBAL=5` 个交易日调仓；每期固定成本 `COST=0.0016`（与 docs 口径一致）。
- 选股：`score` 当期横截面 topN（默认 12），**等权**。
- 成交：`close[d] -> close[d+REBAL]`，收盘到收盘。
- **无未来函数**：过滤器在 d 日用 `close.rolling(w).mean()`，只用到 `close[d]` 及之前；
  打分同样只用到 `close[d]`；成交也在 `close[d]`。信息集与成交时点自洽。
  （注意这与「大盘闸门的 09:31 口径」不是同一回事：那个是**日内执行**，必须以 D-1 为准；
   这里是**收盘执行**，用 close[d] 不算偷看。）

## 用法

    .venv/Scripts/python.exe scripts/43_ma_entry_filter_study.py              # 主口径
    .venv/Scripts/python.exe scripts/43_ma_entry_filter_study.py --no-trd     # 去掉 trd 的控制组
    .venv/Scripts/python.exe scripts/43_ma_entry_filter_study.py --topn 12 --start 2021-07-01

输出：`results/ma_entry_filter_study.json`（`--no-trd` 时为 `..._notrd.json`）+ 控制台配对 t。
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
from quant.data.universe_pit import CSI1000, CSI500, build_mask      # noqa: E402

REBAL = 5           # 调仓周期（交易日）
COST = 0.0016       # 每期换手成本（与 scripts/37 及 docs 口径一致）

# 过滤器定义：名字 -> (方向, 判定函数(close, ma5, ma20) -> bool 面板)
# 方向只用于输出分组，不影响计算。
FILTER_DIR = {
    "above_ma5": "站上", "above_ma20": "站上", "ma5_gt_ma20": "站上",
    "below_ma5": "回踩", "below_ma20": "回踩",
}
FILTER_ORDER = ["above_ma5", "above_ma20", "ma5_gt_ma20", "below_ma5", "below_ma20"]

# 三种持仓口径
MODES = ["cash", "filter", "refill"]


# ============================================================
# 载入
# ============================================================
def _pivot(df, col, fdtype="float64"):
    p = df.pivot_table(index="date", columns="symbol", values=col).sort_index()
    p.index = pd.to_datetime(p.index)
    return p.astype(fdtype)


def load_large(cfg, start: str):
    """老 559 池（有幸存者偏差）。返回 (score用close, amount, 未掩码close)。"""
    con = sqlite3.connect(str(Path(cfg.resolve("data")) / "large_pool.db"))
    bars = pd.read_sql_query(
        "SELECT symbol,date,close,volume,amount FROM large_daily WHERE date>=?",
        con, params=[start])
    con.close()
    close = _pivot(bars, "close")
    amount = _pivot(bars, "amount")
    return close, amount, close.copy()          # large 无掩码，两者相同


def load_pit(cfg, start: str):
    """PIT 无偏宇宙：掩码之外置 NaN，排序自然只在成员内。

    [!] 关键：返回的第三个值是**未掩码**的收盘价。个股均线是它**自己价格序列**的属性，
    不该被成分变动打断 —— 若在掩码后的面板上算 `rolling(20).mean()`，一只票重新进入
    成分后的 20 天内 MA 全是 NaN（被 NaN 污染），过滤器会把它误判成「不通过」。
    所以均线在**掩码之前**算。
    """
    con = sqlite3.connect(str(Path(cfg.resolve("data")) / "full_market.db"))
    dates = pd.DatetimeIndex([r[0] for r in con.execute(
        "SELECT DISTINCT date FROM full_daily WHERE date>=? ORDER BY date", (start,))])
    mask = build_mask(con, (CSI500, CSI1000), dates)
    syms = list(mask.columns)
    ph = ",".join("?" * len(syms))
    df = pd.read_sql_query(
        f"SELECT symbol,date,close,amount FROM full_daily "
        f"WHERE date>=? AND symbol IN ({ph})", con, params=[start, *syms])
    con.close()
    close = _pivot(df, "close", "float32")
    amount = _pivot(df, "amount", "float32")
    raw = close.copy()                          # 先在掩码前留一份，用于算均线
    m = mask.reindex(close.index).fillna(False).astype(bool)
    for p in (close, amount):
        p[~m.reindex(columns=p.columns, fill_value=False)] = np.nan
    return close, amount, raw


# ============================================================
# 打分
# ============================================================
def _z(p: pd.DataFrame) -> pd.DataFrame:
    return p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0)


def build_score(close: pd.DataFrame, amount: pd.DataFrame, use_trd: bool = True) -> pd.DataFrame:
    """4 因子综合分（与 scripts/33/37 口径一致，便于跨宇宙跨脚本对照）。

    use_trd=False：把 `trd`（close/MA20-1，权重 0.15）置 0，其余三项**按比例归一**
    （0.25/0.30/0.30 -> 各除以 0.85）。用于剔除「过滤器与 alpha 同向」的重复计分。
    """
    ret1 = close.pct_change(fill_method=None)
    w = {"mom": 0.25, "trd": 0.15 if use_trd else 0.0, "vol": 0.30, "rev": 0.30}
    tot = sum(w.values())
    return ((w["mom"] / tot) * _z(close.pct_change(20))
            + (w["trd"] / tot) * (_z(close / close.rolling(20).mean() - 1) if use_trd else 0.0)
            + (w["vol"] / tot) * _z(-ret1.rolling(20).std())
            + (w["rev"] / tot) * _z(close.shift(60) / close - 1))


def build_filters(raw_close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """在**未掩码**的收盘价上算均线，返回 名字 -> 是否通过的 bool 面板。"""
    c = raw_close
    ma5 = c.rolling(5).mean()
    ma20 = c.rolling(20).mean()
    return {
        "above_ma5":   (c > ma5),
        "above_ma20":  (c > ma20),
        "ma5_gt_ma20": (ma5 > ma20),
        "below_ma5":   (c <= ma5),
        "below_ma20":  (c <= ma20),
    }


# ============================================================
# 回测
# ============================================================
def run(score: pd.DataFrame, close: pd.DataFrame, topn: int,
        mode: str = "base", filt: pd.DataFrame | None = None):
    """每 REBAL 日调仓，等权 topN。

    mode:
        base   : 不过滤
        cash   : topN 不变；未通过**且未持有**的仓位留现金（线上闸门语义）
        filter : 先 topN 再剔掉未通过的（只数变少）
        refill : 先在通过集合里排名再取 topN（只数保持、成员改变）

    Returns:
        (metrics, per_period, avg_n_invested)
        per_period 是每期收益 Series（index=期末日）—— 做**配对** t 检验用。
        只看总收益会被噪声骗：scripts/37 实测 3 个日收益相关 0.96 的指数，能把同一条
        闸门的总收益给出 11pp 的差距。
    """
    dates = score.index
    nav, curve = 1.0, []
    cur: list[str] = []
    n_inv: list[int] = []

    for i in range(0, len(dates) - REBAL, REBAL):
        d, d2 = dates[i], dates[i + REBAL]
        s = score.loc[d].dropna()
        s = s[np.isfinite(s)]
        if len(s) < topn:
            continue

        pos: list[str] = []          # 实际投钱的票
        weight_scale = 1.0           # 实际投入的仓位占比（cash 模式下 <1）

        if mode == "base" or filt is None:
            pos = list(s.nlargest(topn).index)
        else:
            f = filt.loc[d] if d in filt.index else None
            passed = (f.reindex(s.index).fillna(False).astype(bool)
                      if f is not None else pd.Series(True, index=s.index))
            if mode == "cash":
                # topN 不变；未通过且未持有的 -> 现金；已持有的继续持有（= 不开新仓、保留持仓）
                picks = list(s.nlargest(topn).index)
                pos = [p for p in picks if bool(passed.get(p, False)) or p in cur]
                weight_scale = len(pos) / float(topn) if topn else 0.0
            elif mode == "filter":
                picks = list(s.nlargest(topn).index)
                pos = [p for p in picks if bool(passed.get(p, False))]
                weight_scale = 1.0
            else:                       # refill
                s2 = s[passed.reindex(s.index).fillna(False).astype(bool)]
                pos = list(s2.nlargest(topn).index) if len(s2) else []
                weight_scale = 1.0

        if not pos:
            curve.append((d2, nav))
            n_inv.append(0)
            continue

        p0, p1 = close.loc[d, pos], close.loc[d2, pos]
        seg = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(seg):
            continue
        # cash 模式下没投出去的那部分仓位不产生收益（留现金），故按占比缩放
        r = float(seg.mean()) * weight_scale - COST
        nav *= (1 + r)
        cur = pos
        curve.append((d2, nav))
        n_inv.append(len(pos))

    if not curve:
        return {}, pd.Series(dtype=float), 0.0
    ser = pd.Series(dict(curve))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    tot = ser.iloc[-1] - 1
    per = ser.pct_change().dropna()
    metrics = {"total": round(tot * 100, 1),
               "annual": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else 0.0,
               "maxdd": round(float((ser / ser.cummax() - 1).min()) * 100, 1),
               "sharpe": round(float(per.mean() / per.std() * np.sqrt(252 / REBAL)), 2)
               if per.std() else 0.0}
    return metrics, per, round(float(np.mean(n_inv)), 1) if n_inv else 0.0


def paired(base: pd.Series, other: pd.Series) -> dict | None:
    """配对 t 检验：同一批调仓期、只差过滤器。逐期收益之差做单样本 t。"""
    if base is None or other is None or base.empty or other.empty:
        return None
    j = pd.concat([base.rename("b"), other.rename("o")], axis=1).dropna()
    if len(j) < 3:
        return None
    d = j["o"] - j["b"]
    se = float(d.std(ddof=1)) / np.sqrt(len(d))
    t = float(d.mean()) / se if se else 0.0
    return {"n": int(len(j)), "mean_diff_pct": round(float(d.mean()) * 100, 4),
            "t": round(t, 2), "se_pct": round(se * 100, 4)}


def yearly(per: pd.Series) -> dict:
    if per is None or per.empty:
        return {}
    return {str(y): round(float((1 + g).prod() - 1) * 100, 1)
            for y, g in per.groupby(per.index.year)}


# ============================================================
# 主流程
# ============================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--no-trd", action="store_true", help="把 trd(close/MA20-1) 权重置 0 的控制组")
    args = ap.parse_args()

    cfg = load_config()
    use_trd = not args.no_trd
    out: dict = {"config": {"start": args.start, "topn": args.topn, "rebal": REBAL,
                            "cost": COST, "use_trd": use_trd},
                 "results": {}, "paired": {}, "yearly": {}}

    print(f"[43] 个股 MA 入场过滤器对照回测  start={args.start} topn={args.topn} "
          f"REBAL={REBAL} COST={COST} trd={'on' if use_trd else 'OFF(控制组)'}")

    for uni, loader in (("large", load_large), ("pit", load_pit)):
        print(f"\n[43] ==== 宇宙 {uni} ====")
        close, amount, raw = loader(cfg, args.start)
        score = build_score(close, amount, use_trd=use_trd)
        filts = build_filters(raw)
        print(f"[43] 面板 {close.shape[0]} 日 x {close.shape[1]} 只")

        # 基准
        m0, per0, n0 = run(score, close, args.topn, "base", None)
        print(f"  {'base':<14} {'-':<4} total={m0.get('total'):>7} "
              f"annual={m0.get('annual'):>7} maxdd={m0.get('maxdd'):>7} "
              f"sharpe={m0.get('sharpe'):>6} n_hold={n0}")
        out["results"][f"{uni}|base"] = {**m0, "n_hold": n0}
        out["yearly"][f"{uni}|base"] = yearly(per0)

        for name in FILTER_ORDER:
            for mode in MODES:
                m, per, nh = run(score, close, args.topn, mode, filts[name])
                key = f"{uni}|{name}|{mode}"
                out["results"][key] = {**m, "n_hold": nh}
                p = paired(per0, per)
                if p:
                    out["paired"][key] = p
                if mode == "cash":
                    out["yearly"][key] = yearly(per)
                    pv = (f"t={p['t']:+.2f} n={p['n']}" if p else "t=n/a")
                    print(f"  {name:<14} {mode:<4} total={m.get('total'):>7} "
                          f"annual={m.get('annual'):>7} maxdd={m.get('maxdd'):>7} "
                          f"sharpe={m.get('sharpe'):>6} n_hold={nh:<5} {pv}")
                else:
                    pv = (f"t={p['t']:+.2f}" if p else "t=n/a")
                    print(f"  {'':<14} {mode:<4} total={m.get('total'):>7} "
                          f"annual={m.get('annual'):>7} maxdd={m.get('maxdd'):>7} "
                          f"sharpe={m.get('sharpe'):>6} n_hold={nh:<5} {pv}")

    # ---- 汇总：只有**多种口径同向**才算数 ----
    print("\n[43] ==== 汇总（每种过滤器 3 种口径的 paired t）====")
    print(f"  {'过滤器':<13}{'方向':<6}{'cash t':>13}{'filter t':>13}{'refill t':>13}   判定")
    for name in FILTER_ORDER:
        cell, ts = [], []
        for mode in MODES:
            lp = out["paired"].get(f"large|{name}|{mode}")
            pp = out["paired"].get(f"pit|{name}|{mode}")
            if lp and pp:
                cell.append(f"{lp['t']:+.2f}/{pp['t']:+.2f}")
                ts += [lp["t"], pp["t"]]
            else:
                cell.append("n/a")
        # 判定口径：**6 个 t 值（2 宇宙 x 3 口径）符号一致**且全部 |t|>=2，才算「站得住」。
        # 只看一个宇宙或一种口径就下结论，正是 2026-09-16 那次把噪声当信号的错法。
        strong = [t for t in ts if abs(t) >= 2]
        same = bool(ts) and (all(t > 0 for t in ts) or all(t < 0 for t in ts))
        if same and len(strong) == len(ts):
            verdict = "** 同向且全部显著"
        elif same:
            verdict = f"同向, {len(strong)}/{len(ts)} 显著"
        else:
            verdict = "符号不一致"
        print(f"  {name:<13}{FILTER_DIR[name]:<6}{cell[0]:>13}{cell[1]:>13}{cell[2]:>13}   {verdict}"
              f"   (large/pit)")

    suffix = "_notrd" if args.no_trd else ""
    p = Path(cfg.resolve("results")) / f"ma_entry_filter_study{suffix}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[43] 已落盘 -> {p.name}")
    print("[43] 注：这只说明「过滤器的择时贡献」，不代表策略赚钱；large 池有幸存者偏差。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
