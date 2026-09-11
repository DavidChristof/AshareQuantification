"""第 32 步：换手率因子有效性检验（RankIC + 组合回测 + 与现有因子的相关性）。

问题（用户）：现有选股因子有哪些？加换手率/资金流能否提高模型强度？

本脚本只回答**可回测的那部分**（换手率）：
    1. RankIC：换手率族因子对未来 5/20 日收益的截面预测力（对比现有因子）
    2. 增量性：换手率族与现有因子的横截面相关（若高度相关 → 加进去没用）
    3. 组合回测：现有技术面组合（基准） vs 叠加换手率（加权 / 排除高换手过滤）
       策略 = 600 池 top12 等权、每 5 日调仓、含换手费 0.16%/期

数据：
    data/large_pool.db  large_daily(559 只 OHLCV) + large_turnover(换手率/流通股本，
    由 scripts/31_backfill_turnover.py 回填)

用法：
    python scripts/32_turnover_factor_test.py
    python scripts/32_turnover_factor_test.py --horizons 5 20 --topn 12
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from quant.config import load_config                                # noqa: E402
from quant.factors.analysis import (forward_returns, judge_factor,  # noqa: E402
                                    rank_ic_series, summarize_ic)

REBAL = 5               # 调仓周期（交易日）
COST = 0.0016           # 每期换手成本（与 docs 里既有口径一致）


# ============================================================
# 数据
# ============================================================
def _pivot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    p = df.pivot_table(index="date", columns="symbol", values=col).sort_index()
    p.index = pd.to_datetime(p.index)
    return p


def load_panels(db: Path
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """→ (close, volume, turnover, float_shares) 四个 date × symbol 宽表。"""
    con = sqlite3.connect(str(db))
    bars = pd.read_sql_query(
        "SELECT symbol, date, close, volume, amount FROM large_daily", con)
    tur = pd.read_sql_query(
        "SELECT symbol, date, turnover, outstanding_share FROM large_turnover", con)
    con.close()
    close, vol = _pivot(bars, "close"), _pivot(bars, "volume")
    tv, fl = _pivot(tur, "turnover"), _pivot(tur, "outstanding_share")
    logger.info("面板: close %s, turnover %s（覆盖 %d 只）",
                close.shape, tv.shape, tv.notna().any().sum())
    return close, vol, tv, fl


def build_panels(close: pd.DataFrame, vol: pd.DataFrame, tv: pd.DataFrame) -> dict:
    """换手率族 + 现有对照因子面板。全部只用当日及之前数据（无未来函数）。"""
    ret1 = close.pct_change()
    f: dict[str, pd.DataFrame] = {}

    # ---------- 换手率族（新） ----------
    f["turn1"] = tv                                          # 当日换手率
    f["turn20"] = tv.rolling(20).mean()                      # 20 日平均换手（水平）
    f["turn60"] = tv.rolling(60).mean()
    f["turn_ratio"] = tv / tv.rolling(20).mean().replace(0, np.nan)   # 异常放量
    f["turn_chg"] = (tv.rolling(5).mean() / tv.rolling(20).mean().replace(0, np.nan) - 1)
    f["turn_std20"] = tv.rolling(20).std()                   # 换手波动（投机度）
    # 注：turnover 越小越好（高换手 → 后续收益低），故取负号得到"打分方向"
    f["-turn20"] = -f["turn20"]
    f["-turn_ratio"] = -f["turn_ratio"]
    f["-turn_std20"] = -f["turn_std20"]

    # ---------- 现有因子（对照；口径对齐 selector.py / cross_dataset.py） ----------
    f["mom20"] = close.pct_change(20)
    f["rev60"] = close.shift(60) / close - 1                 # 60 日反转（跌得多→高分）
    f["vol20"] = ret1.rolling(20).std()
    f["trd"] = close / close.rolling(20).mean() - 1          # 趋势（偏离 MA20）
    f["vol_pct_20"] = vol / vol.rolling(20).mean() - 1       # 量比
    return f


# ============================================================
# IC
# ============================================================
def ic_table(panels: dict, close: pd.DataFrame, horizons: tuple[int, ...],
             subset: set[str] | None = None) -> list[dict]:
    rows = []
    for h in horizons:
        rp = forward_returns(close, h)
        for name, fp in panels.items():
            if subset and name not in subset:
                continue
            ic = rank_ic_series(fp, rp, min_stocks=30)
            rep = summarize_ic(ic)
            if rep is None:
                continue
            rows.append({"factor": name, "h": h, "mean_ic": rep["mean_ic"],
                         "icir": rep["icir"], "pos": rep["ic_positive"],
                         "n": rep["n_days"], "judge": judge_factor(rep)})
    return rows


def yearly_ic(fp: pd.DataFrame, close: pd.DataFrame, h: int) -> pd.DataFrame:
    """逐年 RankIC（稳健性）。"""
    rp = forward_returns(close, h)
    ic = rank_ic_series(fp, rp, min_stocks=30)
    df = ic.to_frame("ic")
    df["year"] = df.index.year
    return df.groupby("year")["ic"].agg(["mean", "count"]).round(4)


# ============================================================
# 组合回测
# ============================================================
def zscore_cs(panel: pd.DataFrame) -> pd.DataFrame:
    """截面标准化（每日对全池去均值除标准差）——打分用。"""
    m = panel.mean(axis=1)
    s = panel.std(axis=1).replace(0, np.nan)
    return panel.sub(m, axis=0).div(s, axis=0)


def backtest(score: pd.DataFrame, close: pd.DataFrame, topn: int = 12,
             rebal: int = REBAL, cost: float = COST) -> dict:
    """按 score 每 rebal 日取 topn 等权持有，返回净值/指标。

    口径：d 日收盘按分数选 topn → 等权持有到 d2=d+rebal 日收盘。
    **组合区间收益 = 成分股区间收益的算术平均**（等权），不是连乘
    （连乘会把 12 只股票的收益复利成 −11%，是错的）。
    """
    dates = score.index
    nav, curve, cur = 1.0, [], None
    for i in range(0, len(dates) - rebal, rebal):
        d, d2 = dates[i], dates[i + rebal]
        s = score.loc[d].dropna()
        s = s[np.isfinite(s)]
        if len(s) < topn:
            continue
        picks = s.nlargest(topn).index
        p0, p1 = close.loc[d, picks], close.loc[d2, picks]
        seg = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(seg) == 0 or (p0 <= 0).any():
            continue
        r = float(seg.mean())                       # 等权组合区间收益
        turn = 1.0 if cur is None else 1 - len(set(picks) & set(cur)) / topn
        r -= cost * turn
        cur = list(picks)
        nav *= (1 + r)
        curve.append((d2, nav))
    if not curve:
        return {}
    ser = pd.Series(dict(curve))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    total = ser.iloc[-1] - 1
    ann = (1 + total) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    dd = float((ser / ser.cummax() - 1).min())
    per = ser.pct_change().dropna()
    sharpe = float(per.mean() / per.std() * np.sqrt(252 / rebal)) if per.std() else 0.0
    return {"total": round(total * 100, 1), "annual": round(ann * 100, 2),
            "maxdd": round(dd * 100, 1), "sharpe": round(sharpe, 2),
            "calmar": round(ann / abs(dd), 2) if dd else 0.0,
            "n_periods": len(ser)}


def yearwise(score: pd.DataFrame, close: pd.DataFrame, topn: int, rebal: int = REBAL,
             cost: float = COST) -> pd.DataFrame:
    """逐年拆解回测（稳健性）：每年单独跑一遍，看是否只靠某一年。"""
    rows = []
    for y in sorted({d.year for d in score.index}):
        sc = score[score.index.year == y]
        sub_close = close.loc[close.index.isin(sc.index)]
        m = backtest(sc, sub_close, topn=topn, rebal=rebal, cost=cost)
        if m:
            rows.append({"year": y, "total%": m["total"], "maxdd%": m["maxdd"],
                         "sharpe": m["sharpe"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--start", default="2021-07-01", help="回测起点（预留因子预热期）")
    args = ap.parse_args()

    cfg = load_config()
    db = Path(cfg.resolve("data")) / "large_pool.db"
    close, vol, tv, fl = load_panels(db)
    panels = build_panels(close, vol, tv)

    # ---------- 1. RankIC ----------
    print("\n==================== 换手率族 vs 现有因子：RankIC ====================")
    print("判定：|IC|>0.05 有效 · >0.10 优秀 · ICIR>0.5 高质量（负号=反向因子，越大越好用绝对值看）")
    rows = ic_table(panels, close, tuple(args.horizons))
    df = pd.DataFrame(rows).sort_values(["h", "mean_ic"])
    for h in args.horizons:
        sub = df[df["h"] == h].copy()
        if sub.empty:
            continue
        print(f"\n--- 预测期 {h} 日 ---")
        sub["|IC|"] = sub["mean_ic"].abs()
        sub = sub.sort_values("|IC|", ascending=False)
        print(sub[["factor", "mean_ic", "icir", "pos", "n", "judge"]].to_string(index=False))

    # ---------- 2. 与现有因子的相关（增量性） ----------
    print("\n==================== 增量性：换手率族 vs 现有因子的截面相关 ====================")
    base = ["mom20", "rev60", "vol20", "trd", "vol_pct_20"]
    new = ["turn20", "turn_ratio", "turn_std20", "turn_chg"]
    corr_rows = []
    for n in new:
        r = {}
        for b in base:
            a_, b_ = panels[n].stack(), panels[b].stack()
            j = pd.concat([a_, b_], axis=1).dropna()
            r[b] = round(float(j.iloc[:, 0].corr(j.iloc[:, 1], method="spearman")), 2) if len(j) > 100 else None
        corr_rows.append({"因子": n, **r})
    print(pd.DataFrame(corr_rows).to_string(index=False))
    print("（|相关| > 0.7 视为信息重复 → 加进去基本没用）")

    # ---------- 3. 逐年稳健性（最佳换手率因子） ----------
    best = df[(df["h"] == args.horizons[0])].reindex(
        df[df["h"] == args.horizons[0]]["mean_ic"].abs().sort_values(ascending=False).index
    )["factor"].iloc[0]
    if best in panels:
        print(f"\n==================== 逐年 RankIC（{best}, h={args.horizons[0]}）====================")
        print(yearly_ic(panels[best], close, args.horizons[0]).to_string())

    # ---------- 4. 组合回测 ----------
    print(f"\n===== 组合回测：600 池 top{args.topn} 等权 · 每 {REBAL} 日调仓 · "
          f"含费 {COST:.2%}/期 =====")
    base_score = (0.25 * zscore_cs(panels["mom20"]) + 0.15 * zscore_cs(panels["trd"])
                  + 0.30 * zscore_cs(-panels["vol20"]) + 0.30 * zscore_cs(panels["rev60"]))
    start = pd.Timestamp(args.start)
    variants = {"A 基准（现有 4 因子）": base_score}
    for w in (0.2, 0.4):
        variants[f"B +{int(w * 100)}% 低换手（-turn20）"] = base_score + w * zscore_cs(-panels["turn20"])
    variants["C 基准 + 排除换手率前 30%"] = base_score.where(
        panels["turn20"].rank(axis=1, pct=True) <= 0.7)
    variants["D 仅低换手（-turn20 单因子）"] = zscore_cs(-panels["turn20"])

    res = {}
    cols = close.columns                    # 换手率表含个别不在日线表的代码 → 对齐列
    for name, sc in variants.items():
        m = backtest(sc.loc[start:].reindex(columns=cols),
                     close.loc[start:], topn=args.topn)
        res[name] = m
        if m:
            print(f"{name:<28} 总收益 {m['total']:>7.1f}%  年化 {m['annual']:>6.2f}%  "
                  f"最大回撤 {m['maxdd']:>6.1f}%  Sharpe {m['sharpe']:>5.2f}  期数 {m['n_periods']}")

    # ---------- 5. 稳健性：单因子 D 的 Sharpe 1.19 是真的吗？ ----------
    print("\n==================== 稳健性检验（先证伪再上线）====================")
    # 线上口径 = 选股时点的**当日成交额** ≥ 1 亿（selector.MIN_AMOUNT），不是 20 日均额。
    # 两者结果差很多（当日额 −3.4% vs 20日均额 −22.8%），必须用线上口径。
    mask = (close * vol).loc[start:] >= 1e8
    n_ok = mask.sum(axis=1)
    print(f"\n【5.1】叠加线上同款流动性过滤（**当日成交额** ≥ 1 亿，与 selector.MIN_AMOUNT 一致）"
          f"后重跑：\n      可交易只数：中位 {int(n_ok.median())} / 559（最少 {int(n_ok.min())}）")

    def zf(p):
        """截面标准化必须在**同一个可交易宇宙内**做，否则分数口径不一致。"""
        return zscore_cs(p.loc[start:].where(mask))

    turn_rank = tv.reindex(columns=cols).loc[start:].rank(axis=1, pct=True)
    base_f = (0.25 * zf(panels["mom20"]) + 0.15 * zf(panels["trd"])
              + 0.30 * zf(-panels["vol20"]) + 0.30 * zf(panels["rev60"]))
    filt = {
        "A 基准 + 流动性": base_f,
        "B +10% 低换手": base_f + 0.1 * zf(-panels["turn20"]),
        "C 排除换手前 30%": base_f.where(turn_rank <= 0.7),
        "D 仅低换手": zf(-panels["turn20"]),
    }
    res_f = {}
    for name, sc in filt.items():
        m = backtest(sc.reindex(columns=cols), close.loc[start:], topn=args.topn)
        res_f[name] = m
        if m:
            print(f"  {name:<24} 总收益 {m['total']:>7.1f}%  年化 {m['annual']:>6.2f}%  "
                  f"回撤 {m['maxdd']:>6.1f}%  Sharpe {m['sharpe']:>5.2f}")

    print("\n【5.2】市值暴露（组合的流通市值中位数，亿元）——低换手是不是「小市值」的马甲？")
    mcap = close * fl                            # 流通市值（close 未复权 → 近似）
    rows_mc = []
    for d in list(base_score.loc[start:].index)[::20]:
        for tag, sc in [("全池", None), ("基准 top12", base_score),
                        ("低换手 top12", zscore_cs(-panels["turn20"]))]:
            if sc is None:
                v = mcap.loc[d].median()
            else:
                s = sc.loc[d].dropna()
                s = s[np.isfinite(s)]
                if len(s) < args.topn:
                    continue
                v = mcap.loc[d, s.nlargest(args.topn).index].median()
            rows_mc.append({"date": str(d)[:10], "组": tag, "流通市值中位(亿)": round(v / 1e8, 1)})
    mc = pd.DataFrame(rows_mc).groupby("组")["流通市值中位(亿)"].mean().round(1)
    print(mc.to_string())

    print(f"\n【5.3】逐年拆解（top{args.topn} 等权，含费）：")
    for name, sc in [("A 基准", base_score), ("B +20%低换手", base_score + 0.2 * zscore_cs(-panels["turn20"])),
                     ("D 仅低换手", zscore_cs(-panels["turn20"]))]:
        yw = yearwise(sc.loc[start:].reindex(columns=cols), close.loc[start:], args.topn)
        print(f"  {name}: " + " | ".join(
            f"{int(r.year)} {r['total%']:+.0f}%" for _, r in yw.iterrows()))

    print("\n【5.4】分半样本（前段 2021-07~2024-01 / 后段 2024-01~2026-09）：")
    mid = pd.Timestamp("2024-01-01")
    for name, sc in [("A 基准", base_score), ("B +20%低换手", base_score + 0.2 * zscore_cs(-panels["turn20"])),
                     ("D 仅低换手", zscore_cs(-panels["turn20"]))]:
        segs = []
        for a, b in [(start, mid), (mid, None)]:
            sub = sc.loc[a:b] if b is not None else sc.loc[a:]
            cl_ = close.loc[sub.index]
            m = backtest(sub.reindex(columns=cols), cl_, topn=args.topn)
            segs.append(f"{m['total']:+.0f}%/S{m['sharpe']:.2f}" if m else "n/a")
        print(f"  {name}: 前段 {segs[0]}   后段 {segs[1]}")

    out = Path(cfg.resolve("results")) / "turnover_factor_test.json"
    out.write_text(json.dumps({"ic": rows, "corr": corr_rows, "backtest": res,
                               "backtest_liquidity_filter": res_f,
                               "mcap_median": mc.to_dict()},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("结果已写入 %s", out)


if __name__ == "__main__":
    main()
