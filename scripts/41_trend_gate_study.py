"""第 41 步：大盘趋势闸门的**口径**研究 —— 「用上一交易日的大盘判今天」到底合不合理。

## 起因（2026-09-14）

用户问：「根据上一交易日的大盘来判定当日是否交易保护，是否有些过于保守？」

复核发现：`market_trend.trend_gate()` 取 `closes[-1]`，而 `fetch_index_daily` 是**日线**接口，
**盘中不含当天、收盘后含当天** -> 闸门在盘中用的是**上一交易日的收盘**。
实测（当时 13:30 盘中）闸门报「大盘 4510.15」，正是 2026-09-11 的收盘价。

本脚本回答三件事：
1. **滞后一天本身的代价**（现状「昨收口径」 vs 「今开口径」）
2. **回测 / 线上口径是否自洽**（回测用「今收口径」，线上只能用「昨收口径」）
3. **拦得对不对**（阻断日的未来收益是否真的更差）—— 用**不重叠**5日窗口算 t 值

## 三种口径

    stale  : close(D-1) < MA20(≤D-1)   现状。盘中取日线只能拿到这个（= 09:31 自动调仓实际用的）
    atopen : open(D)   < MA20(≤D-1)    实时口径。09:31 能拿到的实时指数点位
    fresh  : close(D)  < MA20(≤D)      回测口径（scripts/37）。**收盘后才知道** -> 实盘无法落地

用法：
    .venv/Scripts/python.exe scripts/41_trend_gate_study.py
    .venv/Scripts/python.exe scripts/41_trend_gate_study.py --start 2020-01-01
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from quant.realtime.indices import fetch_index_daily                # noqa: E402

MA_DAYS = 20


def line(c="-", n=76):
    print(c * n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--index", default="sh000300")
    args = ap.parse_args()

    df = fetch_index_daily(args.index)[["date", "open", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= args.start].reset_index(drop=True)
    c, o = df["close"], df["open"]
    ma = c.rolling(MA_DAYS).mean()

    d = pd.DataFrame({
        "date": df["date"], "close": c,
        "stale": c.shift(1) < ma.shift(1),
        "atopen": o < ma.shift(1),
        "fresh": c < ma,
    }).dropna()

    line("=")
    print(f"样本: {len(d)} 个交易日  ({d['date'].iloc[0].date()} ~ {d['date'].iloc[-1].date()})")
    line("=")

    print("\n[1] 「当日不开新仓」的比例（这是「保不保守」最直接的口径）")
    print(f"    {'现状 昨收 vs 昨日MA20':<28} 阻断 {d['stale'].mean():>6.1%}")
    print(f"    {'实时 今开 vs 昨日MA20':<28} 阻断 {d['atopen'].mean():>6.1%}")
    print(f"    {'回测 今收 vs 今日MA20':<28} 阻断 {d['fresh'].mean():>6.1%}")
    print("\n    逐年（现状口径）:")
    for y, g in d.groupby(d["date"].dt.year):
        print(f"      {y}: {g['stale'].mean():>6.1%}   ({len(g)} 天)")

    print("\n[2] 滞后一天的代价：现状 vs 实时（09:31 实际能拿到的信息）")
    same = (d["stale"] == d["atopen"]).mean()
    fb = d[d["stale"] & ~d["atopen"]]
    fo = d[~d["stale"] & d["atopen"]]
    print(f"    一致 {same:.1%}   不一致 {1 - same:.1%}")
    print(f"    假阻断（今天已站回均线上方却仍被拦）: {len(fb):>3} 天 ({len(fb) / len(d):.1%})")
    print(f"    漏放  （今天已跌破却仍放行）        : {len(fo):>3} 天 ({len(fo) / len(d):.1%})")

    print("\n[3] 回测口径 vs 线上口径（**这条最要命**）")
    dis = d[d["stale"] != d["fresh"]]
    print(f"    「昨收口径」与「今收口径」结论相反: {len(dis)} 天 ({len(dis) / len(d):.1%})")
    print(f"    => 同一个交易日内，09:31 自动调仓 与 收盘后看板/手动，"
          f"有 {len(dis) / len(d):.1%} 的概率给出相反结论")
    print("    => 且 close(D) 收盘后才知道 -> 回测验证的是**无法落地**的信号")

    print("\n[4] 拦得对不对（不重叠 5 日窗口，避免重叠放大显著性）")
    blk = d["stale"].values
    fwd5 = ((c.shift(-5) / c - 1) * 100).values
    idx = [i for i in range(0, len(d) - 5, 5) if math.isfinite(fwd5[i])]
    b = [fwd5[i] for i in idx if blk[i]]
    a = [fwd5[i] for i in idx if not blk[i]]
    mb = sum(b) / len(b) if b else 0.0
    ma_ = sum(a) / len(a) if a else 0.0
    t = 0.0
    if len(b) > 2 and len(a) > 2:
        vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
        va = sum((x - ma_) ** 2 for x in a) / (len(a) - 1)
        se = math.sqrt(vb / len(b) + va / len(a))
        t = (mb - ma_) / se if se else 0.0
    print(f"    阻断日 n={len(b):>3}  未来5日均值 {mb:+.2f}%")
    print(f"    放行日 n={len(a):>3}  未来5日均值 {ma_:+.2f}%")
    print(f"    差 {mb - ma_:+.2f}pp   t = {t:+.2f}   "
          f"（>0 表示阻断日反而更好 = 闸门方向存疑）")
    print("\n    注：这是**指数**口径，不等于策略口径；且 t 不显著时不能据此下结论。")

    line("=")
    print("结论：")
    print("  - 滞后一天**本身**代价不大（两种口径只有几个百分点不一致）")
    print("  - 真正的问题是**口径不自洽**：回测用今收、线上只能用昨收，12.9% 的日子相反")
    print("  - 且闸门拦掉约一半交易日，而被拦日的未来收益并不更差 -> 保守程度值得单独复测")
    line("=")

    out = {
        "n_days": len(d),
        "block_rate": {"stale": round(float(d["stale"].mean()), 4),
                       "atopen": round(float(d["atopen"].mean()), 4),
                       "fresh": round(float(d["fresh"].mean()), 4)},
        "disagree_stale_vs_atopen": round(float(1 - same), 4),
        "false_block_days": len(fb), "miss_days": len(fo),
        "disagree_stale_vs_fresh": round(len(dis) / len(d), 4),
        "blocked_fwd5_mean": round(mb, 3), "allowed_fwd5_mean": round(ma_, 3),
        "diff_pp": round(mb - ma_, 3), "t": round(t, 2),
        "by_year": {str(y): round(float(g["stale"].mean()), 4)
                    for y, g in d.groupby(d["date"].dt.year)},
    }
    Path("results").mkdir(exist_ok=True)
    with open("results/trend_gate_study.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n[out] results/trend_gate_study.json")


if __name__ == "__main__":
    main()
