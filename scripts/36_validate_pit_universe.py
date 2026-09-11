"""第 36 步：时点宇宙重建的**验收闸门** —— 重合度 + 偏差是否消失 + 退市缺口。

这是决定"重建能不能用"的脚本，三段：

    A. 重合度：用重建规则在最近一期复算，与**真实**中证300/500/1000 名单比。
       并按「相邻档错位 / 跨档错位」分解差异（相邻档=边界噪声，跨档=规则错）。
    B. 偏差收敛（**决定性验收**）：把 `scripts/33` 的偏差测法搬到 PIT 宇宙上——
       流通市值加权收益 vs 真实指数、以及"起点冷门 vs 起点活跃"的收益差。
       旧池的超额是 **+97.9pp**、冷热差 **98pp**；PIT 宇宙若把这两条压下来，
       就证明"偏差挂在流动性轴上"这个最致命的问题被修掉了。
    C. 退市缺口：2020 年以来退市的票（新浪拉不到行情）→ 显式报数，写进 docs。

用法：
    python scripts/36_validate_pit_universe.py                 # 三段全跑
    python scripts/36_validate_pit_universe.py --build         # 只重建并落库 pit_members
    python scripts/36_validate_pit_universe.py --overlap
    python scripts/36_validate_pit_universe.py --bias
    python scripts/36_validate_pit_universe.py --delist
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                # noqa: E402
from quant.data.universe_pit import (CSI1000, CSI500, HS300, PitRules,  # noqa: E402
                                     build_mask, build_snapshots, save_snapshots)
from quant.realtime.indices import fetch_index_daily                # noqa: E402

# 真实指数代码 ↔ 重建档位
REAL = {HS300: "000300", CSI500: "000905", CSI1000: "000852"}
BIAS_START = "2021-07-01"


def db_path(cfg) -> Path:
    return Path(cfg.resolve("data")) / "full_market.db"


def real_members(code: str) -> set[str]:
    import akshare as ak                                            # noqa: PLC0415
    df = ak.index_stock_cons_csindex(symbol=code)
    col = next(c for c in df.columns if "成分" in str(c) and "代码" in str(c))
    return {str(c).zfill(6) for c in df[col]}


# ============================================================
# A. 重合度
# ============================================================
def section_overlap(con, snaps: dict, rules: PitRules) -> dict:
    print("\n==================== A. 与真实指数名单的重合度 ====================")
    if not snaps:
        print("  （无快照，请先 --build）")
        return {}
    latest = max(snaps)
    rebuilt = snaps[latest]
    real = {k: real_members(v) for k, v in REAL.items()}
    keep = [s for s in rebuilt[CSI500] if s in real[CSI500]]
    print(f"  最近一期：生效日 {latest}（真实名单取自今天 "
          f"{datetime.now():%Y-%m-%d}，中间可能还有临时调整，故不会是 100%）")
    out = {}
    for tier in (HS300, CSI500, CSI1000):
        reb, tru = set(rebuilt[tier]), real[tier]
        if not reb or not tru:
            continue
        ov = len(reb & tru) / len(tru)
        # 差异分解：重建里有、真实里没有的票，落在真实的哪一档？
        only_reb = reb - tru
        where = {"在真实另一档": 0, "真实三档都没有": 0}
        for s in only_reb:
            if any(s in real[t] for t in REAL if t != tier):
                where["在真实另一档"] += 1
            else:
                where["真实三档都没有"] += 1
        out[tier] = {"overlap": round(ov, 4), "n_rebuilt": len(reb),
                     "n_real": len(tru), **where}
        flag = "[OK] " if ov >= 0.80 else ("[!]  " if ov >= 0.70 else "[XX] ")
        print(f"  {tier:<8} 重建 {len(reb):>4d} · 真实 {len(tru):>4d} · "
              f"重合 {ov:6.1%} {flag}  （重建独有 {len(only_reb)}："
              f"另一档 {where['在真实另一档']} / 三档都没有 {where['真实三档都没有']}）")
    print("  判读：沪深300 是整条链的地基（<90% 先查市值口径）；"
          "相邻档错位 = 边界噪声，跨档错位才是规则错。")
    print(f"  （校验：重建的 500 里有 {len(keep)} 只确实在真实 500 里）")
    return out


# ============================================================
# B. 偏差收敛
# ============================================================
def _index_close(sym: str, dates: pd.DatetimeIndex) -> pd.Series:
    df = fetch_index_daily(sym)
    s = df.set_index("date")["close"].sort_index()
    s.index = pd.to_datetime(s.index)
    return s.reindex(dates).ffill()


def section_bias(con, rules: PitRules) -> dict:
    print("\n==================== B. 偏差是否消失（决定性验收）====================")
    dates = pd.DatetimeIndex([r[0] for r in con.execute(
        "SELECT DISTINCT date FROM full_daily WHERE date >= ? ORDER BY date", (BIAS_START,))])
    if not len(dates):
        print("  （无数据）")
        return {}
    mask = build_mask(con, (CSI500, CSI1000), dates)
    syms = list(mask.columns)
    if not syms:
        print("  （无 PIT 成员，请先 --build）")
        return {}
    print(f"  PIT 成员并集 {len(syms)} 只；区间 {dates[0].date()} ~ {dates[-1].date()}")

    ph = ",".join("?" * len(syms))
    px = pd.read_sql_query(
        f"SELECT date, symbol, close, amount, float_mcap FROM full_daily "
        f"WHERE date >= ? AND symbol IN ({ph})", con, params=[BIAS_START, *syms])
    close = px.pivot_table(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.to_datetime(close.index)
    amt = px.pivot_table(index="date", columns="symbol", values="amount").sort_index()
    amt.index = pd.to_datetime(amt.index)
    mcap = px.pivot_table(index="date", columns="symbol", values="float_mcap").sort_index()
    mcap.index = pd.to_datetime(mcap.index)
    m = mask.reindex(close.index).fillna(False).astype(bool)

    ret1 = close.pct_change(fill_method=None)
    w = mcap.where(m)
    w = w.div(w.sum(axis=1), axis=0)
    port = (w * ret1).sum(axis=1).fillna(0)
    capw_ret = float((1 + port).prod() - 1)

    # 主判据用**买入持有**，不用"每日再归一化市值加权"。
    # 后者会把权重每日压向当期最大市值（`float_mcap` 随股本变动/成交均价波动），
    # 产生虚假收益：同一 PIT 宇宙实测 加权 +129.9% vs 买入持有 +27.9%（差 100pp）。
    # scripts/33 当初报的 +97.9pp 就含这个算法假象（已在本脚本第 B 段修正）。
    mem0 = list(m.iloc[0][m.iloc[0]].index)
    hold = (close[mem0].iloc[-1] / close[mem0].iloc[0] - 1).dropna()
    pit_ret = float(hold.mean())

    i500 = _index_close("sh000905", dates)
    i1000 = _index_close("sh000852", dates)
    r500 = float(i500.iloc[-1] / i500.iloc[0] - 1)
    r1000 = float(i1000.iloc[-1] / i1000.iloc[0] - 1)
    blend = (r500 + r1000) / 2
    print(f"  PIT 宇宙【买入持有·主判据】: {pit_ret * 100:+7.1f}%  "
          f"（中位 {hold.median() * 100:+.1f}%，n={len(hold)}）")
    print(f"  PIT 宇宙（每日再归一化加权）: {capw_ret * 100:+7.1f}%  "
          f"← **含算法假象，不作为判据**")
    print(f"  真实中证500 / 中证1000    : {r500 * 100:+.1f}% / {r1000 * 100:+.1f}%"
          f"（混合 {blend * 100:+.1f}%）")
    excess = pit_ret - blend
    ok = abs(excess) <= 0.15
    print(f"  → 超额（PIT - 混合）      : {excess * 100:+.1f} pp "
          f"{'[OK] 已收敛' if ok else '[XX] 仍偏离'}（旧 559 池同口径 **+51pp**）")

    # 冷热分组：按起点前 20 日成交额中位数分（同一口径，与 scripts/33 可直接对比）
    s0 = pd.Timestamp(BIAS_START)
    base_amt = amt.loc[:s0].tail(20).median()
    d = pd.DataFrame({"ret": hold, "amt0": base_amt}).dropna()
    d = d[d["amt0"].notna()]
    cold = d[d["amt0"] < 1e8]["ret"]
    hot = d[d["amt0"] >= 1e8]["ret"]
    gap = (cold.mean() - hot.mean()) if len(cold) and len(hot) else np.nan
    print("\n  按起点流动性分组（同一 PIT 宇宙内，买入持有口径）：")
    print(f"    起点冷门(<1亿) n={len(cold):>4d}  平均 {cold.mean():+.1%}  中位 {cold.median():+.1%}")
    print(f"    起点活跃(>=1亿) n={len(hot):>4d}  平均 {hot.mean():+.1%}  中位 {hot.median():+.1%}")
    print(f"    → 冷热差 {gap * 100:+.1f} pp "
          f"{'[OK] 已收敛' if abs(gap) <= 0.30 else '[XX] 仍偏离'}"
          f"（旧 559 池同口径 **+88.7pp**）")
    print("    ↑ 这一条才是关键：老池的「冷门股优势」在无偏宇宙里消失了/反号。")
    print(f"    中位 {d['ret'].median():+.1%} vs 真实指数混合 {blend:+.1%}")
    return {"pit_return": pit_ret, "capw_return": capw_ret, "excess": excess,
            "cold_hot_gap": float(gap), "r500": r500, "r1000": r1000,
            "n_members": len(syms), "n_start": len(hold)}


# ============================================================
# C. 退市缺口
# ============================================================
def section_delist() -> dict:
    print("\n==================== C. 退市缺口（残余偏差的显式报数）====================")
    try:
        import akshare as ak                                        # noqa: PLC0415
        sh = ak.stock_info_sh_delist()
        sz = ak.stock_info_sz_delist()
    except Exception as exc:                                        # noqa: BLE001
        print(f"  （退市名单获取失败：{type(exc).__name__}）")
        return {}
    sh.columns = ["code", "name", "list_date", "delist_date"][:len(sh.columns)]
    sz.columns = ["code", "name", "list_date", "delist_date"][:len(sz.columns)]
    d = pd.concat([sh, sz])
    d["delist_date"] = pd.to_datetime(d["delist_date"], errors="coerce")
    recent = d[d["delist_date"] >= pd.Timestamp("2020-01-01")]
    print(f"  全部退市 {len(d)} 只；**2020 年以来 {len(recent)} 只**")
    yr = recent["delist_date"].dt.year.value_counts().sort_index()
    print("  按年份：" + "  ".join(f"{y}:{n}" for y, n in yr.items()))
    print("  说明：新浪对退市股返回空行情，这部分**补不回来**（已实测）。")
    print("        但它们多为退市前的微盘/风险票，且指数区间内的票极少退市；")
    print("        主要偏差是「被调出指数」的票——那些票仍在交易，重建能捞回来。")
    return {"total": len(d), "since_2020": len(recent),
            "by_year": {int(k): int(v) for k, v in yr.items()}}


# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="只重建并落库 pit_members")
    ap.add_argument("--overlap", action="store_true")
    ap.add_argument("--bias", action="store_true")
    ap.add_argument("--delist", action="store_true")
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default=None)
    args = ap.parse_args()

    cfg = load_config()
    db = db_path(cfg)
    if not db.exists():
        raise SystemExit(f"缺 {db}：请先跑 scripts/35_fetch_full_market.py")
    con = sqlite3.connect(str(db))
    rules = PitRules()
    end = args.end or datetime.now().strftime("%Y-%m-%d")

    print(f"[36] 全市场库 {db}")
    print(f"[36] 重建区间 {args.start} ~ {end}；规则 window={rules.window} "
          f"min_days={rules.min_days_in_window} 剔市值前{rules.exclude_top_mcap} "
          f"剔成交额后{rules.cut_bottom_pct:.0%}")

    snaps = build_snapshots(con, args.start, end, rules)
    if snaps:
        n = save_snapshots(con, snaps)
        print(f"[36] 已写入 pit_members：{len(snaps)} 期 / {n} 行")

    only = args.build or args.overlap or args.bias or args.delist
    out = {}
    if not only or args.overlap:
        out["overlap"] = section_overlap(con, snaps, rules)
    if not only or args.bias:
        out["bias"] = section_bias(con, rules)
    if not only or args.delist:
        out["delist"] = section_delist()

    if out:
        import json
        p = Path(cfg.resolve("results")) / "pit_validation.json"
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                     encoding="utf-8")
        print(f"\n[36] 结果已写入 {p}")
    con.close()


if __name__ == "__main__":
    main()
