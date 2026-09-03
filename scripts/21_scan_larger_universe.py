"""第 21 步：更大股票池的因子 IC 预扫描（低成本判断「扩池值不值」）。

背景：40 只同质化大盘股的横截面 RankIC 全部≈0。怀疑瓶颈是「横截面太窄/风格同质」，
本脚本把截面扩到 ~100 只（现池 40 + 沪深300 补抽 60），离线重算单因子 IC：

    - 若某些因子在大池 |mean_ic| 明显上升(≥0.03) → 值得把模型切换到大横截面
    - 若仍全部≈0 → 说明价量因子本身无信息，扩池也不值得，就此收手（省下大改）

下载结果会缓存到 results/large_scan_cache.pkl（重跑免下载）。

用法：python scripts/21_scan_larger_universe.py [--limit 60] [--no-download 用缓存]
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                                    # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                   # noqa: E402
from quant.data.fetcher import fetch_daily                             # noqa: E402
from quant.data.loader import load_all                                 # noqa: E402
from quant.data.universe import fetch_hs300                            # noqa: E402
from quant.factors.analysis import (                                   # noqa: E402
    _prepare_panels, build_factor_panels, forward_returns, judge_factor,
    rank_ic_series, summarize_ic,
)
from quant.factors.alpha101 import mine_factor_panels                  # noqa: E402

CACHE = Path("results/large_scan_cache.pkl")
parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=60, help="额外下载沪深300数量")
parser.add_argument("--no-download", action="store_true", help="用缓存不下载")
args = parser.parse_args()

cfg = load_config()
start = cfg["data"]["start_date"]
end = datetime.now().strftime("%Y-%m-%d")
logger = logging.getLogger("scan")


def scan(data: dict, tag: str) -> pd.DataFrame:
    """对给定横截面计算 9 基础因子 + ~30 挖掘因子的 RankIC 报告。"""
    factors = {**build_factor_panels(data), **mine_factor_panels(data)}
    close_panel, _ = _prepare_panels(data)
    rows = []
    for name, fpanel in factors.items():
        for h in (5, 20):
            ics = rank_ic_series(fpanel, forward_returns(close_panel, h))
            rep = summarize_ic(ics)
            if rep is None:
                continue
            rep["factor"], rep["horizon"], rep["judge"] = name, h, judge_factor(rep)
            rows.append(rep)
    df = pd.DataFrame(rows)
    df = df[["factor", "horizon", "mean_ic", "icir", "ic_positive",
             "abs_ic", "n_days", "judge"]]
    return df.sort_values(["horizon", "abs_ic"], ascending=[True, False]).reset_index(drop=True)


def main():
    base = load_all(cfg)
    print(f"[现池] 已加载 {len(base)} 只（含内存数据，不重复下载）")

    # ---------- 下载补充股票 ----------
    pool_codes = []
    if not args.no_download:
        hs = fetch_hs300()
        existing = set(base)
        extra = sorted({c["code"] for c in hs} - existing)
        random.Random(42).shuffle(extra)
        pool_codes = extra[:args.limit]
        print(f"[下载] 沪深300 补抽 {len(pool_codes)} 只 → 总截面 {len(base) + len(pool_codes)}")

    new_data: dict = {}
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            cache = pickle.load(f)
        if not args.no_download and cache.get("codes") == pool_codes:
            new_data = cache["data"]
            print("[缓存] 命中，直接复用已下载数据")
        elif args.no_download:
            new_data = cache.get("data", {})
            print("[缓存] 使用缓存数据（--no-download）")
    if not new_data and pool_codes:
        done = 0
        def _one(code: str):
            try:
                df = fetch_daily(code, start, end)
                return code, df
            except Exception as exc:  # noqa: BLE001
                logger.error("%s 失败: %s", code, exc)
                return code, None
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_one, c): c for c in pool_codes}
            for fut in as_completed(futs):
                code, df = fut.result()
                done += 1
                if df is not None and not df.empty:
                    new_data[code] = df
                if done % 10 == 0 or done == len(pool_codes):
                    print(f"  下载进度 {done}/{len(pool_codes)}（成功 {len(new_data)}）", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump({"codes": pool_codes, "data": new_data}, f)
        print(f"[下载完成] 成功 {len(new_data)}/{len(pool_codes)} 只")

    large = {**base, **new_data}
    print(f"[截面] 现池 {len(base)} 只 vs 大池 {len(large)} 只\n")

    # ---------- 两侧因子 IC ----------
    print("正在算 40 只池因子 IC ...")
    df40 = scan(base, "base40")
    print("正在算 大池因子 IC ...")
    dfL = scan(large, "large")

    for h in (5, 20):
        print(f"\n{'='*70}\n预测期 {h} 日：|mean_ic| 最大的 12 个因子\n{'='*70}")
        cols = ["factor", "mean_ic", "icir", "ic_positive", "abs_ic", "n_days", "judge"]
        a = df40[df40.horizon == h].set_index("factor")
        b = dfL[dfL.horizon == h].set_index("factor")
        merged = a[["mean_ic", "icir", "judge"]].join(
            b[["mean_ic", "icir", "judge"]], lsuffix="_40", rsuffix="_大", how="outer")
        merged = merged.assign(
            diff=merged["mean_ic_大"].fillna(0) - merged["mean_ic_40"].fillna(0))
        top = merged.reindex(merged["abs_ic"].sort_values(ascending=False).index) \
            if "abs_ic" in merged.columns else merged
        # 用大池 abs_ic 排序展示
        merged["abs_ic大"] = b["abs_ic"].reindex(merged.index).fillna(0)
        merged = merged.sort_values("abs_ic大", ascending=False).head(12)
        print(f"{'因子':<16}{'IC_40':>8}{'IC_大池':>9}{'差值':>8}{'ICIR_40':>8}{'ICIR_大池':>9}  判定(大池)")
        for name, r in merged.iterrows():
            print(f"{name:<16}{r['mean_ic_40']:>8.4f}{r['mean_ic_大']:>9.4f}"
                  f"{r['diff']:>8.4f}{r['icir_40']:>8.3f}{r['icir_大']:>9.3f}  {r['judge_大']}")

    # 大池里达到「可用」阈值(≥0.03)的因子汇总
    for h in (5, 20):
        hit = dfL[(dfL.horizon == h) & (dfL["mean_ic"].abs() >= 0.03)]
        print(f"\n[结论 h={h}] 大池中 |IC|>=0.03 的因子数: {len(hit)}/{(dfL.horizon==h).sum()}")
        if len(hit):
            print(hit.sort_values("abs_ic", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
