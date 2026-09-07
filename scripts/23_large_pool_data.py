"""第 23 步（阶段一·数据与验证）：构建「600 池」持久化日线库 + 40vs600 因子 IC 对比。

背景（见 docs/2026-09-07-change-notes.md 后续实验记录）：
    v2 短期模型训练在 40 只大盘池 → 横截面 RankIC≈0；怀疑瓶颈是「截面太窄/风格同质」。
    今日选股已用 ~600 池（现池40 ∪ 中证500/1000 中小盘），因子 IC 实证更好。
    本脚本把 ~560 只中小盘**全历史日线下载并落库**（可复用、不再依赖易被清掉的
    results/large_scan_cache.pkl），并复用 21/22 的因子 IC 分析，给出
    「现池40 vs 大池600」的横截面 IC 对比报告 —— 决定下一步是否值得把短期模型切到大截面训练。

产物：
    - results/large_scan_cache.pkl      （格式与 22 兼容：{"data": {code: df}}，选股器可直接读）
    - data/large_pool.db  表 large_daily （symbol,date,... 持久化，训练数据源候选）
    - results/large_pool_ic_compare.json（40 vs 600 因子 IC 对比报告）

用法：
    python scripts/23_large_pool_data.py                 # 下载(500:260 + 1000:300) + 落库 + IC 对比
    python scripts/23_large_pool_data.py --no-download   # 只用已有缓存，只落库 + IC 对比
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import socket
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                                    # noqa: E402

# 全局 socket 超时：避免 akshare 某个请求无限期挂起
socket.setdefaulttimeout(15)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                   # noqa: E402
from quant.data.fetcher import fetch_daily                             # noqa: E402
from quant.data.loader import load_all                                 # noqa: E402
from quant.factors.analysis import (                                   # noqa: E402
    _prepare_panels, build_factor_panels, forward_returns, judge_factor,
    rank_ic_series, summarize_ic,
)
from quant.factors.alpha101 import mine_factor_panels                  # noqa: E402

CACHE = Path("results/large_scan_cache.pkl")
POOL_DB = Path("data/large_pool.db")
REPORT = Path("results/large_pool_ic_compare.json")
# 目标：现池40 + 中证500中盘 + 中证1000小盘 ≈ 600 池（与 selection.universe=large 同源）
IDX_PLAN = [("000905", "中证500", 260), ("000852", "中证1000", 300)]

parser = argparse.ArgumentParser()
parser.add_argument("--no-download", action="store_true", help="只用已有缓存，跳过下载")
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--skip-db", action="store_true", help="不写 sqlite（只跑 IC）")
args = parser.parse_args()

cfg = load_config()
start = cfg["data"]["start_date"]
end = datetime.now().strftime("%Y-%m-%d")


# ----------------------------------------------------------------------
# 下载
# ----------------------------------------------------------------------
def fetch_components(index_code: str) -> list[str]:
    """中证指数成分；多源链（index_stock_cons → csindex），单次短超时，失败返回空。

    2026-09-07 晚实测：csindex 偶发整次挂起(>90s)，index_stock_cons 稳定(~5s)，
    故按顺序回退，各自只重试 1 次避免整晚卡死。
    """
    import akshare as ak
    for fn in (ak.index_stock_cons, ak.index_stock_cons_csindex):
        for attempt in range(2):
            try:
                df = fn(symbol=index_code)
                code_col = next((c for c in df.columns if "代码" in str(c)), df.columns[0])
                return sorted({str(v).zfill(6) for v in df[code_col]
                               if str(v).isdigit() and len(str(v)) <= 6})
            except Exception as exc:  # noqa: BLE001
                print(f"[{index_code}] 成分获取失败({fn.__name__}, {attempt + 1}/2): {exc}",
                      flush=True)
                time.sleep(2)
    return []


def _flush(cache: dict):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"data": cache}, f)


def download(codes: list[str], workers: int, cache: dict) -> None:
    """下载并就地写入 cache；每 100 只落一次盘（中断后重跑可续，进度不丢）。"""
    def _one(code: str):
        try:
            return code, fetch_daily(code, start, end)
        except Exception:  # noqa: BLE001 - 超时/失败返回 None
            return code, None

    todo = [c for c in codes if c not in cache]
    done, ok = 0, 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_one, c): c for c in todo}
        pending = set(futs)
        deadline = time.time() + len(todo) * 25 + 180
        while pending and time.time() < deadline:
            done_set, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for fut in done_set:
                code, df = fut.result()
                done += 1
                if df is not None and not df.empty:
                    ok += 1
                    df = df.copy()
                    df["symbol"] = code          # 统一 6 位
                    cache[code] = df
                if done % 100 == 0:
                    print(f"  下载 {done}/{len(todo)}（成功 {ok}）→ 落盘缓存", flush=True)
                    _flush(cache)
        if pending:
            print(f"[警告] {len(pending)} 只超时未完成，已跳过", flush=True)
    _flush(cache)


def persist_to_db(cache: dict):
    """把大池日线写入 sqlite（symbol,date 唯一），已存在跳过。"""
    POOL_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(POOL_DB))
    con.executescript("""
        CREATE TABLE IF NOT EXISTS large_daily (
            symbol TEXT NOT NULL, date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL, PRIMARY KEY(symbol, date));
    """)
    added = 0
    for code, df in cache.items():
        if df is None or df.empty:
            continue
        rows = []
        have = set(r[0] for r in con.execute(
            "SELECT date FROM large_daily WHERE symbol=?", (code,)).fetchall())
        for _, r in df.iterrows():
            d = r["date"]
            ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            if ds in have:
                continue
            rows.append((code, ds, float(r["open"]), float(r["high"]),
                         float(r["low"]), float(r["close"]),
                         float(r.get("volume", 0) or 0), float(r.get("amount", 0) or 0)))
        if rows:
            con.executemany(
                "INSERT OR REPLACE INTO large_daily "
                "(symbol,date,open,high,low,close,volume,amount) VALUES (?,?,?,?,?,?,?,?)",
                rows)
            added += len(rows)
    con.commit()
    n = con.execute("SELECT COUNT(DISTINCT symbol) FROM large_daily").fetchone()[0]
    con.close()
    print(f"[落库] 新增 {added} 行 → data/large_pool.db 累计 {n} 只", flush=True)


# ----------------------------------------------------------------------
# 因子 IC
# ----------------------------------------------------------------------
def scan(data: dict, tag: str) -> pd.DataFrame:
    factors = {**build_factor_panels(data), **mine_factor_panels(data)}
    close_panel, _ = _prepare_panels(data)
    rows = []
    for name, fpanel in factors.items():
        for h in (5, 20):
            rep = summarize_ic(rank_ic_series(fpanel, forward_returns(close_panel, h)))
            if rep is None:
                continue
            rep.update(factor=name, horizon=h, judge=judge_factor(rep))
            rows.append(rep)
    df = pd.DataFrame(rows)[["factor", "horizon", "mean_ic", "icir",
                             "ic_positive", "abs_ic", "n_days", "judge"]]
    return df.sort_values(["horizon", "abs_ic"], ascending=[True, False]).reset_index(drop=True)


def main():
    base = load_all(cfg)
    print(f"[现池] {len(base)} 只", flush=True)

    cache_data: dict = {}
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            cache_data = pickle.load(f).get("data", {})
        print(f"[缓存] 已有 {len(cache_data)} 只", flush=True)

    # ---------- 下载缺口（中证500 + 中证1000） ----------
    if not args.no_download:
        todo: list[str] = []
        have = set(base) | set(cache_data)
        for index, tag, cap in IDX_PLAN:
            comp = fetch_components(index)
            if not comp:
                print(f"[{tag}] 成分获取失败，跳过本指数", flush=True)
                continue
            rest = sorted(set(comp) - have)
            random.Random(index).shuffle(rest)
            pick = rest[:cap]
            have |= set(pick)
            todo += pick
            print(f"[{tag}] 成分 {len(comp)}，本批补 {len(pick)} 只", flush=True)
        print(f"[下载] 共 {len(todo)} 只（可跳过已缓存）", flush=True)
        if todo:
            download(todo, args.workers, cache_data)
            print(f"[缓存写入] large_scan_cache.pkl 累计 {len(cache_data)} 只", flush=True)
    else:
        print("[no-download] 跳过下载", flush=True)

    if not args.skip_db:
        persist_to_db(cache_data)

    large = {**base, **cache_data}
    print(f"\n[对比截面] 现池40={len(base)}  大池600={len(large)}\n", flush=True)

    t0 = time.time()
    print("正在算 40 池因子 IC ...", flush=True)
    df40 = scan(base, "base40")
    print("正在算 大池 600 因子 IC ...", flush=True)
    dfL = scan(large, "large")
    print(f"IC 计算完成，耗时 {time.time() - t0:.0f}s", flush=True)

    # ---------- 对比输出（h=5, 20） ----------
    summary = {"generated_at": datetime.now().isoformat(timespec="seconds"),
               "base_n": len(base), "large_n": len(large),
               "idx_plan": IDX_PLAN}
    for h in (5, 20):
        cols = ["factor", "mean_ic", "icir", "ic_positive", "abs_ic", "n_days", "judge"]
        a = df40[df40.horizon == h].set_index("factor")
        b = dfL[dfL.horizon == h].set_index("factor")
        merged = a[["mean_ic", "icir", "ic_positive", "abs_ic", "n_days", "judge"]].join(
            b[["mean_ic", "icir", "ic_positive", "abs_ic", "n_days", "judge"]],
            lsuffix="_40", rsuffix="_大", how="outer")
        merged["diff"] = merged["mean_ic_大"].fillna(0) - merged["mean_ic_40"].fillna(0)
        merged = merged.sort_values("abs_ic_大", ascending=False).head(12)
        print(f"\n{'=' * 78}\n预测期 {h} 日 · |IC|最大的 12 个因子（按大池排序）\n{'=' * 78}")
        print(f"{'因子':<14}{'IC_40':>8}{'IC_600':>9}{'差值':>8}{'ICIR_40':>8}{'ICIR_600':>9}  判定(600)")
        for name, r in merged.iterrows():
            print(f"{name:<14}{r['mean_ic_40']:>8.4f}{r['mean_ic_大']:>9.4f}"
                  f"{r['diff']:>8.4f}{r['icir_40']:>8.3f}{r['icir_大']:>9.3f}  {r['judge_大']}")
        hit40 = df40[(df40.horizon == h) & (df40["mean_ic"].abs() >= 0.03)]
        hitL = dfL[(dfL.horizon == h) & (dfL["mean_ic"].abs() >= 0.03)]
        print(f"[结论 h={h}] |IC|>=0.03 因子数：40池 {len(hit40)}/{len(df40[df40.horizon == h])}"
              f"  →  600池 {len(hitL)}/{len(dfL[dfL.horizon == h])}")
        summary[f"h{h}"] = {
            "hit40": int(len(hit40)), "hitL": int(len(hitL)),
            "top": [{"factor": str(n), "ic40": round(float(r["mean_ic_40"]), 4) if pd.notna(r["mean_ic_40"]) else None,
                     "ic600": round(float(r["mean_ic_大"]), 4) if pd.notna(r["mean_ic_大"]) else None,
                     "diff": round(float(r["diff"]), 4)} for n, r in merged.iterrows()],
        }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(f"\n[报告] 已存 results/large_pool_ic_compare.json")


if __name__ == "__main__":
    main()
