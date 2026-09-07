"""第 26 步：每日把大池(559 只)最新日线增量补进 data/large_pool.db。

背景：40 只现池由 service 的 15:30 自动刷新维护（market.db）；影子 600 池需要另维护。
本脚本每天收盘后跑一次：对每只大池拉日线（新浪源，全量拉但只写回缺失日期），
保证 large_pool.db 与市场同步，供 scripts/27 影子 A/B 与选股使用。

用法（建议每天 ~15:40 跑，可装 Windows 计划任务）：
    python scripts/26_refresh_largepool.py             # 增量更新全部大池
    python scripts/26_refresh_largepool.py --symbols 605117,000021
"""
from __future__ import annotations

import argparse
import logging
import socket
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from pathlib import Path                                            # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                # noqa: E402
from quant.data.fetcher import fetch_daily                          # noqa: E402

socket.setdefaulttimeout(20)
cfg = load_config()
DB = Path(cfg.resolve("data")) / "large_pool.db"
parser = argparse.ArgumentParser()
parser.add_argument("--symbols", default=None, help="只更新这些(逗号分隔)；默认全部")
parser.add_argument("--workers", type=int, default=8)
args = parser.parse_args()


def main():
    con = sqlite3.connect(str(DB))
    if args.symbols:
        codes = [c.strip() for c in args.symbols.split(",") if c.strip()]
    else:
        codes = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM large_daily")]
    # 每只已有最大日期
    have = {c: con.execute("SELECT MAX(date) FROM large_daily WHERE symbol=?", (c,)).fetchone()[0]
            for c in codes}
    start = cfg["data"]["start_date"]

    def _one(code: str):
        try:
            df = fetch_daily(code, start)
            return code, df
        except Exception:  # noqa: BLE001
            return code, None

    todo = codes
    print(f"[26] 更新 {len(todo)} 只大池日线（增量到 {start}→今日）", flush=True)
    done, ok, added = 0, 0, 0
    rows_buf = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, c): c for c in todo}
        pending = set(futs)
        while pending:
            ds, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
            for fut in ds:
                code, df = fut.result()
                done += 1
                if df is None or df.empty:
                    continue
                ok += 1
                mx = have.get(code)
                for _, r in df.iterrows():
                    d = r["date"]
                    ds_ = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                    if mx and ds_ <= mx:
                        continue
                    rows_buf.append((code, ds_, float(r["open"]), float(r["high"]),
                                     float(r["low"]), float(r["close"]),
                                     float(r.get("volume", 0) or 0),
                                     float(r.get("amount", 0) or 0)))
                if len(rows_buf) >= 2000:
                    con.executemany("INSERT OR REPLACE INTO large_daily "
                                    "(symbol,date,open,high,low,close,volume,amount) "
                                    "VALUES (?,?,?,?,?,?,?,?)", rows_buf)
                    con.commit(); added += len(rows_buf); rows_buf = []
                if done % 100 == 0:
                    print(f"  {done}/{len(todo)}", flush=True)
    if rows_buf:
        con.executemany("INSERT OR REPLACE INTO large_daily "
                        "(symbol,date,open,high,low,close,volume,amount) "
                        "VALUES (?,?,?,?,?,?,?,?)", rows_buf)
        con.commit(); added += len(rows_buf)
    n = con.execute("SELECT COUNT(DISTINCT symbol) FROM large_daily").fetchone()[0]
    mx = con.execute("SELECT MAX(date) FROM large_daily").fetchone()[0]
    con.close()
    print(f"[26] 完成：成功 {ok}/{len(todo)}，新增 {added} 行 → large_pool.db {n} 只，最新 {mx}")


if __name__ == "__main__":
    main()
