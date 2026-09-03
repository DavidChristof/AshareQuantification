"""第 22 步：扩展到中证500（中小盘）再扫一遍因子 IC —— 判断"全市场/中小盘才有效"假设。

复用 scripts/21 的下载缓存 results/large_scan_cache.pkl（缓存会累积），
每次用 --index 追加一个新指数成分、--limit 控制追加数量：
    python scripts/22_scan_mid_small_cap.py --index 000905 --limit 250   # 中证500 中盘
    python scripts/22_scan_mid_small_cap.py --index 000852 --limit 250   # 再补中证1000 小盘

扫描在「现池40 + 已下载全部」上进行，重点看：
    1) 20 日 60日反转 rev60（40池 0.008 → 沪深300池 0.035）在中小盘是否更强
    2) 有没有新因子达到 |IC|>=0.03
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                                    # noqa: E402

# 全局 socket 超时：避免 akshare 某个请求无限期挂起（曾因此卡一晚）
socket.setdefaulttimeout(20)

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

parser = argparse.ArgumentParser()
parser.add_argument("--index", default="000905", help="中证指数代码：000905=中证500, 000852=中证1000, 000300=沪深300")
parser.add_argument("--limit", type=int, default=250, help="本次追加下载数量")
parser.add_argument("--scan-only", action="store_true", help="只扫描现有缓存不下载")
args = parser.parse_args()

cfg = load_config()
start = cfg["data"]["start_date"]
end = datetime.now().strftime("%Y-%m-%d")


def fetch_components(index_code: str) -> list[str]:
    """获取指数成分；带 3 次重试 + 超时，失败返回空。"""
    import akshare as ak
    for attempt in range(3):
        try:
            df = ak.index_stock_cons_csindex(symbol=index_code)
            code_col = next(c for c in df.columns if "成分" in c and "代码" in c)
            codes = [str(v).zfill(6) for v in df[code_col] if str(v).isdigit()]
            return sorted(set(codes))
        except Exception as exc:  # noqa: BLE001
            print(f"[{index_code}] 成分获取失败(第{attempt + 1}/3 次): {exc}", flush=True)
            time.sleep(3)
    return []


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
    cache_data = {}
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            cache_data = pickle.load(f).get("data", {})

    # ---- 下载缺口（本次 index 成分中还没下载的） ----
    if not args.scan_only:
        comp = fetch_components(args.index)
        if not comp:
            print(f"[{args.index}] 成分获取彻底失败，中止（已有缓存仍可 --scan-only）")
            return
        have = set(base) | set(cache_data)
        todo = [c for c in comp if c not in have]
        random.Random(args.index).shuffle(todo)
        todo = todo[:args.limit]
        print(f"[{args.index}] 成分 {len(comp)}，本批取 {len(todo)} 只（含超时保护，失败自动跳过）", flush=True)

        def _one(code: str):
            try:
                return code, fetch_daily(code, start, end)
            except Exception:  # noqa: BLE001 - 超时/失败都返回 None
                return code, None

        done, ok = 0, 0
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(_one, c): c for c in todo}
            pending = set(futs)
            deadline = time.time() + len(todo) * 25 + 120   # 每只最多 ~25s + 缓冲
            while pending and time.time() < deadline:
                done_set, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                for fut in done_set:
                    code, df = fut.result()
                    done += 1
                    if df is not None and not df.empty:
                        ok += 1
                        cache_data[code] = df
                if done % 50 == 0 or (done and done == len(todo)):
                    print(f"  下载 {done}/{len(todo)}（成功 {ok}）", flush=True)
            if pending:
                print(f"[警告] {len(pending)} 只超时未完成，已跳过（进程不再挂起）", flush=True)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump({"data": cache_data}, f)
        print(f"[下载完成] 本次成功 {ok}/{len(todo)}，缓存累计 {len(cache_data)} 只", flush=True)

    large = {**base, **cache_data}
    print(f"\n[扫描截面] 现池 {len(base)} + 缓存 {len(cache_data)} = {len(large)} 只\n")

    dfL = scan(large, "large")
    for h in (5, 20):
        sub = dfL[dfL.horizon == h]
        top = sub.sort_values("abs_ic", ascending=False).head(12)
        print(f"\n{'='*60}\n预测期 {h} 日 · |IC| 前 12\n{'='*60}")
        print(top[["factor", "mean_ic", "icir", "ic_positive", "abs_ic", "n_days", "judge"]].to_string(index=False))
        hit = sub[sub["mean_ic"].abs() >= 0.03]
        print(f"[h={h}] |IC|>=0.03 因子数: {len(hit)}/{len(sub)}")
    # 重点因子明细
    print("\n[重点] rev60/mom60/ma_dev_60 各预测期:")
    key = dfL[dfL.factor.isin(["rev60", "mom60", "ma_dev_60", "rev20"])]
    print(key[["factor", "horizon", "mean_ic", "icir", "judge"]].to_string(index=False))


if __name__ == "__main__":
    main()
