"""第 38 步：下载**按报告期的季度财务指标** → data/fundamentals.db（线上选股可回测的关键）。

## 为什么需要它

`docs/2026-09-11-pit-universe.md` 末尾指出的最大未解问题：
**线上 `select_daily` 有 60% 权重是 PE/ROE，而它俩的历史时点值此前拿不到**
⇒ 我们一直只能用「技术面 40% 的近似」回测，再拿结果推断线上策略，这一步不成立。

已验证可行，按报告期的季度序列可拿到（含每股收益 / 净资产收益率 / 每股净资产）。
**数据源踩坑记录（重要，省得再走一遍）**：

| 源 | 结果 |
|---|---|
| 新浪 `stock_financial_analysis_indicator` | 可用，但**每只要打 ~8 个分页请求** → 3 workers/0.4s ≈ 20 req/s 时 **600 只只成功 65 只**，随后该 host 对所有请求返回**空表**（软封） |
| 新浪 `stock_financial_abstract` | 1 请求/只、返回已是数值，但限速到 2.5 req/s 仍在 ~100 只后被封（`JSONDecodeError`） |
| **同花顺** `stock_financial_abstract_ths` | **主力源**：另一个 host、返回全历史；1.5s/只（≈0.67 req/s）实测 100/100 零失败 |

⇒ 现为**同花顺优先、新浪回退**。新浪系的金融数据接口对本机限速非常敏感，别用高并发打。

⚠️ **两个必须处理的点**：
1. **数值是「年内累计」的**（600519 2020：Q1 11.04 → H1 19.05 → Q3 28.54 → FY 39.42）。
   直接当季度值用会错。TTM 需要先**去累计**再滚动加总（见 `quant/data/fundamentals.py`）。
2. **这是报告期，不是公告日** ⇒ 直接用会**偷看未来**（Q2 的数据 6/30 就有了，但 8 月底才公告）。
   时点构造必须加**披露滞后**（在 `fundamentals.py` 里统一处理，不在这层）。

范围：默认只下 **PIT 宇宙成员**（`full_market.db` 的 `pit_members` 并集，约 2445 只），
这样够回测用，请求量也只有全市场的一半。

产物：`data/fundamentals.db`
    financials(symbol, report_date, eps_diluted, eps_weighted, roe, roe_weighted, bps)
    fetch_state(symbol, status, rows, first_date, last_date, updated_at)

用法：
    python scripts/38_fetch_fundamentals.py --limit 20      # 冒烟
    python scripts/38_fetch_fundamentals.py                 # 全量（可中断，重跑续传）
    python scripts/38_fetch_fundamentals.py --all-market    # 不限 PIT 成员，下全市场
    python scripts/38_fetch_fundamentals.py --stats
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import socket
import pandas as pd
import sqlite3
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                # noqa: E402

socket.setdefaulttimeout(30)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS financials (
    symbol TEXT NOT NULL, report_date TEXT NOT NULL,
    eps_diluted REAL, eps_weighted REAL, roe REAL, roe_weighted REAL, bps REAL,
    PRIMARY KEY (symbol, report_date));
CREATE TABLE IF NOT EXISTS fetch_state (
    symbol TEXT PRIMARY KEY, status TEXT, rows INTEGER,
    first_date TEXT, last_date TEXT, updated_at TEXT, err TEXT);
"""

# 指标名关键词 → 目标字段（按顺序取第一个命中的行）
_WANT = {
    "eps_diluted": ("基本每股收益", "稀释每股收益"),
    "eps_weighted": ("稀释每股收益", "基本每股收益"),
    "roe": ("净资产收益率(ROE)", "净资产收益率"),
    "roe_weighted": ("净资产收益率(ROE)", "净资产收益率"),
    "bps": ("每股净资产",),
}
_DATE_COL = re.compile(r"^\d{8}$")


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _pick(df, keys):
    """按关键词顺序找列（取第一个命中的）。"""
    for k in keys:
        for c in df.columns:
            if k in str(c):
                return c
    return None


def _target_symbols(cfg, all_market: bool) -> list[str]:
    db = Path(cfg.resolve("data")) / "full_market.db"
    if not db.exists():
        raise SystemExit(f"缺 {db}：请先跑 scripts/35_fetch_full_market.py")
    con = sqlite3.connect(str(db))
    if all_market:
        syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM full_daily")]
    else:
        try:
            syms = sorted({r[0] for r in con.execute("SELECT DISTINCT symbol FROM pit_members")})
        except sqlite3.OperationalError:
            syms = []
        if not syms:
            print("[38] pit_members 为空 → 回退全市场（先跑 scripts/36 --build 可缩小范围）")
            syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM full_daily")]
    con.close()
    return sorted(syms)


class _Gate:
    """全局令牌桶：跨线程保证请求间隔 ≥ min_gap（与 scripts/35 同一套，防新浪限速假死）。"""

    def __init__(self, min_gap: float):
        import threading
        self._gap, self._lock, self._next = float(min_gap), threading.Lock(), 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._gap
        if sleep_for > 0:
            time.sleep(sleep_for)


def _num_ths(v):
    """同花顺的值为字符串（'29.86%' / '2.97' / False）→ float。"""
    if v is None or v is False:
        return None
    s = str(v).strip().replace("%", "").replace(",", "")
    if s in ("", "-", "--", "False", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fetch_ths(code: str, start_year: str) -> list[tuple]:
    """同花顺「财务摘要-按报告期」→ 与新浪同结构的长表。

    ⚠️ 为什么要它：新浪的 `stock_financial_abstract` / `stock_financial_analysis_indicator`
    在约 100~800 次请求后会对本机**软封**（返回空表 / JSONDecodeError），
    即使限速到 2.5 req/s 也一样。同花顺是**另一个 host**，是主力源；新浪留作回退。
    """
    import akshare as ak                                            # noqa: PLC0415
    df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
    if df is None or df.empty or "报告期" not in df.columns:
        return []
    eps_col = _pick(df, ("基本每股收益", "稀释每股收益"))
    bps_col = _pick(df, ("每股净资产",))
    roe_col = _pick(df, ("净资产收益率-摊薄", "净资产收益率(ROE)", "净资产收益率"))
    yr_floor = f"{start_year}-01-01"
    out = []
    for _, r in df.iterrows():
        ds = str(r["报告期"])[:10]
        if not ds or ds < yr_floor or ds.startswith("nan"):
            continue
        eps = _num_ths(r[eps_col]) if eps_col else None
        roe = _num_ths(r[roe_col]) if roe_col else None
        out.append((code, ds, eps, eps, roe, roe, _num_ths(r[bps_col]) if bps_col else None))
    return out


def _fetch_one(code: str, start_year: str) -> list[tuple]:
    """优先同花顺，失败回退新浪。返回 [(symbol, report_date, eps_diluted, ...)]。"""
    try:
        rows = _fetch_ths(code, start_year)
        if rows:
            return rows
    except Exception:                                               # noqa: BLE001
        pass
    return _fetch_sina(code, start_year)


def _fetch_sina(code: str, start_year: str) -> list[tuple]:
    """新浪「财务摘要」→ [(symbol, report_date, eps_diluted, eps_weighted, roe, roe_weighted, bps)]。

    ⚠️ 换源说明：原先用 `stock_financial_analysis_indicator`，但它**每只要打 ~8 个分页请求**，
    实测 600 只里只成功 65 只（随后该 host 直接把所有人打成空表）。
    改用 `stock_financial_abstract`：**1 个请求/只**、返回已是数值、报告期作列，
    含 `基本每股收益 / 净资产收益率(ROE) / 每股净资产`，口径一致（年内累计）。

    返回是**宽表**（行=指标、列=报告期 "YYYYMMDD"），这里转成按报告期的长表。
    """
    import akshare as ak                                            # noqa: PLC0415
    df = ak.stock_financial_abstract(symbol=code)
    if df is None or df.empty or "指标" not in df.columns:
        return []
    dcols = [c for c in df.columns if _DATE_COL.match(str(c))]
    if not dcols:
        return []
    yr_floor = f"{start_year}0101"

    picked: dict[str, pd.Series] = {}
    for field, keys in _WANT.items():
        for k in keys:
            hit = df[df["指标"].astype(str).str.contains(k, regex=False, na=False)]
            if len(hit):
                picked[field] = hit.iloc[0]
                break
    if not picked:
        return []

    out = []
    for c in dcols:
        ds = str(c)                                     # 20260630
        if ds < yr_floor:
            continue
        rds = f"{ds[:4]}-{ds[4:6]}-{ds[6:]}"
        out.append((code, rds, *[_num(picked[f].get(c)) if f in picked else None
                                 for f in _WANT]))
    return out


def _flush(con, rows: list[tuple], states: list[tuple]) -> None:
    if rows:
        con.executemany(
            "INSERT OR REPLACE INTO financials VALUES (?,?,?,?,?,?,?)", rows)
        rows.clear()
    if states:
        con.executemany("INSERT OR REPLACE INTO fetch_state VALUES (?,?,?,?,?,?,?)", states)
        states.clear()
    con.commit()


def _stats(con: sqlite3.Connection) -> None:
    n_sym, n_rows = con.execute(
        "SELECT COUNT(DISTINCT symbol), COUNT(*) FROM financials").fetchone()
    dmin, dmax = con.execute(
        "SELECT MIN(report_date), MAX(report_date) FROM financials").fetchone()
    print(f"[stats] financials : {n_sym} 只 / {n_rows:,} 期（报告期 {dmin} ~ {dmax}）")
    for st, n in con.execute("SELECT status, COUNT(*) FROM fetch_state GROUP BY status"):
        print(f"[stats] state {st:<6}: {n}")


def main():
    ap = argparse.ArgumentParser()
    # 限速：`stock_financial_abstract` 是**1 请求/只**，所以请求速率 ≈ workers/min_gap。
    # 实测安全档约 2.7 req/s（scripts/35 用 3 个 worker 跑完 4478 次请求无失败）；
    # 2/0.8 ≈ 2.5 req/s。**踩过的坑**：换源前用 `stock_financial_analysis_indicator`
    # （每只内部 8 个分页请求）配 3/0.4 ≈ 20 req/s → 600 只里只成功 65 只，该 host 随后
    # 对所有请求返回空表（软封）。
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--min-gap", type=float, default=1.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--start-year", default="2019",
                    help="起始年份（2021-07 回测需要 2020 起的 TTM，留 1 年余量）")
    ap.add_argument("--all-market", action="store_true", help="不限 PIT 成员")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--full", action="store_true", help="忽略状态重拉")
    args = ap.parse_args()

    cfg = load_config()
    db = Path(cfg.resolve("data")) / "fundamentals.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
    con.executescript(_SCHEMA)
    con.commit()

    if args.stats:
        _stats(con)
        con.close()
        return

    codes = _target_symbols(cfg, args.all_market)
    if args.limit:
        codes = codes[: args.limit]
    state = {} if args.full else {
        r[0] for r in con.execute("SELECT symbol FROM fetch_state WHERE status='ok'")}
    todo = [c for c in codes if c not in state]

    print(f"[38] 目标 {len(codes)} 只；已完成 {len(state)} 只；本次 {len(todo)} 只；"
          f"起始年份 {args.start_year}；workers={args.workers}", flush=True)

    gate = _Gate(args.min_gap)

    def _one(code: str):
        gate.wait()
        try:
            return code, _fetch_one(code, args.start_year), None
        except Exception as exc:                                    # noqa: BLE001
            return code, None, f"{type(exc).__name__}: {exc}"[:200]

    done = ok = 0
    rows_buf: list[tuple] = []
    state_buf: list[tuple] = []
    failures: list[str] = []
    t0 = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, c): c for c in todo}
        pending = set(futs)
        while pending:
            done_set, pending = wait(pending, timeout=60, return_when=FIRST_COMPLETED)
            for fut in done_set:
                code, rows, err = fut.result()
                done += 1
                if err is not None or not rows:
                    failures.append(code)
                    state_buf.append((code, "empty", 0, None, None, now, err))
                else:
                    ok += 1
                    rows_buf.extend(rows)

                    d0, d1 = min(r[1] for r in rows), max(r[1] for r in rows)
                    state_buf.append((code, "ok", len(rows), d0, d1, now, None))
                # 行与状态各自触发落盘：失败的股票没有行，若只按 rows_buf 触发，
                # 大批失败时状态会一直不落库（曾经因此看不到失败详情）。
                if len(rows_buf) >= 20000 or len(state_buf) >= 200:
                    _flush(con, rows_buf, state_buf)
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{len(todo)}  成功 {ok}  空/失败 {len(failures)}  "
                          f"用时 {el / 60:.1f}min  剩余约 "
                          f"{el / done * (len(todo) - done) / 60:.0f}min", flush=True)
    _flush(con, rows_buf, state_buf)

    if failures:
        p = Path(cfg.resolve("logs")) / "fundamentals_fail.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(failures, ensure_ascii=False), encoding="utf-8")
        print(f"[38] {len(failures)} 只无数据/失败 → {p}（重跑会自动重试）")

    print(f"[38] 完成：成功 {ok}/{len(todo)}；用时 {(time.time() - t0) / 60:.1f} 分钟",
          flush=True)
    _stats(con)
    con.close()


if __name__ == "__main__":
    main()
