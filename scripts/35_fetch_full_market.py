"""第 35 步：下载**全市场沪深 A 股**日线（含流通股本/换手率）→ data/full_market.db。

背景（为什么做）：
    `docs/2026-09-11-survivorship-bias.md` 证实 559 池有严重幸存者偏差——它是用
    **今天的**中证500/1000 成分名单回填 2020 年以来的历史，池内流通市值加权 +107.7%，
    而真实中证500 只有 +12.9%（超额 **+97.9pp**），且偏差正好挂在流动性轴上
    （2021 年冷门股平均 +124.7% vs 已活跃 +26.8%）。
    ⇒ 要修，必须先有**不带未来信息**的候选全集，再按中证公开编制规则重建「时点宇宙」。

本脚本只负责**第 1 步：把全集数据拿到手**。重建规则见 `quant/data/universe_pit.py`，
校验见 `scripts/36_validate_pit_universe.py`。

范围：
    - 沪深 A 股（60/68/00/30 开头），**排除北交所**（920xxx 共 343 只——中证500/1000
      的样本空间只含沪深两市）。约 5219 只。
    - 起点 = config `data.start_date`（2020-01-01）。已实测：新浪接口虽然忽略日期区间
      （服务端返回全历史、akshare 端切片），但**按 2020 起点取到的数据与按 1990 起点的
      重叠段逐值完全一致**，不会丢行或错位。时点宇宙要"过去一年"回看窗口，回测从
      2021-07 起，故 2020 起点足够。

表结构（**单表**，一次请求拿齐）：
    full_daily(symbol,date,open,high,low,close,volume,amount,
               outstanding_share,turnover,float_mcap)
    - `close` 是**前复权**价（与项目其它库一致），**收益率必须用它**。
    - `outstanding_share`/`turnover` 是**逐日时点值**（非当前快照回填）⇒ 无未来函数。
    - `float_mcap` = `amount / turnover` = 成交均价 × 流通股本。
      [!]️ **不要用 `close × outstanding_share` 算市值**——close 是前复权价，
      实测这样算市值偏高 **3%~11.5%**（分红越多的票偏得越狠）。已实测复权**不改变**
      volume/amount/turnover/outstanding_share（4/4 完全相等），所以一次请求就够。

[!]️ 设计要点（别照抄 scripts/23）：
    - **流式落库，内存恒定**：每只拉完立刻进缓冲、达阈值批量写库并丢弃 DataFrame。
      scripts/23 把所有 DataFrame 攒在 dict 里 + pickle 落盘，是内存炸弹
      （559 只的 pickle 就 49MB，5219 只 ≈ 460MB，解包后数倍；项目曾因内存不足崩过）。
    - **状态与数据同事务**：`download_state` 的更新和该股的行**在同一次 commit** 里落盘，
      否则中途 Ctrl-C 会出现"状态说完成、数据却缺一段"，且永远补不回来。
    - 断点续传靠数据库本身（`download_state`），不依赖外部缓存文件。
    - 请用服务同款解释器运行：`.venv\\Scripts\\python.exe`（系统 Python 的 pandas 较旧，
      实测无法反序列化项目里的 `large_scan_cache.pkl`）。
    - **跑之前建议先停掉 8001 服务**：akshare 每次调用会新建一个 V8(MiniRacer) 上下文，
      内存占用高，收盘时段机器内存紧张时易崩。

用法：
    python scripts/35_fetch_full_market.py --dry-run        # 只看范围，不下载
    python scripts/35_fetch_full_market.py --limit 40       # 冒烟
    python scripts/35_fetch_full_market.py                  # 全量（可随时中断，重跑续传）
    python scripts/35_fetch_full_market.py --stats          # 看库内概况
    python scripts/35_fetch_full_market.py --workers 2      # 内存紧张时降并发
"""
from __future__ import annotations

import argparse
import logging
import socket
import sqlite3
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                # noqa: E402
from quant.data.fetcher import fetch_daily_full                     # noqa: E402

socket.setdefaulttimeout(25)

_FLUSH_ROWS = 20000         # 累积多少行提交一次（行 + 状态同事务）

_SCHEMA = """
CREATE TABLE IF NOT EXISTS full_daily (
    symbol TEXT NOT NULL, date TEXT NOT NULL,
    open REAL, high REAL, low REAL, close REAL, volume REAL, amount REAL,
    outstanding_share REAL, turnover REAL, float_mcap REAL,
    PRIMARY KEY (symbol, date));
CREATE INDEX IF NOT EXISTS idx_full_daily_date ON full_daily(date);
CREATE TABLE IF NOT EXISTS download_state (
    symbol TEXT PRIMARY KEY, status TEXT, rows INTEGER,
    first_date TEXT, last_date TEXT, updated_at TEXT, err TEXT);
"""


class _Gate:
    """全局令牌桶：保证任意两次请求的发起间隔 ≥ min_gap 秒（跨线程）。

    为什么需要：实测 workers=4 且无间隔时，约 800 次请求后新浪开始集体超时，
    下载会"看起来卡死"（4 个 worker 全部陷在 25s 超时 + 重试里）。
    README 也写了公开接口建议「间隔 ≥5 秒」。这里用一个温和的全局上限，
    并把并发降到 2~3，换取稳定的长跑。
    """

    def __init__(self, min_gap: float):
        self._gap = float(min_gap)
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.time()
            sleep_for = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._gap
        if sleep_for > 0:
            time.sleep(sleep_for)


def _hs_codes(cache: Path, refresh: bool = False) -> list[str]:
    """沪深 A 股代码（排除北交所 920xxx）。

    `stock_info_a_code_name()` 会去深交所/上交所站点翻页，实测偶发 ReadTimeout；
    代码表又是**日频不变**的，故缓存到本地文件，重跑不再依赖外网。
    """
    if cache.exists() and not refresh:
        codes = [c.strip() for c in cache.read_text(encoding="utf-8").split() if c.strip()]
        if codes:
            return codes
    import akshare as ak                                            # noqa: PLC0415
    codes: list[str] = []
    # 源 1：新浪全市场快照（稳定，且不依赖交易所站点）
    try:
        spot = ak.stock_zh_a_spot()
        col = next(c for c in spot.columns if c in ("代码", "symbol", "code"))
        codes = sorted({str(c)[-6:].zfill(6) for c in spot[col]})
    except Exception as exc:                                        # noqa: BLE001
        print(f"[35] 新浪代码表失败（{type(exc).__name__}），回退交易所站点", flush=True)
    # 源 2：交易所站点（szse/sse，实测偶发 ReadTimeout）
    if not codes:
        df = ak.stock_info_a_code_name()
        codes = sorted(str(c).zfill(6) for c in df["code"])
    codes = [c for c in codes if c.startswith(("60", "68", "00", "30"))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("\n".join(codes), encoding="utf-8")
    print(f"[35] 代码表已缓存 → {cache}（{len(codes)} 只；用 --refresh-codes 强制刷新）",
          flush=True)
    return codes


def _f(v):
    """转 float；NaN/None → None（别让 NaN 进 SQLite）。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if x != x else x


def _rows(df, code: str, after: str | None) -> list[tuple]:
    """DataFrame → 待写行；`after` 非空时只取该日之后（增量续传）。"""
    out = []
    for d, o, h, lo, c, v, a, sh, tv in zip(
            df["date"], df["open"], df["high"], df["low"], df["close"],
            df["volume"], df["amount"], df["outstanding_share"], df["turnover"]):
        ds = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
        if after and ds <= after:
            continue
        f = _f(a) / _f(tv) if _f(tv) else None      # 流通市值 = 成交额 / 换手率
        out.append((code, ds, _f(o), _f(h), _f(lo), _f(c), _f(v), _f(a), _f(sh), _f(tv), f))
    return out


def _flush(con, rows: list[tuple], states: list[tuple]) -> None:
    """行与状态**同一次 commit**（保证"状态说完成"等价于"数据已落盘"）。"""
    if rows:
        con.executemany("INSERT OR REPLACE INTO full_daily VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
        rows.clear()
    if states:
        con.executemany("INSERT OR REPLACE INTO download_state VALUES (?,?,?,?,?,?,?)", states)
        states.clear()
    con.commit()


def _stats(con: sqlite3.Connection) -> None:
    n_sym, n_rows = con.execute(
        "SELECT COUNT(DISTINCT symbol), COUNT(*) FROM full_daily").fetchone()
    dmin, dmax = con.execute("SELECT MIN(date), MAX(date) FROM full_daily").fetchone()
    print(f"[stats] full_daily : {n_sym} 只 / {n_rows:,} 行（{dmin} ~ {dmax}）")
    for st, n in con.execute(
            "SELECT status, COUNT(*) FROM download_state GROUP BY status"):
        print(f"[stats] state {st:<6}: {n}")
    bad = con.execute("SELECT COUNT(*) FROM full_daily WHERE float_mcap IS NULL").fetchone()[0]
    if bad:
        print(f"[stats] 注意: float_mcap 为空的行: {bad:,}（换手率为 0/缺失的停牌日，属正常）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--min-gap", type=float, default=0.4,
                    help="全局最小请求间隔（秒，跨线程）。实测无间隔时约 800 次请求后"
                         "新浪集体超时，下载会假死；0.4s×3 workers 是稳定档。")
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 只（冒烟用）")
    ap.add_argument("--stats", action="store_true", help="只打印库内概况")
    ap.add_argument("--dry-run", action="store_true", help="只列范围")
    ap.add_argument("--full", action="store_true", help="忽略状态，全量重拉（覆盖式）")
    ap.add_argument("--refresh", action="store_true",
                    help="日常增量：每股只追加其最后日期之后的行（不跳过任何票）")
    ap.add_argument("--refresh-codes", action="store_true", help="强制重取沪深代码表")
    args = ap.parse_args()

    cfg = load_config()
    db = Path(cfg.resolve("data")) / "full_market.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.executescript("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")
    con.executescript(_SCHEMA)
    con.commit()

    if args.stats:
        _stats(con)
        con.close()
        return

    start = str(cfg["data"]["start_date"])
    end = datetime.now().strftime("%Y-%m-%d")
    codes = _hs_codes(Path(cfg.resolve("data")) / "full_market_codes.txt",
                      refresh=args.refresh_codes)
    if args.limit:
        codes = codes[: args.limit]

    if args.dry_run:
        print(f"[dry-run] 沪深A股 {len(codes)} 只；区间 {start} ~ {end}；库 {db}")
        con.close()
        return

    # 断点续传：`state` = 已成功的 {symbol: 最后日期}
    state = {r[0]: r[1] for r in con.execute(
        "SELECT symbol, last_date FROM download_state WHERE status='ok'")}
    if args.full:                      # 全量重拉：都跑，且不按日期过滤（OR REPLACE 覆盖）
        todo, after_of = codes, {}
    elif args.refresh:                 # 日常增量：都跑，但每股只追加其最后日期之后的行
        todo, after_of = codes, state
    else:                              # 默认：只补没成功过的（失败/空的会自动重试）
        todo, after_of = [c for c in codes if c not in state], state

    print(f"[35] 沪深A股 {len(codes)} 只；已完成 {len(state)} 只；"
          f"本次待处理 {len(todo)} 只；区间 {start} ~ {end}；workers={args.workers}",
          flush=True)
    print("[35] 提示：akshare 每次调用新建 V8 上下文占内存，建议先停掉 8001 服务。", flush=True)

    gate = _Gate(args.min_gap)

    def _one(code: str):
        gate.wait()
        try:
            return code, fetch_daily_full(code, start, end), None
        except Exception as exc:                                    # noqa: BLE001
            return code, None, f"{type(exc).__name__}: {exc}"[:200]

    done = ok = fresh = 0
    rows_buf: list[tuple] = []
    state_buf: list[tuple] = []
    failures: list[str] = []
    empty: list[str] = []
    t0 = time.time()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, c): c for c in todo}
        pending = set(futs)
        while pending:
            done_set, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for fut in done_set:
                code, df, err = fut.result()
                done += 1
                if err is not None or df is None:
                    failures.append(code)
                    state_buf.append((code, "error", 0, None, None, now, err))
                elif df.empty:
                    empty.append(code)
                    state_buf.append((code, "empty", 0, None, None, now, None))
                else:
                    ok += 1
                    if code not in state:
                        fresh += 1
                    rows_buf.extend(_rows(df, code, after_of.get(code)))
                    d0 = str(df["date"].iloc[0])[:10]
                    d1 = str(df["date"].iloc[-1])[:10]
                    state_buf.append((code, "ok", len(df), d0, d1, now, None))
                    del df
                if len(rows_buf) >= _FLUSH_ROWS:
                    _flush(con, rows_buf, state_buf)
                if done % 100 == 0:
                    el = time.time() - t0
                    eta = el / done * (len(todo) - done) / 60
                    print(f"  {done}/{len(todo)}  成功 {ok}  新增 {fresh}  "
                          f"失败 {len(failures)}  空 {len(empty)}  "
                          f"用时 {el / 60:.1f}min  剩余约 {eta:.0f}min", flush=True)
    _flush(con, rows_buf, state_buf)

    logs = Path(cfg.resolve("logs"))
    logs.mkdir(parents=True, exist_ok=True)
    if failures:
        (logs / "full_market_fail.txt").write_text("\n".join(failures), encoding="utf-8")
        print(f"[35] {len(failures)} 只失败 → logs/full_market_fail.txt（重跑本脚本会自动重试）")
    if empty:
        (logs / "full_market_empty.txt").write_text("\n".join(empty), encoding="utf-8")
        print(f"[35] {len(empty)} 只返回空 → logs/full_market_empty.txt")

    print(f"[35] 完成：成功 {ok}/{len(todo)}，新增 {fresh} 只；"
          f"用时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)
    _stats(con)
    con.close()


if __name__ == "__main__":
    main()
