"""第 31 步：为大池(559 只)回填换手率 / 流通股本 → data/large_pool.db 的 large_turnover 表。

背景（为什么）：
    现有因子与模型特征只用到 OHLCV+amount，**没有"换手率"维度**。
    换手率 = 成交量 / 流通股本，是 A 股经典因子（高换手 → 后续收益偏低）。
    日线数据里拿不到它，因为缺"流通股本"这个分母。

数据来源：
    新浪 `ak.stock_zh_a_daily`（就是 quant/data/fetcher.fetch_daily 的源 1），
    返回列里**本来就带** `outstanding_share`（流通股本）与 `turnover`（换手率），
    但 fetcher 的 `_SINA_COLS` 只保留了 OHLCV+amount，把这两列丢掉了。
    本脚本直接调该接口把这俩列捞回来存库。

关键性质（已逐股核验，见 docs/2026-09-11-factors-turnover.md）：
    - `outstanding_share` 是**逐日时点值**（会随增发/解禁/回购变化），
      不是"当前快照回填历史" → **无未来函数**；
    - `turnover` 恒等于 volume/outstanding_share（逐股 max|差|=0）；
    - 与东财口径交叉验证一致（600519 2024-01-02：新浪 0.256%，东财 0.26%）。

写库是**纯增量**：只新建 large_turnover 表，不动 large_daily / 任何既有表。

用法：
    python scripts/31_backfill_turnover.py              # 增量补齐全部大池
    python scripts/31_backfill_turnover.py --symbols 600519,000737
    python scripts/31_backfill_turnover.py --full       # 忽略已有数据，全量重拉
"""
from __future__ import annotations

import argparse
import logging
import socket
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from quant.config import load_config                                # noqa: E402
from quant.data.fetcher import _to_ak_symbol                        # noqa: E402

socket.setdefaulttimeout(30)


def _fetch_turnover(code: str, start: str, end: str) -> list[tuple] | None:
    """单只 → [(symbol, date, outstanding_share, turnover)]；失败返回 None。

    用不复权（adjust=""）拉：换手率/股本与复权无关，省掉复权计算。
    """
    import akshare as ak                                            # noqa: PLC0415
    import time                                                     # noqa: PLC0415
    df = None
    for attempt in range(3):                    # 新浪偶发 RemoteDisconnected，重试即可
        try:
            df = ak.stock_zh_a_daily(symbol=_to_ak_symbol(code),
                                     start_date=start, end_date=end, adjust="")
            if df is not None and not df.empty:
                break
        except Exception as exc:                                    # noqa: BLE001
            logger.debug("%s 拉取失败(第%d次): %s", code, attempt + 1, exc)
        time.sleep(1.5 * (attempt + 1))
    if df is None or df.empty or "turnover" not in df.columns:
        return None
    df = df[["date", "outstanding_share", "turnover"]].dropna()
    rows = []
    for d, sh, tv in zip(df["date"], df["outstanding_share"], df["turnover"]):
        try:
            rows.append((code, str(d)[:10], float(sh), float(tv)))
        except (TypeError, ValueError):
            continue
    return rows or None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=None, help="只补这些(逗号分隔)；默认全部大池")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--full", action="store_true", help="忽略已有数据，全量重拉")
    parser.add_argument("--sleep", type=float, default=0.2, help="每请求后间隔秒")
    args = parser.parse_args()

    cfg = load_config()
    db = Path(cfg.resolve("data")) / "large_pool.db"
    con = sqlite3.connect(str(db))
    con.execute("""CREATE TABLE IF NOT EXISTS large_turnover (
                       symbol TEXT NOT NULL,
                       date   TEXT NOT NULL,
                       outstanding_share REAL,
                       turnover REAL,
                       PRIMARY KEY (symbol, date))""")
    con.commit()

    if args.symbols:
        codes = [c.strip() for c in args.symbols.split(",") if c.strip()]
    else:
        codes = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM large_daily ORDER BY symbol")]
    if not codes:
        raise SystemExit("大池为空：请先跑 scripts/26_refresh_largepool.py")

    # 断点续传：已有数据的股票只从各自最后日期往后补
    have: dict[str, str] = {}
    if not args.full:
        have = {r[0]: r[1] for r in con.execute(
            "SELECT symbol, MAX(date) FROM large_turnover GROUP BY symbol")}
    start_all = str(cfg["data"]["start_date"])
    end = datetime.now().strftime("%Y-%m-%d")

    def _one(code: str):
        start = have.get(code)
        if start:                       # 已有 → 从最后一天重拉（覆盖最后一天，防止当日盘中半截数据）
            return code, _fetch_turnover(code, start, end)
        return code, _fetch_turnover(code, start_all, end)

    todo = codes
    logger.info("[31] 回填 %d 只大池的换手率（%s → %s，%d 只已有数据）",
                len(todo), start_all, end, len(have))
    done = ok = 0
    rows_buf: list[tuple] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one, c): c for c in todo}
        for fut in as_completed(futs):
            code, rows = fut.result()
            done += 1
            if rows:
                ok += 1
                rows_buf.extend(rows)
            if len(rows_buf) >= 20000:                 # 批量落库，避免长时间持锁
                con.executemany(
                    "INSERT OR REPLACE INTO large_turnover VALUES (?,?,?,?)", rows_buf)
                con.commit()
                rows_buf.clear()
            if done % 50 == 0:
                logger.info("  进度 %d/%d（成功 %d）", done, len(todo), ok)
    if rows_buf:
        con.executemany("INSERT OR REPLACE INTO large_turnover VALUES (?,?,?,?)", rows_buf)
        con.commit()

    n_rows = con.execute("SELECT COUNT(*) FROM large_turnover").fetchone()[0]
    n_sym = con.execute("SELECT COUNT(DISTINCT symbol) FROM large_turnover").fetchone()[0]
    dmin, dmax = con.execute("SELECT MIN(date), MAX(date) FROM large_turnover").fetchone()
    logger.info("[31] 完成：成功 %d/%d 只；表内 %d 行 / %d 只（%s ~ %s）",
                ok, len(todo), n_rows, n_sym, dmin, dmax)
    con.close()


if __name__ == "__main__":
    main()
