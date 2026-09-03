"""回填分钟历史：用腾讯源把 5 分钟数据往前补（免费源能拿到的上限 ~7/8 起）。

现状：新浪抓取从系统上线(7/31)起累积，仅 ~25 交易日。腾讯 stock_zh_a_minute
一次返回 ~1970 根（约 41 个交易日，能回溯到 7/8），用它回填 7/8~7/31 缺口，
给分钟模型多覆盖一段不同行情（缓解单边下跌段过拟合）。

用法：python scripts/27_backfill_minute_tencent.py [--limit 0=全部]
"""
import argparse
import logging
import socket
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                  # noqa: E402

socket.setdefaulttimeout(25)
logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                 # noqa: E402
from quant.realtime.minute_store import MinuteStore  # noqa: E402

SCALE = 5


def fetch_tencent(symbol: str) -> pd.DataFrame:
    """腾讯 5 分钟（不复权，与新浪分钟口径一致），返回含 symbol/datetime/OHLCV。"""
    import akshare as ak
    from quant.realtime.quoter import _prefix
    df = ak.stock_zh_a_minute(symbol=_prefix(symbol), period="5", adjust="")
    if df is None or df.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "symbol": symbol,
        "datetime": pd.to_datetime(df["day"]),
        "open": df["open"].astype(float),
        "high": df["high"].astype(float),
        "low": df["low"].astype(float),
        "close": df["close"].astype(float),
        "volume": df["volume"].astype(float) if "volume" in df else 0.0,
        "amount": df["amount"].astype(float) if "amount" in df else 0.0,
    })
    return out.sort_values("datetime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="只回填前 N 只（0=全部）")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config()
    mc = cfg.get("minute", {})
    store = MinuteStore(cfg.resolve(mc.get("db_path", "data/minute.db")))

    # 库里所有 symbol（scale=5）
    with store._connect() as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM minute_bars WHERE scale=5").fetchall()
    symbols = [r[0] for r in rows]
    if args.limit:
        symbols = symbols[:args.limit]
    print(f"[回填] 目标 {len(symbols)} 只（腾讯5min，补 7/8~ 缺口）")

    done, ok, added = 0, 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_tencent, s): s for s in symbols}
        for fut in as_completed(futs):
            s = futs[fut]
            try:
                df = fut.result()
                if df is not None and not df.empty:
                    n = store.save_bars(df, SCALE)
                    added += n
                    ok += 1
            except Exception as e:  # noqa: BLE001
                logging.getLogger("backfill").warning("%s 失败: %s", s, str(e)[:80])
            done += 1
            if done % 10 == 0 or done == len(symbols):
                print(f"  进度 {done}/{len(symbols)} 成功 {ok}", flush=True)
    print(f"[完成] 成功 {ok}/{len(symbols)}，写入/更新 {added} 根")

    # 回填后范围
    with store._connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*), MIN(datetime), MAX(datetime) FROM minute_bars WHERE scale=5").fetchone()
    print(f"[库] 5min bars={row[0]} 最早={row[2] and row[1]} 最晚={row[2]}")


if __name__ == "__main__":
    main()
