"""探测：分钟数据能否扩大时间轴。

1) 分钟库实际存储范围（最早/最晚）
2) 数据源回溯能力实测（东财历史5分钟是否可用 / 腾讯 / 新浪）
据此判断能否把分钟训练数据从 ~25 天扩到更长。
"""
import socket
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                              # noqa: E402

socket.setdefaulttimeout(20)
from quant.config import load_config             # noqa: E402
from quant.realtime.minute_store import MinuteStore  # noqa: E402

cfg = load_config()
mc = cfg.get("minute", {})
db = cfg.resolve(mc.get("db_path", "data/minute.db"))
store = MinuteStore(db)

print("=== 1) 分钟库实际范围 (scale=5) ===")
with store._connect() as conn:
    row = conn.execute(
        "SELECT COUNT(*), COUNT(DISTINCT symbol), MIN(datetime), MAX(datetime) "
        "FROM minute_bars WHERE scale=5").fetchone()
print(f"  bars={row[0]} 股票数={row[1]} 最早={row[2]} 最晚={row[3]}")

print("\n=== 2) 数据源回溯能力实测 (标的 600519 贵州茅台) ===")

# a) 东方财富历史 5 分钟（可指定起止日期）
try:
    import akshare as ak
    df = ak.stock_zh_a_hist_min_em(
        symbol="600519", period="5",
        start_date="2026-05-06 09:30:00", end_date="2026-05-06 15:00:00", adjust="")
    print(f"  [东财5min] 5/6 拉取 {len(df)} 根 | 时间 {df.iloc[0]['时间']} ~ {df.iloc[-1]['时间']}")
    # 试试更早 + 60 分钟能拉多久
    df60 = ak.stock_zh_a_hist_min_em(
        symbol="600519", period="60",
        start_date="2024-01-01 09:30:00", end_date="2024-01-10 15:00:00", adjust="")
    print(f"  [东财60min] 2024/1 拉取 {len(df60)} 根 | {df60.iloc[0]['时间']} ~ {df60.iloc[-1]['时间']}")
except Exception as e:
    print(f"  [东财] 失败（可能仍被墙）: {type(e).__name__}: {str(e)[:120]}")

# b) 腾讯分钟（无日期参数，仅近期）
try:
    import akshare as ak
    df = ak.stock_zh_a_minute(symbol="sh600519", period="5", adjust="")
    print(f"  [腾讯5min] {len(df)} 根 | {df['day'].iloc[0]} ~ {df['day'].iloc[-1]}（近期）")
except Exception as e:
    print(f"  [腾讯] 失败: {type(e).__name__}: {str(e)[:120]}")

# c) 新浪分钟（CN_MarketDataService 走 akshare stock_zh_a_minute? 试新浪专用）
try:
    import akshare as ak
    df = ak.stock_zh_a_minute(symbol="sh600519", period="5", adjust="qfq")
    print(f"  [腾讯qfq5min] {len(df)} 根 | {df['day'].iloc[0]} ~ {df['day'].iloc[-1]}")
except Exception as e:
    print(f"  [腾讯qfq] 失败: {type(e).__name__}: {str(e)[:120]}")
