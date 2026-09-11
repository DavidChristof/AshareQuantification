"""第 34 步：每日快照存档「个股资金流」—— 为将来能回测它攒历史。

背景（为什么这么做）：
    用户问「加资金流入流出量能否提高模型强度」。结论是**现在加不了**：
      - 东财 `stock_individual_fund_flow` 系列本机全部 RemoteDisconnected；
      - 同花顺只有**快照**（即时/3/5/10/20 日排行），**没有长历史**；
      - 概念上主力资金流需要**逐笔成交按单笔金额分类**，日线/5 分钟线还原不了
        （minute.db 只有 40 只 x 近 1 个月）。
    ⇒ 没有历史 = 无法回测 = 不上线（纪律）。**但可以从今天起每天存一次**，
      攒够 1~2 年后就有了可回测的历史 —— 与影子 A/B 同样「先攒数据再判断」的做法。

数据源：同花顺 `ak.stock_fund_flow_individual(symbol="即时")`（全市场 ~5200 只，~6 秒）。
    列：序号/股票代码/股票简称/最新价/涨跌幅/换手率/流入资金/流出资金/净额/成交额
    其中资金类是带单位的字符串（"29.13亿" / "5000万"），本脚本统一解析成元。

产物：`data/fund_flow.db` 表 `fund_flow_snapshot`
    (trade_date, symbol, name, price, pct_chg, turnover, inflow, outflow, net, amount)
    主键 (trade_date, symbol) → 同日重跑是幂等覆盖，不会重复堆积。

用法：
    python scripts/34_snapshot_fundflow.py                # 抓当日快照存档
    python scripts/34_snapshot_fundflow.py --date 2026-09-12   # 补存（数据是当日的，勿乱用）
    python scripts/34_snapshot_fundflow.py --show         # 只看库里已攒了多少，不抓
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))        # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from quant.config import load_config                                # noqa: E402

_NUM = re.compile(r"^-?[\d.]+")


def parse_amount(v) -> float | None:
    """'29.13亿' -> 2.913e9；'-1.64亿' -> -1.64e8；'5000万' -> 5e7；'--' -> None。"""
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if not s or s in {"-", "--", "None", "nan"}:
        return None
    m = _NUM.match(s)
    if not m:
        return None
    try:
        x = float(m.group())
    except ValueError:
        return None
    if "万亿" in s:
        return x * 1e12
    if "亿" in s:
        return x * 1e8
    if "万" in s:
        return x * 1e4
    return x


def parse_float(v) -> float | None:
    s = str(v).strip().replace(",", "").replace("%", "")
    if not s or s in {"-", "--", "None", "nan"}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch_snapshot() -> list[tuple]:
    """同花顺即时个股资金流 → [(symbol, name, price, pct_chg, turnover, 流入, 流出, 净额, 成交额)]。"""
    import akshare as ak                                            # noqa: PLC0415
    df = ak.stock_fund_flow_individual(symbol="即时")
    if df is None or df.empty:
        raise RuntimeError("同花顺资金流快照返回空")
    cols = list(df.columns)
    # 列名是中文，按位置取更稳（顺序：代码/简称/最新价/涨跌幅/换手率/流入/流出/净额/成交额）
    out = []
    for row in df.itertuples(index=False):
        vals = list(row)
        if len(vals) < 10:
            continue
        sym = str(vals[1]).zfill(6)
        if not sym.isdigit():
            continue
        out.append((sym, str(vals[2]), parse_float(vals[3]), parse_float(vals[4]),
                    parse_float(vals[5]), parse_amount(vals[6]), parse_amount(vals[7]),
                    parse_amount(vals[8]), parse_amount(vals[9])))
    logger.info("快照解析：%d 行 / 原始 %d 行（列 %s）", len(out), len(df), cols[:3])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="存档日期（默认今天）")
    ap.add_argument("--show", action="store_true", help="只显示库里已有的存档概况")
    args = ap.parse_args()

    cfg = load_config()
    db = Path(cfg.resolve("data")) / "fund_flow.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db))
    con.execute("""CREATE TABLE IF NOT EXISTS fund_flow_snapshot (
                       trade_date TEXT NOT NULL,
                       symbol     TEXT NOT NULL,
                       name       TEXT,
                       price      REAL,
                       pct_chg    REAL,
                       turnover   REAL,
                       inflow     REAL,
                       outflow    REAL,
                       net        REAL,
                       amount     REAL,
                       PRIMARY KEY (trade_date, symbol))""")
    con.commit()

    if args.show:
        rows = con.execute(
            "SELECT trade_date, COUNT(*) FROM fund_flow_snapshot "
            "GROUP BY trade_date ORDER BY trade_date").fetchall()
        print(f"存档日期数：{len(rows)}")
        for d, n in rows:
            print(f"  {d}  {n:>5d} 只")
        con.close()
        return

    day = args.date or datetime.now().strftime("%Y-%m-%d")
    data = fetch_snapshot()
    con.executemany(
        "INSERT OR REPLACE INTO fund_flow_snapshot VALUES (?,?,?,?,?,?,?,?,?,?)",
        [(day, *r) for r in data])
    con.commit()

    n = con.execute("SELECT COUNT(*) FROM fund_flow_snapshot WHERE trade_date=?", (day,)).fetchone()[0]
    days = con.execute("SELECT COUNT(DISTINCT trade_date) FROM fund_flow_snapshot").fetchone()[0]
    logger.info("[34] %s 存档 %d 只；累计 %d 个交易日", day, n, days)

    # 当日净流入/流出前五（做个可视化校验，确认字段没解析错）
    top_in = con.execute("SELECT symbol, name, net/1e8 FROM fund_flow_snapshot "
                         "WHERE trade_date=? AND net IS NOT NULL "
                         "ORDER BY net DESC LIMIT 5", (day,)).fetchall()
    print(f"\n{day} 主力净流入前五（亿元）：")
    for s, nm, v in top_in:
        print(f"  {s} {nm:<10} {v:+.2f}")
    con.close()


if __name__ == "__main__":
    main()
