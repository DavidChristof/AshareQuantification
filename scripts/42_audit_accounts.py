"""第 42 步：账户**恒等式核对** —— 账本类 bug 的护栏（每天跑，或收盘后跑）。

## 为什么需要

2026-09-17 一天出了两个账本 bug（重复卖出凭空多 9,898 元；日点净值按前一天价格写、
当日收益虚高一天），**两个都零异常、零告警** —— 全靠用户肉眼看「数字不对」才发现。
账本错了程序不会抛异常，**只能靠恒等式去对**。

本脚本核对三条恒等式（见 `quant/trading/audit.py`）：

    ① 现金      == 初始资金 - Σ买入金额 + Σ(卖出金额 - 卖出费用)
    ② 每个日点  equity == cash + Σ持仓 × 当日收盘价
    ③ 当日收益  == Σ持仓浮动 + Σ今日卖出(相对昨收) + Σ今日买入浮动 - Σ当日费用

用法：
    .venv/Scripts/python.exe scripts/42_audit_accounts.py            # 三个账户全查
    .venv/Scripts/python.exe scripts/42_audit_accounts.py --only manual
    .venv/Scripts/python.exe scripts/42_audit_accounts.py --quiet     # 只在有问题时输出

退出码：0 = 全部通过；1 = 有恒等式不成立（便于挂到流水线/定时任务上告警）。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from quant.config import load_config                                # noqa: E402
from quant.trading.audit import audit_account                       # noqa: E402
from quant.trading.paper import PaperBroker                         # noqa: E402
from quant.trading.real_account import RealBroker                   # noqa: E402

# 收盘价查找顺序：40 池 -> 559 池 -> 全市场
_DBS = (("market.db", "daily_bars"),
        ("large_pool.db", "large_daily"),
        ("full_market.db", "full_daily"))


def make_close_of(data_dir: Path):
    """返回 close_of(symbol, date) -> float | None（带缓存，避免逐点开库）。"""
    conns = []
    for db, table in _DBS:
        p = data_dir / db
        if p.exists():
            try:
                conns.append((sqlite3.connect(str(p)), table))
            except sqlite3.Error:
                pass
    cache: dict[tuple[str, str], float | None] = {}

    def close_of(symbol: str, date: str) -> float | None:
        key = (symbol, str(date)[:10])
        if key in cache:
            return cache[key]
        val = None
        for con, table in conns:
            try:
                r = con.execute(
                    f"SELECT close FROM {table} WHERE symbol=? AND date=?",
                    (symbol, key[1])).fetchone()
            except sqlite3.Error:
                continue
            if r and r[0] is not None:
                val = float(r[0])
                break
        cache[key] = val
        return val

    return close_of, conns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只查某个账户: paper|manual|real")
    ap.add_argument("--quiet", action="store_true", help="只在有问题时输出")
    args = ap.parse_args()

    cfg = load_config()
    data_dir = cfg.resolve("data")
    close_of, conns = make_close_of(data_dir)

    # [!] 每个账户的库路径：
    #   - 自动纸面盘：api/main.py 里是**硬编码**的 "paper/paper_account.db"，**没有**配置键
    #     （曾经这里误用了 cfg["data"]["db_path"] = data/market.db —— `PaperBroker.__init__`
    #      会 `CREATE TABLE IF NOT EXISTS`，于是把 4 张账户表建进了**行情库**里。已清理。）
    #   - 所以下面一律先确认文件存在再打开：**绝不新建**，缺了就跳过。
    def _open(cls, rel):
        p = cfg.resolve(rel)
        if not p.exists():
            print(f"  (跳过 {rel}：库不存在 —— 不新建)")
            return None
        return cls(p)

    accounts = [x for x in (
        ("paper", _open(PaperBroker, "paper/paper_account.db")),
        ("manual", _open(PaperBroker, cfg["manual"]["db_path"])),
        ("real", _open(RealBroker, cfg["real"]["db_path"])),
    ) if x[1] is not None]

    bad_total = 0
    for name, broker in accounts:
        if args.only and name != args.only:
            continue
        rep = audit_account(broker, close_of)
        n_find = len(rep["findings"])
        bad_total += n_find
        if not args.quiet or n_find:
            cash = next(c for c in rep["checks"] if c["name"] == "cash")
            pts = next(c for c in rep["checks"] if c["name"] == "daily_points")
            print(f"[{name}] cash={cash['actual']:.2f} "
                  f"(按流水应为 {cash['expected']:.2f}, 差 {cash['diff']:+.2f}) "
                  f"| 日点 {pts['n_bad']} 个不符 | 结论 {'OK' if rep['ok'] else '*** 有问题 ***'}")
            for f in rep["findings"]:
                if f["name"] == "cash":
                    print(f"    现金对不上: 差 {f['diff']:+.2f}（{f['n_trades']} 笔成交）")
                elif f["name"] == "daily_point":
                    print(f"    {f['date']} 日点: mv 实际 {f['market_value_actual']:.2f} "
                          f"应为 {f['market_value_expected']:.2f} (差 {f['diff']:+.2f})")
                else:
                    print(f"    {f.get('note') or f}")

    for con, _t in conns:
        con.close()

    print(f"\n[42] 结论：{'全部恒等式成立' if bad_total == 0 else f'{bad_total} 处不成立'}")
    if bad_total == 0:
        print("[42] 注：这只是「账面自洽」，不代表策略赚钱；有持仓时请连当日收益一起看。")
    return 0 if bad_total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
