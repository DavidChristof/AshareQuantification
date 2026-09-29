"""第 46 步：回退池外持仓 —— 把 688578 那笔买入当**没发生过**，并把历史净值点重新对齐流水。

## 为什么要回退

688578（艾力斯，科创板）是这个手动模拟盘历史上**唯一一只 40 池外的持仓**：
2026-09-24 由「组合调仓」买入 100 股。40 池外的票**买得进、估不出** —— 成交走
`_build_prices`（含池外兜底），估值却长期只认 40 池信号表，于是当天净值静默漏算
一整笔（`docs/2026-09-24-manual-equity-missing-position.md`）。取价路径修好后，池外
持仓仍是持续噪声源（展示层名字回退成裸代码、止盈止损拿不到实时价），
故决定：**回退这笔买入**，且此后不再买池外（`config.trading.allow_out_of_pool: false`）。

## 三步

1. `delete_trade(id)` + `rebuild_from_trades()` —— 删掉那笔买入流水，按剩余流水**全量重放**
   现金与持仓：现金退回 11,303.86，688578 持仓消失，其余 5 只一分不动。
2. `drop_equity_points(...)` —— 删掉**不可重建**的盘中点。它们写的是实时价，而实时快照
   不留存（`data/minute.db` 里 688578 有 0 行）=> 拿收盘价去补等于**注入未来信息**，
   比删掉更糟（09-24 那次就是这么处理的）。
3. `rebuild_daily_points(...)` —— 日点**只依赖当天收盘时可知的信息**（该日为止的现金 +
   该日收盘价），所以能按同一口径**精确重算**，不引入未来信息。

## `2026-09-24 09:00` 那个点必须**保留**

它是**成交前**写的。证据（脚本会自己断言，不靠人记）：它的现金与「09-23 收盘的 PIT
现金」完全相等，而 09-23 的持仓里没有 688578 —— 它不可能是被这笔买入污染的点。
判据也不硬编码 key：规则是「09:00 早于 09:30 开盘，所以只有**买入当天**的 09:00 点
可能写在成交之前」，其余每天的 09:00 反映的是**含该持仓**的前收盘状态，照样要删。

## 用法

    .venv/Scripts/python.exe scripts/46_rollback_poolout_holding.py            # 预演
    .venv/Scripts/python.exe scripts/46_rollback_poolout_holding.py --apply    # 执行

预演**不是干瞪眼**：它把库复制到临时文件、在副本上真跑一遍完整流程并打印前后对照，
再丢掉副本 —— 所以你看到的就是 `--apply` 会得到的结果，而真库一动不动。

[!] `--apply` 之前**先停掉 8001 服务**：运行中的进程会按自己的节奏轮询
`/api/manual/account` 写净值点（09-24 的教训就是旧进程把修好的数据又覆盖了回去）。
本脚本不替你停服务，只依赖你停好了。

退出码：0 = 成功/预演通过；1 = 前置条件不满足或自检失败。
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from quant.config import load_config                                # noqa: E402
from quant.trading.paper import PaperBroker                         # noqa: E402

_OUT_SYMBOL = "688578"                       # 唯一一只池外持仓
_SINCE = "2026-09-24"                        # 这笔买入的日期 = 历史点受影响的第一天
_BACKUP = "paper/manual_account.db.bak-20260929-rollback-out-of-pool"


def _s42():
    """导入 scripts/42 —— 收盘价查找阶梯（40池→559池→全市场）只留一份，别让它漂移。"""
    p = Path(__file__).resolve().parent / "42_audit_accounts.py"
    spec = importlib.util.spec_from_file_location("s42", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _pit_cash(conn, upto: str, exclude_ids=frozenset()) -> float:
    """按流水重放得到 `upto`（含）为止的现金 —— **独立实现，只用于校验**。

    刻意不复用 `PaperBroker.rebuild_daily_points` 的重放：那是**被检查的对象**，
    拿它自己的逻辑去证明它对不对，等于自证。
    """
    r = conn.execute(
        "SELECT value FROM paper_account WHERE key='initial_capital'").fetchone()
    cash = float(r[0]) if r and r[0] is not None else 0.0
    for tid, d, _sym, side, _sh, _px, fee, amt in conn.execute(
            "SELECT id,date,symbol,side,shares,price,fee,amount "
            "FROM paper_trades ORDER BY id"):
        if tid in exclude_ids or str(d)[:10] > upto:
            continue
        cash += -float(amt) if side == "buy" else float(amt) - float(fee or 0)
    return cash


def _equity_rows(conn) -> dict[str, tuple]:
    return {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
        "SELECT date,cash,market_value,equity FROM paper_equity")}


def _print_points(title: str, rows: dict, keys: list[str]) -> None:
    print(f"\n  {title}")
    print(f"    {'日期':<18}{'cash':>14}{'mv':>13}{'equity':>14}")
    for k in keys:
        if k in rows:
            c, mv, eq = rows[k]
            print(f"    {k:<18}{c:>14.2f}{mv:>13.2f}{eq:>14.2f}")
        else:
            print(f"    {k:<18}{'（已删）':>14}")


def _run(cfg, db_path: Path) -> bool:
    close_of, conns = _s42().make_close_of(cfg.resolve("data"))
    broker = PaperBroker(
        db_path,
        initial_capital=cfg["manual"]["initial_capital"],
        commission=cfg["backtest"]["commission"],
        slippage=cfg["backtest"]["slippage"],
        stamp_tax=cfg["backtest"].get("stamp_tax", 0.0005),
        lot_size=int(cfg["manual"].get("lot_size", 100)),
    )
    conn = sqlite3.connect(str(db_path))

    # ---------- 1. 定位那笔买入，并算清要删/要重写的点 ----------
    buys = conn.execute(
        "SELECT id,date,symbol,side,shares,price,fee,amount,remark FROM paper_trades "
        "WHERE symbol=? ORDER BY id", (_OUT_SYMBOL,)).fetchall()
    if len(buys) != 1 or buys[0][3] != "buy":
        print(f"[中止] {_OUT_SYMBOL} 的流水不是「恰好一笔买入」，实际 {len(buys)} 条：{buys}")
        return False
    buy_id, buy_date = buys[0][0], str(buys[0][1])[:10]
    if buy_date != _SINCE:
        print(f"[中止] 买入日期是 {buy_date}，与脚本假定的 {_SINCE} 不符 —— 先核清再跑")
        return False
    print(f"  待删流水：id={buy_id}  {buy_date}  buy {_OUT_SYMBOL} "
          f"{buys[0][4]:.0f}股 @{buys[0][5]:.5f}  金额 {buys[0][7]:.4f}")

    # 受影响的盘中点：09:00 早于 09:30 开盘，所以**只有买入当天**的 09:00 可能写在成交前
    drop_keys, keep_pre_trade = [], None
    for (d,) in conn.execute(
            "SELECT date FROM paper_equity WHERE instr(date,' ')>0 ORDER BY date"):
        day, hm = str(d)[:10], str(d)[11:16]
        if day < _SINCE:
            continue
        if day == _SINCE and hm < "09:30":
            keep_pre_trade = d
            continue
        drop_keys.append(d)

    # 断言「成交前」那条判断成立 —— 对不上就中止，绝不猜
    if keep_pre_trade:
        want = _pit_cash(conn, "2026-09-23", exclude_ids={buy_id})
        got = float(conn.execute(
            "SELECT cash FROM paper_equity WHERE date=?", (keep_pre_trade,)).fetchone()[0])
        if abs(got - want) > 1e-6:
            print(f"[中止] {keep_pre_trade} 的现金 {got:.6f} 与 09-23 收盘的 PIT 现金 "
                  f"{want:.6f} 不等 —— 它不是成交前的点，保留它的前提不成立")
            return False
        print(f"  保留（成交前）：{keep_pre_trade}  cash={got:.6f} == 09-23 PIT 现金")

    daily_dates = [r[0] for r in conn.execute(
        "SELECT date FROM paper_equity WHERE instr(date,' ')=0 AND date>=? ORDER BY date",
        (_SINCE,))]
    before = _equity_rows(conn)
    print(f"  待删盘中点 {len(drop_keys)} 个；待重算日点 {len(daily_dates)} 个 {daily_dates}")

    # ---------- 2. 执行 ----------
    if not broker.delete_trade(buy_id):
        print(f"[中止] 删除流水 id={buy_id} 失败")
        return False
    rebuilt = broker.rebuild_from_trades()
    print(f"  按流水重建：{rebuilt}")

    dropped = broker.drop_equity_points(drop_keys)
    print(f"  删除盘中点：{dropped} 行")
    rep = broker.rebuild_daily_points(close_of, dates=daily_dates)
    print(f"  重算日点：written={rep['written']}  missing={rep['missing']}")

    # ---------- 3. 自检 ----------
    conn2 = sqlite3.connect(str(db_path))
    after = _equity_rows(conn2)
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"    [{'OK' if cond else '!!'}] {label}{('  ' + detail) if detail else ''}")
        ok = ok and bool(cond)

    print("\n  自检：")
    held = {p.symbol for p in broker.query_positions()}
    check(f"{_OUT_SYMBOL} 已不在持仓", _OUT_SYMBOL not in held, f"持仓={sorted(held)}")
    n_tr = conn2.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE symbol=?", (_OUT_SYMBOL,)).fetchone()[0]
    check(f"{_OUT_SYMBOL} 已无流水", n_tr == 0)
    cash_now, cash_want = broker.query_cash(), _pit_cash(conn2, "9999-12-31")
    check("现金与独立重放一致", abs(cash_now - cash_want) < 1e-6,
          f"库={cash_now:.6f} 重放={cash_want:.6f}")
    viol = conn2.execute(
        "SELECT COUNT(*) FROM paper_equity "
        "WHERE abs(equity-(cash+market_value))>1e-6").fetchone()[0]
    check("全表 equity == cash + market_value", viol == 0, f"违例 {viol} 行")
    check("行数 = 原行数 - 删除数",
          len(after) == len(before) - len(drop_keys),
          f"{len(before)} - {len(drop_keys)} -> {len(after)}")
    if keep_pre_trade:
        check(f"{keep_pre_trade} 仍在", keep_pre_trade in after)
    check("被删的盘中点没有被重建",
          all(k not in after for k in drop_keys))
    if rep["missing"]:
        check("没有日点因缺价被跳过", False, str(rep["missing"]))
    else:
        check("所有待重算日点都写成了", True)

    _print_points("日点（回退前）", before, daily_dates)
    _print_points("日点（回退后）", after, daily_dates)
    if keep_pre_trade:
        _print_points("保留的成交前盘中点", after, [keep_pre_trade])
    conn.close()
    conn2.close()
    for c, _t in conns:
        c.close()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="真正写入（默认只在副本上预演，真库一动不动）")
    args = ap.parse_args()

    cfg = load_config()
    db = cfg.resolve(cfg["manual"]["db_path"])
    if not db.exists():
        print(f"[中止] 库不存在：{db}")
        return 1
    backup = cfg.resolve(_BACKUP)

    if args.apply:
        if backup.exists():
            print(f"[中止] 备份已存在，不覆盖：{backup}")
            return 1
        print(f"库：{db}\n备份：{backup}")
        shutil.copy2(db, backup)
        print(f"[备份] 已复制 -> {backup}")
        print("[执行] 注意：服务必须先停 —— 否则运行中的进程会按自己的节奏重写净值点")
        target, tmpdir = db, None
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="rollback46_"))
        target = tmpdir / "manual_account.db"
        shutil.copy2(db, target)
        print(f"库：{db}\n[预演] 在副本上真跑一遍：{target}（真库不动）")

    try:
        ok = _run(cfg, target)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    if not ok:
        print("\n[失败] 自检未全过 —— 真库未被改动（预演）或请用备份回滚（--apply）")
        return 1
    print("\n[完成] 全部自检通过" + ("（预演；加 --apply 才真正写入）" if not args.apply else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
