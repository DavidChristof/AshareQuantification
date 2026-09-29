"""第 47 步：按流水 + 当日收盘**重算所有日点**（抹平日点审计差异）。

## 什么时候用

`scripts/42` 的恒等式 ② 会揪出「日点 equity ≠ cash + Σ持仓×当日收盘」的历史点。
最常见的原因是**那些日期的 40 池收盘价事后被修订过** —— 点写下去的时候是对的，
后来行情源改了数，点就与现在的收盘价对不上了。这类差异不会自己消失，
只会一直挂在审计里（2026-09-29 实测：09-03/04/21/22/23 共 5 处，全部早于 09-24）。

日点**只依赖当天收盘时可知的信息**（该日为止的现金 + 该日收盘价），所以重算
**不引入未来信息**（盘中点就不同了 —— 它们写的实时价不可重建，只能删，见
`drop_equity_points`）。

## 用法

    .venv/Scripts/python.exe scripts/47_rebuild_daily_points.py            # 预演
    .venv/Scripts/python.exe scripts/47_rebuild_daily_points.py --apply     # 执行

预演在**库的副本**上真跑一遍并打印前后对照 + 审计结果，真库一动不动。
**这是一个历史改写操作** —— 只在你确认「审计里的差异是收盘价修订造成的」之后才跑。

退出码：0 = 成功/预演通过；1 = 自检失败。
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
from quant.trading.audit import check_daily_points                  # noqa: E402
from quant.trading.paper import PaperBroker                         # noqa: E402

_BACKUP = "paper/manual_account.db.bak-20260929-daily-point-rebuild"


def _s42():
    """导入 scripts/42 —— 收盘价查找阶梯只留一份（写法同 scripts/45 的 _s18()）。"""
    p = Path(__file__).resolve().parent / "42_audit_accounts.py"
    spec = importlib.util.spec_from_file_location("s42", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _points(db: Path) -> dict:
    conn = sqlite3.connect(str(db))
    try:
        return {r[0]: (r[1], r[2], r[3]) for r in conn.execute(
            "SELECT date,cash,market_value,equity FROM paper_equity "
            "WHERE instr(date,' ')=0")}
    finally:
        conn.close()


def _intraday(db: Path) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM paper_equity WHERE instr(date,' ')>0").fetchone()[0]
    finally:
        conn.close()


def _run(cfg, db: Path) -> bool:
    close_of, conns = _s42().make_close_of(cfg.resolve("data"))
    broker = PaperBroker(
        db,
        initial_capital=cfg["manual"]["initial_capital"],
        commission=cfg["backtest"]["commission"],
        slippage=cfg["backtest"]["slippage"],
        stamp_tax=cfg["backtest"].get("stamp_tax", 0.0005),
        lot_size=int(cfg["manual"].get("lot_size", 100)),
    )
    before_pts = _points(db)
    n_intraday_before = _intraday(db)
    before = check_daily_points(broker, close_of)
    print(f"  审计（改前）：{len(before)} 处差异")

    rep = broker.rebuild_daily_points(close_of)            # dates=None -> 所有日点
    print(f"  重算：written={len(rep['written'])}  missing={rep['missing']}")

    after_pts = _points(db)
    after = check_daily_points(broker, close_of)

    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"    [{'OK' if cond else '!!'}] {label}{('  ' + detail) if detail else ''}")
        ok = ok and bool(cond)

    print("\n  自检：")
    check("改后审计 0 处差异", len(after) == 0,
          "" if not after else str([f["date"] for f in after]))
    check("日点数量没变", len(after_pts) == len(before_pts),
          f"{len(before_pts)} -> {len(after_pts)}")
    conn = sqlite3.connect(str(db))
    viol = conn.execute(
        "SELECT COUNT(*) FROM paper_equity "
        "WHERE abs(equity-(cash+market_value))>1e-6").fetchone()[0]
    check("全表 equity == cash + market_value", viol == 0, f"违例 {viol} 行")
    conn.close()
    check("盘中点一个没动", _intraday(db) == n_intraday_before,
          f"{n_intraday_before} -> {_intraday(db)}（本工具只写日点）")
    if rep["missing"]:
        check("没有日点因缺价被跳过", False, str(rep["missing"]))

    changed = [d for d in sorted(before_pts)
               if abs(before_pts[d][2] - after_pts.get(d, before_pts[d])[2]) > 1e-6]
    if changed:
        print(f"\n  改动的日点（{len(changed)} 个）：")
        print(f"    {'日期':<14}{'equity 改前':>16}{'equity 改后':>16}{'差':>12}")
        for d in changed:
            b_, a_ = before_pts[d][2], after_pts[d][2]
            print(f"    {d:<14}{b_:>16.2f}{a_:>16.2f}{a_ - b_:>12.2f}")
    else:
        print("\n  没有任何日点需要改动（已经与收盘价自洽）")

    for c, _t in conns:
        c.close()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="真正写入（默认只在副本上预演）")
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
        target, tmpdir = db, None
    else:
        tmpdir = Path(tempfile.mkdtemp(prefix="rebuild47_"))
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
