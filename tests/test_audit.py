"""账户恒等式护栏 单元测试（quant/trading/audit.py）。

运行：python tests/test_audit.py

## 为什么有这条测试

2026-09-17 一天两个账本 bug **都零异常、零告警**，全靠人眼发现。
恒等式护栏就是「让数字自己喊」——所以护栏本身必须是对的：
**既要能通过正常账本，也要能抓住被做坏的手脚。**
下面每条检查都配一对用例：clean 必须 OK、broken 必须报出来。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.audit import (audit_account, check_daily_points,   # noqa: E402
                                 check_day_pnl, reconcile_cash)
from quant.trading.paper import PaperBroker                           # noqa: E402

_UID = [0]


# 每次**运行**唯一的后缀。只用 pid 会被操作系统复用 -> 撞上 paper/ 下的残留同名库
# -> PaperBroker 打开到有历史状态的旧库 -> 测试间歇性失败（2026-09-22 实测）。
_RUN_TAG = os.urandom(4).hex()


def _fresh(tag="audit", capital=100_000.0):
    _UID[0] += 1
    tmp = f"paper/_test_{tag}_{os.getpid()}_{_RUN_TAG}_{_UID[0]}.db"
    for suf in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(tmp + suf)
        except OSError:
            pass
    return tmp, PaperBroker(tmp, initial_capital=capital,
                            commission=0.0003, slippage=0.0002, stamp_tax=0.0005)


def _rm(tmp):
    import gc
    import time
    for _ in range(5):
        gc.collect()
        left = False
        for suf in ("", "-journal", "-wal", "-shm"):
            try:
                os.remove(tmp + suf)
            except FileNotFoundError:
                pass
            except OSError:
                left = True
        if not left:
            return
        time.sleep(0.05)


# ============================================================
# 恒等式 1：现金
# ============================================================
def test_cash_reconciles_on_clean_account():
    tmp, b = _fresh("cashok")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        b.sell("000792", 100, 11.0, "2026-09-17")
        r = reconcile_cash(b)
        assert r["ok"] and abs(r["diff"]) < 1e-9, r
    finally:
        _rm(tmp)


def test_cash_catches_phantom_credit():
    """**核心回归**：手工往现金里加一笔钱（模拟「重复卖出凭空造钱」）-> 必须报出来。"""
    tmp, b = _fresh("cashbad")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        good = reconcile_cash(b)
        assert good["ok"]
        with b._connect() as conn:
            conn.execute("UPDATE paper_account SET value=value+9898.02 WHERE key='cash'")
            conn.commit()
        bad = reconcile_cash(b)
        assert not bad["ok"], "凭空多出来的现金没有被发现"
        assert abs(bad["diff"] - 9898.02) < 1e-6, bad
    finally:
        _rm(tmp)


# ============================================================
# 恒等式 2：日点净值
# ============================================================
def test_daily_points_clean():
    tmp, b = _fresh("ptok")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        b.snapshot_equity("2026-09-16", {"000792": 12.0}, price_date="2026-09-16")
        assert check_daily_points(b, lambda s, d: 12.0) == []
    finally:
        _rm(tmp)


def test_daily_points_catch_previous_day_pricing():
    """**核心回归（2026-09-17 事故）**：日点按**前一天**价格写 -> 必须报出来。"""
    tmp, b = _fresh("ptbad")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        # 事故里的写法：键是 09-16，价格却是 09-15 的（此处绕过 price_date 校验直写）
        b.snapshot_equity("2026-09-16", {"000792": 24.71})
        # 而 09-16 真实收盘是 24.75
        find = check_daily_points(b, lambda s, d: 24.75 if d == "2026-09-16" else 24.71)
        assert len(find) == 1, find
        assert find[0]["date"] == "2026-09-16"
        assert abs(find[0]["diff"] - (24.71 - 24.75) * 400) < 1e-6, find[0]
    finally:
        _rm(tmp)


def test_daily_points_skips_hourly_and_missing_close():
    """小时点（实时估值）与「查不到收盘价」都不该误报。"""
    tmp, b = _fresh("ptskip")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        b.snapshot_equity("2026-09-16 10:00", {"000792": 11.0})
        assert check_daily_points(b, lambda s, d: 12.0) == []          # 小时点跳过
        b.snapshot_equity("2026-09-16", {"000792": 12.0})
        assert check_daily_points(b, lambda s, d: None) == []          # 没价 -> 不误报
    finally:
        _rm(tmp)


# ============================================================
# 恒等式 3：当日收益
# ============================================================
def test_day_pnl_matches_the_2026_09_17_incident():
    """用事故当天的真实数字核一遍：601138 跌 + 000792 卖出 + 费用 == -92.90。"""
    tmp, b = _fresh("pnl")
    try:
        # 09-16 建仓（昨收口径的基准）
        b.buy("601138", 100, 62.33, "2026-09-16")
        b.buy("000792", 400, 24.75, "2026-09-16")
        # 09-17：000792 全部卖出（止损），601138 继续持有
        # 传入 24.75（= 昨收），滑点后记账价恰为 24.74505 —— 与真实成交一致
        b.sell("000792", 400, 24.75, "2026-09-17")
        closes = {("601138", "2026-09-16"): 62.33, ("000792", "2026-09-16"): 24.75}
        r = check_day_pnl(b, "2026-09-17", "2026-09-16",
                          lambda s, d: closes.get((s, d)),
                          price_now={"601138": 61.50}, day_pnl_reported=-92.90)
        assert r["ok"] is True, r
        assert abs(r["expected"] - (-92.90)) < 0.02, r
    finally:
        _rm(tmp)


def test_day_pnl_catches_wrong_baseline():
    """基线错了（比如原来那个 +76.10）-> 必须报出来。"""
    tmp, b = _fresh("pnlbad")
    try:
        b.buy("601138", 100, 62.33, "2026-09-16")
        b.buy("000792", 400, 24.75, "2026-09-16")
        # 传入 24.75（= 昨收），滑点后记账价恰为 24.74505 —— 与真实成交一致
        b.sell("000792", 400, 24.75, "2026-09-17")
        closes = {("601138", "2026-09-16"): 62.33, ("000792", "2026-09-16"): 24.75}
        r = check_day_pnl(b, "2026-09-17", "2026-09-16",
                          lambda s, d: closes.get((s, d)),
                          price_now={"601138": 61.50}, day_pnl_reported=76.10)
        assert r["ok"] is False, r
        assert abs(r["diff"] - (76.10 + 92.90)) < 0.02, r
    finally:
        _rm(tmp)


def test_audit_account_end_to_end():
    tmp, b = _fresh("e2e")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        b.snapshot_equity("2026-09-16", {"000792": 12.0}, price_date="2026-09-16")
        rep = audit_account(b, lambda s, d: 12.0)
        assert rep["ok"] and not rep["findings"], rep
    finally:
        _rm(tmp)


if __name__ == "__main__":
    tests = [test_cash_reconciles_on_clean_account,
             test_cash_catches_phantom_credit,
             test_daily_points_clean,
             test_daily_points_catch_previous_day_pricing,
             test_daily_points_skips_hourly_and_missing_close,
             test_day_pnl_matches_the_2026_09_17_incident,
             test_day_pnl_catches_wrong_baseline,
             test_audit_account_end_to_end]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
