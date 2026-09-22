"""纸面账户**并发安全**回归护栏（quant/trading/paper.py）。

运行：python tests/test_paper_atomic.py

## 为什么有这条测试（2026-09-17 事故）

`manual_account.db` 里 000792 出现**两条逐字节相同**的卖出（各 400 股），而账上只有
400 股。多卖的那 400 股**凭空贷记了 9,898.02 元**，总资产从 98,732 跳到 108,782。

根因：`sell()` 是「先开连接 SELECT 校验持仓 → 再开连接 BEGIN 更新现金」，
校验与扣款**不在同一个事务里**（典型 TOCTOU）。而 `_apply_manual_stops()` 挂在
`/api/manual/account` 与 `/api/manual/positions` 两个 GET 上被看板并发轮询，
两个线程都能读到 400 股、都判定「持仓足够」、各卖一次。

修法：`BEGIN IMMEDIATE` 后在同一写事务里完成「读持仓 → 校验 → 扣款」。
第二个调用会等在写锁上，拿到锁时读到的是更新后的持仓 → 正确返回「持仓不足」。

**这条测试必须能复现原 bug**：把 `BEGIN IMMEDIATE` 改回 `BEGIN`（或在 SELECT 之后），
`test_concurrent_sell_does_not_oversell` 就会失败。
"""
import atexit
import gc
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker            # noqa: E402

_UID = [0]
N_THREADS = 8
_MINE: list[str] = []          # 本进程建过的临时库，退出时兜底清理


# 每次**运行**唯一的后缀。只用 pid 会被操作系统复用 -> 撞上 paper/ 下的残留同名库
# -> PaperBroker 打开到有历史状态的旧库 -> 测试间歇性失败（2026-09-22 实测）。
_RUN_TAG = os.urandom(4).hex()


def _tmp_db(tag="atomic"):
    _UID[0] += 1
    p = f"paper/_test_{tag}_{os.getpid()}_{_RUN_TAG}_{_UID[0]}.db"
    _MINE.append(p)
    return p


def _rm(*paths):
    """删临时库（含 -journal/-wal/-shm）。尽力而为，失败就留给 atexit 兜底。

    [!] 为什么删不干净：`sqlite3.connect` 的连接在 `with conn:` 退出时**只提交、不关闭**，
    要等引用计数回收。回收晚一步，Windows 就还锁着文件、`os.remove` 抛 OSError。
    （项目里 600+ 个 `paper/_test_*.db` 残留就是这么来的 —— 旧测试把 OSError 静默吞了。）
    """
    for p in paths:
        for suf in ("", "-journal", "-wal", "-shm"):
            try:
                os.remove(p + suf)
            except OSError:
                pass


def _sweep_at_exit():
    """进程退出时兜底清理：此时所有连接都已析构，文件锁必然释放。"""
    for _ in range(3):
        gc.collect()
        left = False
        for p in _MINE:
            for suf in ("", "-journal", "-wal", "-shm"):
                if os.path.exists(p + suf):
                    try:
                        os.remove(p + suf)
                    except OSError:
                        left = True
        if not left:
            return
        time.sleep(0.05)


atexit.register(_sweep_at_exit)


def _fresh(tag="atomic"):
    tmp = _tmp_db(tag)
    _rm(tmp)
    return tmp, PaperBroker(tmp, initial_capital=100_000.0,
                            commission=0.0003, slippage=0.0002, stamp_tax=0.0005)


def test_concurrent_sell_does_not_oversell():
    """**核心回归**：8 个线程同时全平同一笔持仓 -> 只能成交 1 次，现金只贷记 1 次。"""
    tmp, b = _fresh()
    try:
        assert b.buy("000792", 400, 10.0, "2026-09-16").success     # 前一日买入（避开 T+1）
        cash_after_buy = b.query_cash()

        barrier = threading.Barrier(N_THREADS)
        results, errors = [], []

        def worker():
            try:
                barrier.wait(timeout=10)          # 尽量让 8 个线程同时进 sell()
                results.append(b.sell("000792", 400, 10.0, "2026-09-17"))
            except Exception as exc:              # noqa: BLE001
                errors.append(repr(exc))

        ts = [threading.Thread(target=worker) for _ in range(N_THREADS)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)

        assert not errors, f"并发卖出不应该抛异常: {errors}"
        ok = [r for r in results if r.success]
        assert len(ok) == 1, (
            f"同一笔 400 股持仓被成交了 {len(ok)} 次（应恰好 1 次）—— "
            f"sell() 的「校验持仓」与「扣款」又没有放进同一个写事务"
        )

        # 持仓必须清空，且现金只增加**一次**卖出的净额
        assert not b.query_positions(), "持仓应当已清空"
        delta = b.query_cash() - cash_after_buy
        one_net = 400 * (10.0 * (1 - 0.0002)) - ok[0].fee
        assert abs(delta - one_net) < 1e-6, (
            f"现金增加了 {delta:.4f}，但一次卖出只应增加 {one_net:.4f}"
            f"（多出来的就是凭空造的钱）"
        )
    finally:
        _rm(tmp)


def test_concurrent_sell_records_exactly_one_trade_row():
    """成交表里也必须只有一行 —— 原 bug 的表征就是「两条一模一样的卖出」。"""
    tmp, b = _fresh("rows")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        barrier = threading.Barrier(4)

        def worker():
            barrier.wait(timeout=10)
            b.sell("000792", 400, 10.0, "2026-09-17")

        ts = [threading.Thread(target=worker) for _ in range(4)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)

        with b._connect() as conn:
            n = conn.execute(
                "SELECT COUNT(*) FROM paper_trades WHERE side='sell'").fetchone()[0]
        assert n == 1, f"卖出流水有 {n} 行，应为 1 行"
    finally:
        _rm(tmp)


def test_concurrent_buy_does_not_overdraw():
    """**核心回归**：8 个线程同时买光同一笔现金 -> 只能成交到钱花完，不能透支。

    与重复卖出同一类 TOCTOU：原来是事务外 `query_cash()` 校验、事务内 `BEGIN` 扣款。
    """
    tmp, b = _fresh("buy")
    try:
        cash0 = b.query_cash()
        # 每次买 300 股 x 10 元 ≈ 3,001 元；8 个线程一起上，钱只够 ~33 次
        n = 8
        barrier = threading.Barrier(n)
        results, errors = [], []

        def worker():
            try:
                barrier.wait(timeout=10)
                results.append(b.buy("000792", 300, 10.0, "2026-09-16"))
            except Exception as exc:              # noqa: BLE001
                errors.append(repr(exc))

        ts = [threading.Thread(target=worker) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)

        assert not errors, f"并发买入不应该抛异常: {errors}"
        cash1 = b.query_cash()
        assert cash1 > -1e-6, f"现金被买成负数({cash1:.2f}) —— 校验与扣款不在同一事务"
        spent = cash0 - cash1
        ok = [r for r in results if r.success]
        assert abs(spent - sum(r.amount for r in ok)) < 1e-6, \
            "扣掉的现金必须恰好等于成交金额之和（不多扣、不重复扣）"
    finally:
        _rm(tmp)


def test_concurrent_buy_cannot_all_succeed():
    """钱只够一次 -> 并发下也只能成交一次（不能都判「资金足够」）。"""
    tmp = _tmp_db("buypoor")
    _rm(tmp)
    b = PaperBroker(tmp, initial_capital=4000.0,
                    commission=0.0003, slippage=0.0002, stamp_tax=0.0005)
    try:
        n = 6
        barrier = threading.Barrier(n)
        results, errors = [], []

        def worker():
            try:
                barrier.wait(timeout=10)
                results.append(b.buy("000792", 300, 10.0, "2026-09-16"))   # ~3001 元
            except Exception as exc:              # noqa: BLE001
                errors.append(repr(exc))

        ts = [threading.Thread(target=worker) for _ in range(n)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=20)

        assert not errors, f"不应该抛异常: {errors}"
        ok = [r for r in results if r.success]
        assert len(ok) == 1, (
            f"4000 元只够买一次(≈3001 元)，却成交了 {len(ok)} 次 —— "
            f"并发买入透支了现金")
        assert b.query_cash() > -1e-6
    finally:
        _rm(tmp)


def test_oversell_is_still_rejected_sequentially():
    """顺序调用下「持仓不足」照旧生效（修事务没改变业务语义）。"""
    tmp, b = _fresh("seq")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        assert b.sell("000792", 100, 10.0, "2026-09-17").success
        r = b.sell("000792", 400, 10.0, "2026-09-17")
        assert not r.success and "持仓不足" in r.message
        assert b.sell("000792", 300, 10.0, "2026-09-17").success
        assert not b.query_positions()
    finally:
        _rm(tmp)


def test_t1_rule_still_enforced():
    """T+1 校验搬进事务后仍然生效：当日买入当日不能卖。"""
    tmp, b = _fresh("t1")
    try:
        b.buy("000792", 400, 10.0, "2026-09-17")
        r = b.sell("000792", 400, 10.0, "2026-09-17")
        assert not r.success and "T+1" in r.message
        assert b.sell("000792", 400, 10.0, "2026-09-18").success
    finally:
        _rm(tmp)


if __name__ == "__main__":
    tests = [test_concurrent_sell_does_not_oversell,
             test_concurrent_sell_records_exactly_one_trade_row,
             test_concurrent_buy_does_not_overdraw,
             test_concurrent_buy_cannot_all_succeed,
             test_oversell_is_still_rejected_sequentially,
             test_t1_rule_still_enforced]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
