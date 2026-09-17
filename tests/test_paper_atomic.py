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
import os
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker            # noqa: E402

_UID = [0]
N_THREADS = 8


def _tmp_db(tag="atomic"):
    _UID[0] += 1
    return f"paper/_test_{tag}_{os.getpid()}_{_UID[0]}.db"


def _rm(*paths):
    for p in paths:
        for suf in ("", "-journal", "-wal", "-shm"):
            try:
                os.remove(p + suf)
            except OSError:
                pass


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
             test_oversell_is_still_rejected_sequentially,
             test_t1_rule_still_enforced]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
