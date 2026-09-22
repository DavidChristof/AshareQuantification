"""净值点的「日期」必须等于「价格所属交易日」—— 回归护栏。

运行：python tests/test_equity_point.py

## 为什么有这条测试（2026-09-17 事故）

用户问「工业富联不是跌了吗？当日收益是哪来的？」

`manual_account.db` 的**日点**净值一直是用**前一天**的收盘价写成的：
盘中 `manual_order` / 09:31 自动调仓调 `snapshot_equity(today, SIGNALS收盘价)`，
`today` 是**日历日 D**，而信号表最后一行是 **D-1**（当日行情收盘后才刷新）。
于是每天的日点都低一天，而 `prev_close_equity`（当日收益的基线）正是取这个点
=> **次日的「当日收益」凭空多算了前一天的涨跌**。

实测（精确到分）：09-16 日点 mv=15,964 = 400×24.71(09-15收盘) + 100×60.80(09-15收盘)，
而同日 15:00 点 mv=16,133 = 400×24.75 + 100×62.33（09-16 真实收盘）。
差额 169 元正好是 09-16 当天的涨幅，被算进了 09-17。

## 两处修复

1. `daily_point_date()`（纯函数）：判定日点该不该写、写哪天。
2. `snapshot_equity(..., price_date=)`：日期与价格所属日不一致时**直接拒绝写入**。
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker, daily_point_date     # noqa: E402

_UID = [0]


# 每次**运行**唯一的后缀。只用 pid 会被操作系统复用 -> 撞上 paper/ 下的残留同名库
# -> PaperBroker 打开到有历史状态的旧库 -> 测试间歇性失败（2026-09-22 实测）。
_RUN_TAG = os.urandom(4).hex()


def _tmp_db(tag="eq"):
    _UID[0] += 1
    return f"paper/_test_{tag}_{os.getpid()}_{_RUN_TAG}_{_UID[0]}.db"


def _fresh(tag="eq"):
    tmp = _tmp_db(tag)
    for suf in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(tmp + suf)
        except OSError:
            pass
    return tmp, PaperBroker(tmp, initial_capital=100_000.0,
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
# 1. 纯函数：日点该写在哪一天
# ============================================================
def test_daily_point_date_normal():
    """收盘后：行情已推进到今天，账户最后成交也在今天 -> 就写今天。"""
    assert daily_point_date("2026-09-16", "2026-09-16") == "2026-09-16"


def test_daily_point_date_premarket_idempotent():
    """盘前：行情与账户都停在昨天 -> 写昨天（值与昨天相同时是幂等重写）。"""
    assert daily_point_date("2026-09-15", "2026-09-15") == "2026-09-15"


def test_daily_point_date_refuses_when_account_is_ahead():
    """**核心回归**：账户已经成交到今天(D)、行情还是昨天(D-1) -> 不许写。

    这正是 2026-09-14 那类事故：把 D 的成交算到 D-1 的点上。
    """
    assert daily_point_date("2026-09-16", "2026-09-17") is None


def test_daily_point_date_restart_after_downtime():
    """服务停过一段时间后重启的两种情形。

    ① 停摆期间账户**没动过**（最后成交还是 latest 那天）-> 可以写：
       当前账户状态就是 latest 收盘时的状态，重写是幂等的、正确的。
    ② 停摆期间账户**动过**（最后成交晚于 latest，比如行情源坏了但服务在跑）
       -> 不许写（由「账户不能跑到 latest 之后」这条拦下）。
    """
    assert daily_point_date("2026-09-16", "2026-09-16") == "2026-09-16"
    assert daily_point_date("2026-09-16", "2026-09-20") is None


def test_daily_point_date_handles_empty():
    assert daily_point_date("", "2026-09-16") is None
    assert daily_point_date("2026-09-16", None) == "2026-09-16"     # 还没成交过 -> 可以写


def test_daily_point_date_accepts_timestamp_like():
    """传进来的可能是带时间的字符串（'2026-09-16 00:00:00'）-> 取日期部分比较。"""
    assert daily_point_date("2026-09-16 00:00:00", "2026-09-16 00:00:00") == "2026-09-16"


# ============================================================
# 2. broker 层：日期与价格不同天 -> 直接拒写
# ============================================================
def test_snapshot_equity_rejects_date_price_mismatch():
    """**核心回归**：键是 D、价格却是 D-1 的 -> ValueError，不许默默写进去。

    这正是事故里 manual_order / portfolio_apply 干的事。
    """
    tmp, b = _fresh("mismatch")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        try:
            b.snapshot_equity("2026-09-17", {"000792": 10.0}, price_date="2026-09-16")
            raise AssertionError("日期与价格不同天时应拒绝写入")
        except ValueError as exc:
            assert "不是同一天" in str(exc)
        # 拒绝写入 = 曲线上不该多出这个点
        assert not [r for r in b.equity_history() if str(r["date"])[:10] == "2026-09-17"]
    finally:
        _rm(tmp)


def test_snapshot_equity_accepts_matching_price_date():
    """同一天 -> 正常写入（修复不能把正常路径也堵了）。"""
    tmp, b = _fresh("match")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        eq = b.snapshot_equity("2026-09-16", {"000792": 12.0}, price_date="2026-09-16")
        row = [r for r in b.equity_history() if str(r["date"])[:10] == "2026-09-16"]
        assert row and abs(row[0]["market_value"] - 400 * 12.0) < 1e-6
        assert abs(eq - row[0]["equity"]) < 1e-6
    finally:
        _rm(tmp)


def test_snapshot_equity_without_price_date_still_works():
    """不传 price_date 时行为不变（engine.py 等老调用点零回归）。"""
    tmp, b = _fresh("legacy")
    try:
        b.buy("000792", 400, 10.0, "2026-09-16")
        eq = b.snapshot_equity("2026-09-16", {"000792": 11.0})
        assert eq > 0
    finally:
        _rm(tmp)


# ============================================================
# 3. 事故的算术复现：日点被按前一天价格写 -> 次日「当日收益」虚高
# ============================================================
def test_incident_arithmetic_is_reproduced():
    """把事故的数字摆一遍，防止将来有人「优化」掉这个校验。

    09-15 收盘 601138=60.80、09-16 收盘 62.33；持仓 100 股。
    日点若按 09-15 价写 -> 6,080；按 09-16 价写 -> 6,233。差 153 = 09-16 当天涨幅，
    次日就会以 6,080 为基线，把 153 算成「今天赚的」。
    """
    tmp, b = _fresh("arith")
    try:
        b.buy("601138", 100, 60.0, "2026-09-16")
        # 事故里的写法（价格取 09-15 收盘）现在会被拒 —— 换个日期试，证明是校验在挡
        try:
            b.snapshot_equity("2026-09-16", {"601138": 60.80}, price_date="2026-09-15")
            raise AssertionError("按前一天价格写日点应被拒绝")
        except ValueError:
            pass
        b.snapshot_equity("2026-09-16", {"601138": 62.33}, price_date="2026-09-16")
        row = [r for r in b.equity_history() if str(r["date"])[:10] == "2026-09-16"][0]
        assert abs(row["market_value"] - 100 * 62.33) < 1e-6, \
            "日点必须按 09-16 的收盘价写；写成 60.80 就会让次日当日收益虚高 153 元"
    finally:
        _rm(tmp)


if __name__ == "__main__":
    tests = [test_daily_point_date_normal,
             test_daily_point_date_premarket_idempotent,
             test_daily_point_date_refuses_when_account_is_ahead,
             test_daily_point_date_restart_after_downtime,
             test_daily_point_date_handles_empty,
             test_daily_point_date_accepts_timestamp_like,
             test_snapshot_equity_rejects_date_price_mismatch,
             test_snapshot_equity_accepts_matching_price_date,
             test_snapshot_equity_without_price_date_still_works,
             test_incident_arithmetic_is_reproduced]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
