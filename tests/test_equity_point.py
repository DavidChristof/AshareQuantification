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

## 第 2 个事故（2026-09-24）：**池外持仓被静默按 0 计入市值**

用户问「今天买了 688578 但未计入账户净值曲线」。

688578 是科创板票，**不在 40 池信号表里**；而日点的价格源当时是 `SIGNALS`
（**只有 40 池**）=> 取不到价 => `snapshot_equity` 的 `if _valid_price(price)`
直接跳过它、照样写点。于是 09-24 净值写成 86,288.84（应 96,710.84，少 10,422.00），
曲线凭空多出一根 -12.27% 的假暴跌（应 -1.68%）。

根因是**成交路径与估值路径不对称**：成交走 `_build_prices`（有第三级兜底
「选股候选 price，覆盖池外」）=> 买得进；估值这条没有 => 估不出。
孪生的 `_sync_real_equity` 在 2026-09-17 已改用 `_close_on_date`，
**手动盘漏打了这个补丁**。

## 三处修复（本轮）

1. `_sync_manual_equity` 的**日点**改用 `_close_on_date`（三个库按交易日取收盘价），
   缺一个就**不写**（原来只有 40 池、缺价静默当 0）。
2. 它的**盘中点**改用 `_build_prices`（覆盖池外），仍缺价就不写。
3. `snapshot_equity` 遇到无有效价的持仓**必须 warning 点名**（原来是完全静默的）。
"""
import logging
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


def _capture_warnings(fn):
    """跑 fn()，收集 `quant.trading.paper` 这个 logger 的告警文案（不依赖 pytest 的 caplog）。"""
    import quant.trading.paper as paper_mod
    got = []
    handler = logging.Handler()
    handler.emit = lambda rec: got.append(rec.getMessage())     # noqa: E731
    log = paper_mod.logger
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    try:
        fn()
    finally:
        log.removeHandler(handler)
        log.setLevel(old_level)
    return got


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


# ============================================================
# 4. 第 2 个事故（2026-09-24）：无价的持仓不得**静默**按 0 计
# ============================================================
def test_snapshot_equity_must_not_silently_zero_an_unpriced_position():
    """**核心回归**：取不到价的持仓必须被**点名告警**，不能悄无声息地当 0 计。

    那天手动盘买入 688578（科创板，不在 40 池信号表里），而日点的价格源当时只有 40 池
    => 它取不到价 => `if _valid_price(price)` 直接跳过、照样写点 => 净值少了 10,422.00。

    「算少」这件事由调用方负责避免（`api.main._sync_manual_equity` 现在是「缺价不写」）；
    broker 层这一条守的是**可见性**：漏了必须有日志，否则下一次还是查不出来。
    """
    tmp, b = _fresh("unpriced")
    try:
        b.buy("601138", 100, 60.0, "2026-09-24")
        b.buy("688578", 100, 113.0, "2026-09-24")
        recs = _capture_warnings(
            lambda: b.snapshot_equity("2026-09-24", {"601138": 61.0},
                                      price_date="2026-09-24"))
        row = [r for r in b.equity_history() if str(r["date"])[:10] == "2026-09-24"][0]
        assert abs(row["market_value"] - 100 * 61.0) < 1e-6
        text = " ".join(recs)
        assert "688578" in text, f"取不到价的持仓必须被点名告警；实际日志: {text!r}"
        assert "601138" not in text, "有价的持仓不该被误报"
    finally:
        _rm(tmp)


def test_snapshot_equity_does_not_warn_when_every_position_is_priced():
    """配上「不该吼的时候别吼」——否则告警会变成噪声、被忽略掉。"""
    tmp, b = _fresh("allpriced")
    try:
        b.buy("601138", 100, 60.0, "2026-09-24")
        b.buy("688578", 100, 113.0, "2026-09-24")
        recs = _capture_warnings(
            lambda: b.snapshot_equity("2026-09-24", {"601138": 61.0, "688578": 104.22},
                                      price_date="2026-09-24"))
        assert recs == [], f"价格齐全时不该有告警，实际: {recs!r}"
    finally:
        _rm(tmp)


def test_incident_0924_arithmetic_is_reproduced():
    """把 2026-09-24 那次的数字摆一遍，防止将来有人把「池外持仓」的取价又改窄。

    持仓与当日收盘：601138 100x61.00、603993 500x17.08、002558 400x23.62、
    688578 100x104.22、000792 400x23.70。
      * 价格齐全    => 市值 43,990.00（净值 96,710.84，当日 -1.68%）
      * 漏掉 688578 => 市值 33,568.00（净值 86,288.84，当日 -12.27%）
    差额 10,422.00 = 100 x 104.22，正好是那一笔的市值。
    """
    tmp, b = _fresh("arith0924")
    try:
        closes = {"601138": 61.00, "603993": 17.08, "002558": 23.62,
                  "688578": 104.22, "000792": 23.70}
        shares = {"601138": 100, "603993": 500, "002558": 400,
                  "688578": 100, "000792": 400}
        for s, n in shares.items():
            b.buy(s, n, 1.0, "2026-09-24")          # 成本价随便给，本测试只看市值

        def mv(prices):
            b.snapshot_equity("2026-09-24", prices, price_date="2026-09-24")
            return [r for r in b.equity_history()
                    if str(r["date"])[:10] == "2026-09-24"][0]["market_value"]

        full = mv(dict(closes))
        short = mv({k: v for k, v in closes.items() if k != "688578"})
        assert abs(full - 43990.00) < 1e-6, f"齐全时应是 43,990.00，实际 {full}"
        assert abs(short - 33568.00) < 1e-6, f"漏掉 688578 时应是 33,568.00，实际 {short}"
        assert abs(full - short - 10422.00) < 1e-6, \
            "差额必须正好是 688578 的市值 100 x 104.22"
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
             test_incident_arithmetic_is_reproduced,
             test_snapshot_equity_must_not_silently_zero_an_unpriced_position,
             test_snapshot_equity_does_not_warn_when_every_position_is_priced,
             test_incident_0924_arithmetic_is_reproduced]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
