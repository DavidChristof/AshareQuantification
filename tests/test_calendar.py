"""交易日/长假识别 单元测试。

运行：python tests/test_calendar.py
"""
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.risk.calendar import (  # noqa: E402
    is_ashare_trading_day, parse_dates, upcoming_closure_run, daily_worker_step,
)


def test_is_trading_day_weekend_and_holiday():
    sat = date(2026, 9, 5)          # 周六
    assert not is_ashare_trading_day(sat, set())
    mon = date(2026, 9, 7)
    assert is_ashare_trading_day(mon, set())
    # 法定休市日（周一）不算交易日
    assert not is_ashare_trading_day(date(2026, 10, 1), parse_dates(["2026-10-01"]))
    print("PASS test_is_trading_day_weekend_and_holiday")


def test_closure_run_normal_weekend_not_trigger():
    """普通周末(2天)不足以触发长假降仓（min_days=3）。"""
    fri = date(2026, 9, 4)          # 周五，无额外假日 → 明起休 2 天
    assert upcoming_closure_run(fri, set(), min_days=3) is None
    print("PASS test_closure_run_normal_weekend_not_trigger")


def test_closure_run_long_break_trigger_on_last_day():
    """长假前最后一个交易日（周四）→ 明天起连续休市≥3 → 触发。"""
    holidays = parse_dates(["2026-10-01", "2026-10-02", "2026-10-05",
                            "2026-10-06", "2026-10-07", "2026-10-08"])  # 2026 国庆示例
    last_trading = date(2026, 9, 30)    # 周三 交易，明起 10/1 长假
    info = upcoming_closure_run(last_trading, holidays, min_days=3)
    assert info is not None and info["days_off"] >= 3
    # 更早一天（9/29 周二）——明天 9/30 仍交易 → 不触发
    assert upcoming_closure_run(date(2026, 9, 29), holidays, min_days=3) is None
    print("PASS test_closure_run_long_break_trigger_on_last_day")


# ---------- daily_worker_step：每日一次 worker 的调度决策 ----------
# 背景（2026-10-08）：api/main.py 的开盘自动调仓 / 600 池维护两个 worker 原本
# 「处理完一天就 return」，线程永久死亡 —— 见 quant/risk/calendar.py 的 docstring。

def test_step_before_trigger_waits():
    now = datetime(2026, 10, 8, 9, 0)
    assert daily_worker_step(now, "09:31", 25, None, True) == \
        ("wait", datetime(2026, 10, 8, 9, 31))
    print("PASS test_step_before_trigger_waits")


def test_step_at_trigger_runs():
    action, wake = daily_worker_step(datetime(2026, 10, 8, 9, 31), "09:31", 25, None, True)
    assert action == "run" and wake == datetime(2026, 10, 9, 9, 31)
    print("PASS test_step_at_trigger_runs")


def test_step_marker_today_skips():
    """今天已跑过（marker=今天）-> 不重复；也覆盖「窗口内中途重启」不重跑。"""
    assert daily_worker_step(datetime(2026, 10, 8, 9, 40), "09:31", 25,
                             "2026-10-08", True)[0] == "skip_done"
    print("PASS test_step_marker_today_skips")


def test_step_window_open_but_marker_stale_runs_0930_incident():
    """2026-09-30 事故形态：窗口开着、marker 仍是昨天（worker 已死）-> 应判 run 可补跑。

    线程是否真的活着由 worker 的常驻循环保证；本函数只证明「决策本身是对的」。
    """
    assert daily_worker_step(datetime(2026, 10, 8, 9, 31), "09:31", 25,
                             "2026-10-07", True)[0] == "run"
    print("PASS test_step_window_open_but_marker_stale_runs_0930_incident")


def test_step_holiday_skips_non_trading():
    """2026-10-01（周四）：holiday_dates 为空时 is_ashare_trading_day 会误判交易日，
    但触发点之后快照给出 is_trading=False -> 必须 skip_non_trading。

    这是假期空转（logs/shadow_ab_2026-10-01.log 那份孤零零的日志）的对症用例。
    """
    assert daily_worker_step(datetime(2026, 10, 1, 16, 55), "16:55", 120,
                             None, False)[0] == "skip_non_trading"
    print("PASS test_step_holiday_skips_non_trading")


def test_step_pre_open_stale_snapshot_must_wait_not_skip():
    """**核心回归**：交易日凌晨重启（快照还停在上个交易日 -> is_trading=False）。

    若「是否交易日」先于「到没到点」判定，这里会得出 skip_non_trading ——
    把一个正常交易日整日跳过，比原 bug 更糟。2026-10-08 03:00:56 的复活正是这个时点。
    """
    action, wake = daily_worker_step(datetime(2026, 10, 8, 3, 0), "09:31", 25, None, False)
    assert action == "wait" and wake == datetime(2026, 10, 8, 9, 31), \
        f"凌晨重启必须 wait 等触发点，实际 {action}"
    print("PASS test_step_pre_open_stale_snapshot_must_wait_not_skip")


def test_step_restart_at_night_never_runs():
    """夜里重启（23:58）不得触发成交 —— 那是「迟到成交」的硬边界。"""
    assert daily_worker_step(datetime(2026, 10, 8, 23, 58), "16:55", 120,
                             None, True)[0] == "skip_grace"
    assert daily_worker_step(datetime(2026, 10, 8, 23, 58), "09:31", 25,
                             None, True)[0] == "skip_grace"
    print("PASS test_step_restart_at_night_never_runs")


def test_step_suspended_across_window_never_trades_late():
    """机器 09:00 挂起、11:00 唤醒：窗口 [09:31,09:56] 已过 -> 绝不迟到成交。"""
    assert daily_worker_step(datetime(2026, 10, 8, 11, 0), "09:31", 25,
                             None, True)[0] == "skip_grace"
    print("PASS test_step_suspended_across_window_never_trades_late")


def test_step_resumed_inside_window_runs():
    """机器 09:00 挂起、09:40 唤醒：仍在窗口内 -> 应补跑。"""
    assert daily_worker_step(datetime(2026, 10, 8, 9, 40), "09:31", 25,
                             None, True)[0] == "run"
    print("PASS test_step_resumed_inside_window_runs")


def test_step_grace_boundary_is_inclusive():
    """与旧实现一致：恰好 trigger+grace 仍可执行，过一秒才 skip_grace。"""
    assert daily_worker_step(datetime(2026, 10, 8, 9, 56), "09:31", 25,
                             None, True)[0] == "run"
    assert daily_worker_step(datetime(2026, 10, 8, 9, 57), "09:31", 25,
                             None, True)[0] == "skip_grace"
    print("PASS test_step_grace_boundary_is_inclusive")


if __name__ == "__main__":
    tests = [test_is_trading_day_weekend_and_holiday,
             test_closure_run_normal_weekend_not_trigger,
             test_closure_run_long_break_trigger_on_last_day,
             test_step_before_trigger_waits,
             test_step_at_trigger_runs,
             test_step_marker_today_skips,
             test_step_window_open_but_marker_stale_runs_0930_incident,
             test_step_holiday_skips_non_trading,
             test_step_pre_open_stale_snapshot_must_wait_not_skip,
             test_step_restart_at_night_never_runs,
             test_step_suspended_across_window_never_trades_late,
             test_step_resumed_inside_window_runs,
             test_step_grace_boundary_is_inclusive]
    for fn in tests:
        fn()
    print("全部通过")
