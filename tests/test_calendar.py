"""交易日/长假识别 单元测试。

运行：python tests/test_calendar.py
"""
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.risk.calendar import (  # noqa: E402
    is_ashare_trading_day, parse_dates, upcoming_closure_run,
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


if __name__ == "__main__":
    tests = [test_is_trading_day_weekend_and_holiday,
             test_closure_run_normal_weekend_not_trigger,
             test_closure_run_long_break_trigger_on_last_day]
    for fn in tests:
        fn()
    print("全部通过")
