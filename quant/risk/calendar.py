"""交易日/长假工具（纯逻辑，便于单测）。

用途：识别「明天起即将有一段较长休市（长假）」→ 供长假前降仓风控使用。

休市 = 周末 或 配置的法定休市日（holiday_dates，不含周末，A股节假日休市公告来源：
交易所每年年末公布次年安排，用户需在 config risk.pre_holiday.holiday_dates 维护）。
"""
from __future__ import annotations

from datetime import date, timedelta


def is_ashare_trading_day(d: date, holidays: set[date]) -> bool:
    """是否为 A股交易日：非周末且非法定休市日。"""
    return d.weekday() < 5 and d not in holidays


def upcoming_closure_run(today: date, holidays: set[date],
                         min_days: int = 3, max_scan: int = 20) -> dict | None:
    """统计「从明天起」连续休市（周末+法定假日）的天数。

    若连续休市 ≥ min_days，视为"即将进入长假"，返回信息；否则返回 None。
    用于在长假开始前的最后（些）个交易日触发降仓。max_scan 防止极端超长循环。
    """
    cnt = 0
    d = today + timedelta(days=1)
    for _ in range(max_scan):
        if is_ashare_trading_day(d, holidays):
            break
        cnt += 1
        d += timedelta(days=1)
    if cnt >= min_days:
        return {
            "days_off": cnt,
            "break_starts": str(today + timedelta(days=1)),
        }
    return None


def parse_dates(raw) -> set[date]:
    """把 config 里的 YYYY-MM-DD 字符串列表解析成 date 集合（忽略非法项）。"""
    out: set[date] = set()
    for s in raw or []:
        try:
            out.add(date.fromisoformat(str(s)))
        except (ValueError, TypeError):
            continue
    return out
