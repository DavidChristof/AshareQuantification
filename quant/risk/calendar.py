"""交易日/长假工具（纯逻辑，便于单测）。

用途：识别「明天起即将有一段较长休市（长假）」→ 供长假前降仓风控使用。

休市 = 周末 或 配置的法定休市日（holiday_dates，不含周末，A股节假日休市公告来源：
交易所每年年末公布次年安排，用户需在 config risk.pre_holiday.holiday_dates 维护）。
"""
from __future__ import annotations

from datetime import date, datetime, timedelta


def is_ashare_trading_day(d: date, holidays: set[date]) -> bool:
    """是否为 A股交易日：非周末且非法定休市日。"""
    return d.weekday() < 5 and d not in holidays


def daily_worker_step(now: datetime, run_time: str, grace_min: int,
                      marker_value: str | None,
                      is_trading: bool) -> tuple[str, datetime]:
    """「每个交易日跑一次」的后台 worker 此刻该做什么 —— 纯函数，便于测试。

    ## 为什么必须是纯函数

    `api/main.py` 的 worker **无法被测试导入**（import 该模块会加载模型、起后台线程），
    所以调度决策必须与线程分离。照抄 `quant/trading/paper.py::daily_point_date` 的做法：
    纯逻辑在 quant 里、有测试，副作用留在 api 层。

    ## 它修的是什么（2026-10-08）

    `api/main.py` 的两个 worker（开盘自动调仓、600 池维护）原本**处理完一天就 `return`**，
    线程永久结束 —— 于是只有「服务恰好在当天窗口前被重启过」的日子才会跑。
    2026-09-30 因此整天空过（用户只能手动调仓）；假期 10-02/05/06/07 四天没有日志，
    因为线程早在 10-01 就死了。本函数把它变成「每轮都重新问一次此刻该做什么」，
    配合 worker 侧的 `while True` 常驻循环，重启时刻就不再重要。

    ## [!] 条件顺序是**安全边界**，不可重排

    1. `now < trigger` → `wait`。**必须先判这条，再判 is_trading** —— 见下。
    2. `now > trigger + grace` → `skip_grace`：**本日不补跑**。这是「绝不迟到成交」的硬边界：
       服务 11:00 才起，就绝不能拿 11:00 的现价去补一笔本该在开盘做的单。
    3. 窗口内且 `not is_trading` → `skip_non_trading`（法定节假日）。
    4. `marker_value == 今天` → `skip_done`（防同日重复，含中途重启）。
    5. 否则 → `run`。

    ### 为什么第 1 条必须在 is_trading 之前

    调用方用 `_today_is_trading()` 求 `is_trading`，而它的判据是
    **实时快照的时间戳日期是否等于今天**（`api/main.py::_market_status`，否则 `market_closed`）。
    **交易日凌晨（如 03:00）快照还停在上一交易日 => `is_trading` 为 False。**
    若先判 is_trading，凌晨重启会把**正常交易日**误判成休市、整日跳过 —— 那比原来的 bug 更糟。
    （2026-10-08 那次复活恰好就是 03:00:56 重启，正踩在这个点上。）
    把 is_trading 推迟到触发点之后，快照就可靠了（09:31 => open；16:55 => closed_today；
    节假日 => market_closed），**于是不需要节假日清单也能认出休市** —— 而
    `risk.pre_holiday.holiday_dates` 目前是**空的**，`is_ashare_trading_day(10-01)` 会把
    国庆误判成交易日（这正是 10-01 空转的成因）。

    Args:
        now: 当前时间（墙钟）。
        run_time: 触发时刻 `'HH:MM'`。注意开盘自动调仓的真实触发点是
            `09:30:00 + delay_sec`，调用方需自行算好再传（`delay_sec=60` => `'09:31'`）。
        grace_min: 触发点过后仍允许执行的分钟数。
        marker_value: 运行标记文件的内容（已跑则等于今天的 ISO 日期）。
        is_trading: 今天是否确为交易日 —— 调用方请在**触发点之后**求值。

    Returns:
        `(action, wake_at)`。action ∈ `{run, wait, skip_grace, skip_done, skip_non_trading}`；
        wake_at = 下次最早该醒来的时刻（`wait` 为今日触发点，其余为明日触发点）。
        **调用方不要直接睡到 wake_at**：请 sleep 一个较短上限（如 60s）后重新调用本函数，
        这样机器睡眠/唤醒与时钟跳变后仍能用墙钟重新判定。
    """
    hh, mm = (int(x) for x in str(run_time).split(":"))
    today = now.date()
    trigger = datetime(today.year, today.month, today.day, hh, mm)
    nxt = trigger + timedelta(days=1)
    if now < trigger:                                   # [!] 必须最先判，理由见 docstring
        return "wait", trigger
    if (now - trigger).total_seconds() > grace_min * 60:
        # > 而非 >=：与旧实现一致 —— 恰好 trigger+grace 仍可执行
        return "skip_grace", nxt
    if not is_trading:
        return "skip_non_trading", nxt
    if marker_value and str(marker_value).strip() == today.isoformat():
        return "skip_done", nxt
    return "run", nxt


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
