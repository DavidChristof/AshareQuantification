# 2026-09-08 改动说明：开盘前（集合竞价）当日收益归零

> 按「每次大改动在 docs/ 下留说明」的约定。已合入 main。

## 问题

周二早盘未开盘（09:15–09:30 集合竞价），dashboard「当日收益」却显示 −¥18 之类非零值。
根因：实时源（sina）在竞价期返回**竞价撮合价**（time=今日、price 已非 0），
`_live_prices` 用它估值 → 持仓市值相对昨收基准出现小幅漂移 → 被当作“当日收益”。

与 09-04 修的“周六假当日收益”同源：那一次只处理了「非交易日归零」，
漏了「交易日但**还没开盘**」——竞价价不是真实成交，不该算当日盈亏。

## 改动

- `quant/trading/paper.py:live_summary`：新增 `session_started` 参数（默认 True 保持兼容）。
  当日收益改为在 **trading_today 且 session_started** 时才实时计；否则基准对齐当前 → 当日收益 0。
- `api/main.py:manual_account`：按当前时间 `>= 09:30` 传 `session_started`；
  连续交易开始后自动恢复真实当日收益，收盘后（晚间）仍显示当日全天收益，午休不受影响。

## 验证

- 09:25（集合竞价、未开盘）`GET /api/manual/account` → `day_pnl=0.0, day_return=0.0`（基准=当前权益）。
- 回归：`tests/test_engine_hold.py`、`test_calendar.py`、`test_bracket_limit.py`、`test_volatility.py` 全 PASS。
- 开盘(09:30)后自动恢复实时当日收益（逻辑未变，仅加时段门）。
