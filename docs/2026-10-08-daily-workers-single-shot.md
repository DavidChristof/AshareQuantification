# 「每日一次」后台 worker 是**一天一次性**的 —— 自动调仓静默失能

**日期**：2026-10-08
**起因**：用户问「之前不是让你做了自动调仓吗，怎么今天早上还是需要我手动操作？」
**结论**：不是自动调仓没写对，是**它跑完一天就把自己杀了**。

---

## 1. 事故

`api/main.py` 末尾起了 4 个 daemon 线程，其中两个是「每个交易日跑一次」的：

| worker | 职责 | 触发点 | grace |
|---|---|---|---|
| `_auto_open_execute_worker` | 开盘自动组合调仓（真下单） | 09:30 + `delay_sec`(60s) = **09:31** | 25 min |
| `_shadow_ab_worker` | 600 池维护流水线（26→27→28→35） | `run_time` = **16:55** | 120 min |

**它们都在处理完一天后无条件 `return`** —— 线程永久结束，直到进程重启。而且都在线程启动
那一刻就把「今天」固定下来（`today = datetime.now().date()`）。

于是它们**只在「服务恰好在当天窗口之前被重启过」的日子才会跑**：

```
2026-09-29  worker 09:31 执行完 -> return，线程死
2026-09-30  早上 09:31 无人接点 -> 用户手动调仓（本可避免）
            （当晚我重启过服务，但新 worker 取到的「今天」已是 09-29、早已过 grace
              -> 启动那一瞬间就 return。所以那次失败**与重启无关**，是设计如此。）
2026-10-01  18:46 放弃并 return（假期拿到 0 行日线）
2026-10-02/05/06/07  四天**没有任何 shadow_ab 日志** <- 线程早在 10-01 就死了
2026-10-08  服务 03:00:56 被重启 -> 两个 worker 复活 -> 09:31 自动调仓真的自动跑了
            （logs/auto_open_execute_date = 2026-10-08），600 也在 16:55 跑完
```

**10-08 之所以正常，纯粹是因为那天凌晨重启过。** 这不是修复，是运气。

对照组：`_scheduler`（自动刷新）与 `_manual_stop_worker`（止盈止损巡检）本来就是
`while True` 常驻 —— **炸的只有这两个**。

## 2. 放大器：事故**无痕**

`api/main.py` **从未配置过 logging**。uvicorn 的 `LOGGING_CONFIG` 里**没有 root entry**
（只配了 `uvicorn` / `uvicorn.error` / `uvicorn.access` 三个 logger）=> 应用 logger 冒泡到
一个**没有 handler 的 root**，只剩 Python 的 `lastResort` 兜底：**仅 WARNING 及以上、
且无时间戳无级别**。

实测确认：

```
root.handlers = []
log.info(...)    -> 丢弃（什么都不写）
log.warning(...) -> 落到 stderr（裸格式）
```

所以 worker 那句「错过开盘触发点」是 `logger.info` —— **一个字都没落盘**。
`logs/api.log` / `api.err.log` 里什么都没有，只能靠人去数成交发现。这是本次真正
昂贵的部分：**故障不吭声**。

## 3. 修复

### 3.1 调度决策抽成纯函数 `quant/risk/calendar.py::daily_worker_step`

理由：`tests/` 里**没有任何文件 import `api/main.py`**（导它就会加载模型、起线程），
决策若不与线程分离就**无法测试**。做法照抄 `quant/trading/paper.py::daily_point_date`
（纯函数 + 事故编号 docstring + 单测）。

```python
daily_worker_step(now, run_time, grace_min, marker_value, is_trading) -> (action, wake_at)
action ∈ {run, wait, skip_grace, skip_done, skip_non_trading}
```

### 3.2 [!] 条件顺序是安全边界，不可重排

1. `now < trigger` → `wait`         ← **必须最先判**
2. `now > trigger + grace` → `skip_grace`（**本日不补跑**）
3. 窗口内且 `not is_trading` → `skip_non_trading`
4. `marker == 今天` → `skip_done`
5. 否则 → `run`

**为什么第 1 条必须在 `is_trading` 之前**：`is_trading` 来自 `_today_is_trading()`，而它的
判据是**实时快照的时间戳日期是否等于今天**（`_market_status`，否则 `market_closed`）。
**交易日凌晨（如 03:00）快照还停在上一个交易日 => `is_trading` 为 False。**
若先判 `is_trading`，凌晨重启会把**正常交易日误判成休市、整日跳过** —— 比原 bug 更糟。
（10-08 那次复活恰好就是 03:00:56 重启，正踩在这个点上。有专门回归测试钉住。）

把 `is_trading` 推迟到触发点之后还**顺带解决了假期识别**：那时快照可靠
（09:31 => `open`；16:55 => `closed_today`；节假日 => `market_closed`），
**不需要节假日清单也能认出休市** —— 而 `risk.pre_holiday.holiday_dates` 目前是**空的**，
`is_ashare_trading_day(2026-10-01)` 会把国庆误判成交易日（这正是 10-01 空转的成因）。

**安全边界（`run` 的充分条件）**：只能在 `[trigger, trigger+grace]` 且今天确为交易日时。
服务 11:00 才起 → **绝不拿 11:00 的现价补一笔开盘单**；机器睡眠穿过窗口 → 醒来跳过当天。
窗口外一律 `skip_grace`。边界与旧实现一致（`>` 而非 `>=`，恰好 `trigger+grace` 仍可跑）。

### 3.3 两个 worker 改成常驻循环

- 每轮用 `datetime.now()` 重新推导「今天」→ **重启时刻不再重要**。
- `try/except` 移进**循环内**：坏配置或单次故障只废掉一轮，不能杀死线程（旧版把 try 包在
  整个函数外面，一处异常就永久退出 —— 与「错过就死」同源）。
- `_shadow_ab_worker` 的内层重试（日线发布探测 / 内存闸门 / `retry_until`）逐字搬进
  `_shadow_ab_run_once()`，让那些「本日放弃」的 `return` 只结束这一轮，而不是杀掉 worker。
- **睡眠上限 60 秒**，不睡到 `wake_at`：Windows 的 `time.sleep` 不保证计入系统挂起时间，
  一觉睡到触发点在「挂起穿过窗口」时行为不可预测。60 秒上限保证唤醒后必定用墙钟重判 ——
  「挂起到 11:00」稳定 `skip_grace`（正确），「挂起到 09:40」稳定 `run`（正确）。

### 3.4 [!] 同日至多一次（开盘自动调仓专属）

`portfolio_apply` 先卖后买，**中途抛非 HTTPException 会留下部分成交且不写 marker**。
旧的单次 worker 抛完就死，无所谓；改成循环后 60 秒后仍在窗口内 → **会再跑一遍 → 重复下单**。

所以 worker 内存一个 `attempted` 日期，`run` 且今天已尝试过则降级 `skip_done`。
这恰好保留了原代码 `[!]` 注释的既定语义：「执行异常不写 marker，**grace 窗口内重启可补**」。

**600 池维护刻意没有这道闸门** —— 它是幂等的数据重算（26 增量 INSERT OR REPLACE、
27/28 重算重写），窗口内因偶发异常重跑一遍**无害且有益**（多一次拿到当日日线的机会）。
差异是刻意的，不是漏写。

### 3.5 把「今天跑没跑」喊出来

- `_daily_run_status()`：读标记 + 交易日 + 窗口终点 → `{enabled, expect, ran, date, missed, window_end}`。
- 挂到 `GET /api/portfolio` 的 `daily_runs`（与 `drawdown_brake` 同一个 dict）。
  这是仓库里**唯一一条**「状态文件 → HTTP 字段 → 看板横幅」的既有链路，照抄它。
- 看板「模拟炒股 → 组合调仓」面板加两条横幅：红条「今日开盘自动调仓未执行」+
  黄条「600 池维护今日未执行」，样式照抄 `drawdown_brake` 那块。
- 后台再补一条 `logger.warning`（每个 worker×日期 只吼一次，进程内集合去重）。
  **刻意不写文件**去重：写标记会与「运行标记」混为一谈，让一次漏跑看起来像跑过了。

### 3.6 给 `api/main.py` 接上 logging

`_setup_app_logging()`：root 挂一个 `logs/app.log` 的 FileHandler（utf-8、INFO、
带时间戳+级别）。幂等；**不用 `basicConfig`**（那会往 root 加 StreamHandler，和 uvicorn
抢控制台）。不会重复 uvicorn 的行 —— 已核对：`uvicorn` 与 `uvicorn.access` 都是
`propagate: False`，`uvicorn.error` 的父级 `uvicorn` 是 False，在 root 前就截断了；
另加 `_NoUvicorn` filter 兜底。

## 4. 验证

```
tests/test_calendar.py        13/13（新增 10 条覆盖 daily_worker_step）
全套 tests/test_*.py          31/31 文件全绿
```

新增用例覆盖：触发前 `wait` / 到点 `run` / `marker=今天` 跳过 / **交易日 03:00 重启必须
`wait` 而不是整日跳过（核心回归）** / 10-01 16:55 判 `skip_non_trading` / 23:58 重启
`skip_grace` / 挂起穿过窗口不迟到成交 / 挂起在窗口内补跑 / grace 边界含端点。

**明早（下一个交易日）实测**：`logs/auto_open_execute_date` 应等于当天日期（mtime 09:31），
`logs/app.log` 里应出现 `[auto-open] 触发...`。

## 5. 遗留 / 不做

- **`holiday_dates` 仍为空**（`config/config.yaml`）：交易所休市安排我不能编。本修复
  **不依赖它**（靠快照证据认节假日），但它的另一个用途 —— 长假前降仓 `_pre_holiday` ——
  仍然失灵，需要按交易所公告补。**另案处理**。
- `_scheduler`（自动刷新）也把 `enabled` / `interval` / `update_time` 在读一次后就固定了，
  且改动 `config.yaml` 仍需重启（`cfg` 在 import 时加载）。本次**不做热加载**。
- 本次未改 `_scheduler` / `_manual_stop_worker` —— 它们本来就是常驻的。

## 相关

- `quant/risk/calendar.py::daily_worker_step` —— 安全边界与其推导、单测入口
- `tests/test_calendar.py` —— 10 条调度决策用例
- `api/main.py` —— `_setup_app_logging` / `_read_marker` / `_warn_missed_once` /
  `_daily_run_status` / `_daily_runs_status` / 两个 worker
- `docs/2026-09-14-auto-open-fix.md` —— 同一族问题的上一次（import 途中抢跑 → NameError）
- `docs/2026-09-17-duplicate-auto-sell.md` —— 「读接口里下单」那次
