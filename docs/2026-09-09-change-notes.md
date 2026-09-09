# 2026-09-09 改动说明：600 池每日维护改由「服务收盘后自动跑」，提前到 15:45

> 按「每次大改动在 docs/ 下留说明」约定。拟合入 main。

## 为什么改

600 池 + 影子 A/B（`scripts/26 → 27 → 28`）此前依赖 **Claude 会话的 17:17 定时**才跑：
- 时间晚（收盘后 2 小时），且要求当时开着 Claude 会话。
- 用户要求：**把每天跑 600 池数据的时间提前一点 —— 下午 3 点收盘时只要服务挂着就自动跑。**

因此把这条日常流水线挂到 **FastAPI 服务自身**（用户看盘时服务是开着的），与已有的
「9:31 开盘自动调仓」(`auto_open_execute`) 同一套思路：服务在触发时刻正运行才执行，
没挂则当日跳过，次日自然恢复。

## 改动

### config/config.yaml（`auto_refresh.shadow_ab`）
```yaml
auto_refresh:
  update_time: '15:30'      # 40 池收盘刷新（原样，先保证 market.db 今天就绪）
  shadow_ab:
    enabled: true
    run_time: '15:45'       # 服务在 15:45 挂着 → 自动跑 26→27→28（比 17:17 提前 ~1.5h）
    grace_min: 120          # 触发点过后 2h 内服务才开 → 当天仍补跑；超过当日跳过
    workers: 4              # scripts/26 大池拉取并发（服务在跑别用 8）
    recent: 700             # scripts/27 每只取最近 N 交易日打分
```

### api/main.py（新增两个函数 + 启动线程）
- `_shadow_ab_pipeline(sab)`：在**独立子进程**里依次跑 26→27→28（与每日选股 scripts/30
  同策略，subprocess 隔离崩溃），逐步释放内存；完整输出落 `logs/shadow_ab_<日期>.log`，
  并把每步尾部打进服务日志；强制子进程 UTF-8 防中文乱码。
- `_shadow_ab_worker()`：类比 `_auto_open_execute_worker` —— 睡到 `run_time` 触发；
  错过 `grace_min` 窗口 → 当日跳过；`logs/shadow_ab_date` 标记防同日重复（重启不重跑）；
  非交易日（含法定节假日 `risk.pre_holiday.holiday_dates`）不跑。跑 27 前先确保 40 池
  market.db 今日已刷新（若 15:30 刷新因服务刚起还没跑，这里补一次幂等 `_run_auto_update`）。
- 启动即挂 daemon 线程。

## 时序（服务当日一直开着时）
- 15:00 收盘
- 15:30 `auto_refresh` 例行：拉 40 池日线 → 重算信号 → 自动纸面调仓 → 触发每日选股
- **15:45 `[shadow]` 自动跑 26(补大池 559) → 27(两模型同 ~599 截面打分，落 shadow_ab.db) → 28(判 STABLE/NOT_YET)**

## 与原有 17:17 定时关系
Claude 的 17:17 定时**保留为兜底 + STABLE 提醒**：服务若当天没挂/跑挂，17:17 仍会补跑；
脚本全幂等（26 增量、27 upsert），重复跑无害。判定 STABLE 后仍只**提醒用户**上线 600 池，
不自动切换。

## 验证
- `python -m py_compile api/main.py` OK；`load_config()` 读到新 `auto_refresh.shadow_ab` 块。
- 回归 `tests/test_engine_hold / test_calendar / test_bracket_limit / test_volatility` 全 PASS
  （本改动只动调度，不动交易/风控逻辑）。
- 服务重启后：15:45 前开着 → 日志出现 `[shadow] 触发：维护 600 池...` 与三行步骤尾部；
  `logs/shadow_ab_2026-09-09.log` 生成。

> 注：若想 15:00 收盘就立刻跑（更早），需先手动把 `auto_refresh.update_time` 提前
> （40 池数据先就绪 27 才能全 599 打分），再把 `shadow_ab.run_time` 跟着提前。
