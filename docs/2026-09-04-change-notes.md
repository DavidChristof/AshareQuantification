# 2026-09-04 改动说明（手动组合调仓 · 休市/长假风控）

> 本文件按「每次大改动在 docs/ 下留一篇 markdown」的约定记录。涉及的提交均已合入 main：
> `844822b`（手动盘整手决策）· `86533e1`（资金顺位补买 + 双阈值并入 main）· `4124ffc`（休市感知）· `648b416`（长假前降仓）。

## 一、自动纸面盘：双阈值「多日持有 + 部分减持」（引擎，本次并入 main）

**改动前**：只持当日概率 top5，掉出核心榜即清仓 → 近似每日整仓轮动（高换手、难吃趋势）。
**改动后**（`quant/trading/engine.py` + config `trading`）：

- 掉出核心榜但概率仍 ≥ `clear_threshold(0.45)` → **减持到半仓观察**，不清仓
- 跌破 `clear_threshold`（或无当日信号）→ 全仓清出
- 仍在核心榜 → 照常补仓；止盈止损/ATR 先行逻辑不变
- 设 `trading.clear_threshold: null` 即回退旧「掉榜即清」

**入口**：`_run_auto_update` → `rebalance_auto` → `TradingEngine.rebalance`（`updater.py`/`05_update.py` 已传参）。
**测试**：`tests/test_engine_hold.py`（减持保留 / 破线清仓 / 旧模式回归）。

## 二、手动一键调仓：整手决策，已持仓不再误报「资金不足」

**问题**：A股只能整手(100股)买，原逻辑把「已持有但差额不足一手」也算一次买入 → 全报「资金不足以买入一手」，明明现金充足却什么也买不了。
**修复**（`api/main.py:_portfolio_allocation`）：按**目标整手股数**决策——
- 已持有且达目标整手 / 差额不足一手 → **持有**
- 已持有、不足且可整手补 → `加仓`
- 未持有、预算/现金够 → `买入`；仍买不起 → 明示「现价高/超单票上限」，不再笼统报资金不足
- `/api/portfolio` 的「建议动作」改为与资金决策对齐

## 三、手动组合：按实时资金「顺位补买」，凑出买得起的实际组合

**问题**：目标固定 top5，若前几名买不起一手（如茅台 1 手≈13 万），名额空置、资金闲置，榜单后面大量买得起的票（德业/中国铝业…）却不会买。
**改动**（`api/main.py:_portfolio_from_selection`）：候选 = 今日选股全部 12 只（score 降序）；
1. **保留**已持有且仍在今日候选内的（低换手）
2. 空位按排名**顺位补**「能整手起配」（1 手 ≤ 单票上限且现金够）的候选
3. 买不起的不入选，名额让给后面可买的 → 实际目标 ≤ `trading.max_positions`(5)

卖出语义不变：已持有但彻底掉出今日候选 → apply 卖出。

## 四、休市 / 法定节假日感知

**背景**：之前只认「周六日」，法定节假日落在工作日时（国庆 10/3 之类），交易时段判定与 15:30 自动刷新都会误放行/空跑。

**改动**（api `_in_trading_hours` / `_run_auto_update`，storage `MarketDB.latest_date`）：
- 交易时段：复用实时行情时间戳 `_market_status`——最近成交日 ≠ 今天（`market_closed`）→ 判休市（法定节假日），不放行下单/分钟信号；快照缺失回退按周几+时段
- 自动刷新：拉取后**库内最新交易日未推进** → 跳过调仓/选股（休市/已更新不再空跑交易）

## 五、长假前降仓风控

**背景**：长假后首日跳空风险模型无感知，属风控侧问题。
**改动**：`quant/risk/calendar.py`（纯逻辑）识别「明起连续休市 ≥ N 天」；config `risk.pre_holiday`：

```yaml
pre_holiday:
  enabled: true
  min_days_off: 3      # 明起连续休市 ≥3 天 → 视为长假
  reduce_to_pct: 0.5   # 长假前目标仓位上限（50%）
  holiday_dates: []    # ⭐ 当年法定休市日（不含周末），每年交易所公布后维护一次；留空则不生效
```

- 触发时：`_portfolio_from_selection`（选目标）与 `_portfolio_allocation`（分配）的 `pos_pct` 取「弱势 vs 长假」**更严**的值
- `/api/portfolio` 返回 `market_weak.pre_holiday`；`apply` 加「⚠ 长假前…买入仓位降至 X%」提示
- **注意**：长假在未来，行情无法推断 → 必须维护 `holiday_dates`；普通周末(2天)不会触发
- **测试**：`tests/test_calendar.py`（周末不触发 / 法定休市识别 / 长假前最后交易日触发）

## 涉及文件一览

| 层 | 文件 |
|---|---|
| 配置 | `config/config.yaml`（trading.clear_threshold/hold_trim_pct、risk.pre_holiday） |
| 引擎 | `quant/trading/engine.py`、`quant/trading/updater.py`、`scripts/05_update.py` |
| API | `api/main.py`（_portfolio_from_selection / _portfolio_allocation / _in_trading_hours / _run_auto_update / _market_weakness / apply 提示） |
| 存储 | `quant/data/storage.py`（latest_date） |
| 新增模块 | `quant/risk/calendar.py` |
| 测试 | `tests/test_engine_hold.py`、`tests/test_calendar.py` |

## 验证

- `python tests/test_engine_hold.py`、`python tests/test_calendar.py` 全 PASS
- `/api/portfolio` 实测：已有持仓显示「持有」而非误报资金不足；买不起的票让位、补入可买的一手；`market_weak.pre_holiday` 字段存在（未配日期为 `{}`）
- 服务重启后 `GET /api/stocks` 200

## 使用提醒

- **长假降仓要生效**：每年初把当年法定休市日（非周末部分）填进 `risk.pre_holiday.holiday_dates`
- **回退旧调仓**：`trading.clear_threshold: null` → 自动盘回到「掉榜即清」
- 手动盘「顺位补买」候选 = 今日选股（需先有当日选股结果）；候选池范围与目标数可调 `selection` / `trading.max_positions`

## 补充 · 当日收益改为「较昨收」口径

**问题**：Tab3 模拟炒股卡片「当日收益」原来 = 现总资产 − 初始本金（累计口径，标签还写"较本金"）。
**修复**：`quant/trading/paper.py: live_summary` 新增 `prev_close_equity / day_pnl / day_return`——
基准 = 净值历史里「日期早于今天」的最后一个点（即上一交易日收盘权益）；
`frontend/index.html` 当日收益用该基准（`dayBase`，无历史回退本金），标签改「较昨收」。
首日/账户刚重置（无历史净值）时自动回退本金口径，不报错。

## 补充二 · 追高/接飞刀保护 + 回测式开盘执行（组合调仓买入）

**背景**：盘中随手点"一键调仓"会吃到当日高点（如东鹏 122.11 买在当天最高区 +2.7%）。

**追高/接飞刀保护**（`portfolio_risk.chase_guard`，默认开）：
新买/加仓前按 **现价 vs 昨收** 当日涨跌幅判定——涨幅 > `high_limit_pct`(2%) → 跳过（追高）；跌幅 < `drop_limit_pct`(-3%) → 跳过（接飞刀）。理由写清百分比。基准昨收：SIGNALS 最新收盘 / 池外候选回退选股 price。

**回测式开盘执行**（`portfolio_risk.exec_mode`，默认 `live`）：
- `exec_mode: open` 时，组合调仓只能在**开盘后前 N 分钟**（9:30 起 `open_window_minutes=15`）执行；
- 成交用**参考价 = 昨收 × (1 + `open_premium_pct`=0.5%)**（近似开盘成交/回测口径），不再随手吃实时盘中高价；
- 改回 `live` 即恢复旧行为。

**验证**：`_guard_check` 实测——东鹏 122.11（+2.68%）→ 拦；+0.9% / 现价 +1.8% → 放行；-3.3% → 拦。服务重启后 `/api/portfolio` 正常。
