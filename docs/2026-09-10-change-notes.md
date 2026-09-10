# 2026-09-10 改动说明：NaN 价格事故（接口 500）根因修复 + 防御加固

> 按「每次大改动在 docs/ 下留说明」约定。已拟合入 main。

## 现象（检查服务时发现）

自动纸面盘的相关接口全部 **500**：`/`、`/api/dashboard`、`/api/account`；
其余接口 200，**手动盘（manual_account.db）不受影响**（你练手交易的账户是好的）。

## 根因链（03-09 收盘后自动调仓触发）

1. `quant/trading/updater.py:rebalance_auto` 从信号表取价：
   `prices = {s: float(t["close"].iloc[-1]) ...}` —— 个别行（该日 **300750**）`close` 是 **NaN**。
2. `quant/trading/engine.py` 的守卫 `if not price:` 对 NaN **恒为 False**（NaN 是 truthy），
   NaN 价格一路通过；`paper.buy` 里 `total_cost > cash` 对 NaN 也恒 False → **NaN 单子成交**。
3. `cash = cash - NaN = NaN`，SQLite 把 NaN 写成 **NULL** →
   `paper.query_cash()` 的 `float(row[0])`（row[0]=None）抛 `TypeError` →
   三个接口 500。同时落了一条 NULL 成交（#15）与一条 NULL 持仓（300750）。

## 修复

### 1) 防御加固（主修复）
- `quant/trading/paper.py`
  - 新增 `_valid_price(price)`：非 None / 非 NaN / 非 inf 且 > 0。
  - `buy()/sell()`：价格无效或 shares 非有限 → **直接拒单**（不再写 NULL）。
  - `query_cash()` / `account_summary()`：容忍 NULL（回退 0 / 现金），历史坏数据不再打挂接口。
  - `snapshot_equity()`：只按有效价计市值（防 NaN 净值落库）。
- `quant/trading/engine.py`：全部 `if not price` → `if not _valid_price(price)`；
  `_current_equity` 只累加有效价（防总资产变 NaN）。
- `quant/trading/updater.py`：价格映射剔除无效收盘价（源头拦截）。
- `api/main.py:_run_auto_update`：重建信号后补 `_sanitize_signal_table`（close 缺失→前收、
  prob NaN→0.5），自动路径与 API 展示路径口径一致。

### 2) 数据修复（自动纸面盘，已备份）
`paper/paper_account.db`（备份 `paper_account.db.bak_20260909`）：
- 删坏成交 `#15`（2026-09-09 300750 buy，shares/price/fee/amount 全 NULL）；
- 删坏持仓 `300750`（shares/avg_cost NULL）；
- `cash` 由 NULL 置回 **0.0**（与 09-08 收盘快照一致；09-09 无合法成交）。

修后：`cash=0.0, equity=100586.41, 7 只正常持仓`，成交 14 条。

## 验证
- 新增 `tests/test_nan_guard.py`（6/6 PASS）：NaN 价/份额拒单、现金不被污染、
  引擎跳过 NaN 价票、混合价只买有效票且成交皆有限、`cash=NULL` 不再抛错。
- 回归 `tests/test_engine_hold / test_calendar / test_bracket_limit / test_volatility` 全 PASS。
- 修复数据后接口立即恢复 200；重启服务加载新代码后再扫 **18 个接口全 200**。

## 服务重启（本次采用，记录备查）
旧进程（PID 30064，旧代码）在新防护生效前每 30 分钟会重试自动调仓、可能再次写坏 cash。
本次在**确认当日 09:31 开盘自动调仓已完成**（marker `logs/auto_open_execute_date` = 2026-09-10）后
替换进程，避免打断半程调仓：
```powershell
# 1) 结束旧进程   2) 分离方式重启（同 bat 命令，日志 logs/api.log）
Start-Process -FilePath '.venv\Scripts\python.exe' `
  -ArgumentList '-u','-m','uvicorn','api.main:app','--host','127.0.0.1','--port','8001' `
  -WorkingDirectory <repo> -WindowStyle Minimized `
  -RedirectStandardOutput 'logs\api.log' -RedirectStandardError 'logs\api.err.log'
```

> 注：本次未改任何策略/参数；纯属健壮性修复。新 600 池收盘自动流水线（`auto_refresh.shadow_ab`，
> 15:45）随本次重启一并生效（见 `docs/2026-09-09-change-notes.md`）。
