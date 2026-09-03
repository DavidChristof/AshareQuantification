# 📈 A股量化预测系统

一个**从零搭建**的 A 股量化项目：数据获取 → 特征工程 → 深度学习预测 → 回测验证 → **纸面交易跟踪** → Web 看板，全链路可运行、可扩展，并**预留实盘对接接口**。

> ⚠️ **声明**：本项目仅用于学习和研究，不构成任何投资建议。市场有风险，模型预测存在大量噪声，切勿据此实盘交易。

---

## 🏗️ 架构总览

```
┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ 数据获取    │ → │  SQLite 存储  │ → │  特征工程      │ → │  时序窗口     │
│ akshare    │   │ daily_bars   │   │  MA/RSI/MACD  │   │ X:30天→y:5天 │
└────────────┘   └──────────────┘   └───────────────┘   └──────┬───────┘
                                                               ▼
┌────────────┐   ┌──────────────┐   ┌───────────────┐   ┌──────────────┐
│ Web 看板    │ ← │  FastAPI     │ ← │  纸面交易      │ ← │ PyTorch 模型  │
│ Vue3+ECharts│   │  展示接口     │   │ PaperBroker   │   │ LSTM/Transformer│
└────────────┘   └──────────────┘   └──────┬───────┘   └──────────────┘
                                            │
                              ┌─────────────┴──────────────┐
                              │   Broker 抽象层（可替换）    │
                              │  PaperBroker → QMTBroker   │
                              └────────────────────────────┘
```

## 📁 目录结构

```
量化/
├── config/config.yaml        # 全局配置（股票池/窗口/模型/回测参数）
├── quant/                    # 核心库
│   ├── config.py             # 配置加载
│   ├── data/                 # 数据层：获取(fetcher)/存储(storage)/加载(loader)
│   ├── features/             # 特征层：技术指标 + 窗口流水线
│   ├── models/               # 模型层：LSTM/Transformer + 训练器 + 预测器
│   ├── strategy/             # 策略层：概率 → 交易信号
│   ├── backtest/             # 回测层：引擎 + 绩效指标
│   ├── trading/              # 交易层：Broker 抽象 + 纸面交易 + 调仓引擎
│   └── advisor/              # 决策辅助：模型概率 + 技术面 → 买卖建议
├── scripts/                  # 可运行脚本（按顺序执行）
│   ├── 01_fetch_data.py      # 下载数据
│   ├── 02_build_features.py  # 特征工程演示
│   ├── 03_train.py           # 训练模型
│   ├── 04_backtest.py        # 回测验证
│   └── 05_update.py          # 每日更新：拉数据→预测→纸面调仓
├── api/main.py               # FastAPI 接口（含 CORS / 手动下单）
├── frontend/index.html       # Web 看板（Vue3 + ECharts，三个 Tab）
├── data/                     # 运行时生成（SQLite + CSV 备份）
├── paper/                    # 运行时生成（自动纸面盘 + 手动模拟盘两个账户）
└── results/                  # 运行时生成（模型权重/图表/汇总）
```

## 🚀 快速开始

```bash
# 1. 安装依赖（建议用虚拟环境）
pip install -r requirements.txt

# 2. 下载 A 股日线数据 → SQLite
python scripts/01_fetch_data.py

# 3. （可选）查看特征工程效果
python scripts/02_build_features.py --symbol 600519

# 4. 训练 LSTM 模型（Transformer 加 --model transformer）
python scripts/03_train.py

# 5. 回测：策略 vs 买入持有
python scripts/04_backtest.py

# 10. 组合策略回测
python scripts/10_backtest_portfolio.py

# 11. 定期重训模型（纳入最新数据）
python scripts/11_retrain.py            # 拉最新日线 + 重训，自动备份旧模型
python scripts/11_retrain.py --no-fetch # 只重训不拉数据

# 12. 因子有效性检验（Rank IC / 平均 IC / ICIR）
python scripts/12_factor_ic.py          # 教材「因子检验模块」

# 13. 多因子回归（Fama-MacBeth + Pooling OLS）
python scripts/13_factor_regression.py  # 教材「多因子回归」：联合解释力+边际显著性

# 14. 因子中性化（剔除规模/波动风格暴露）
python scripts/14_factor_neutralization.py  # 教材「中性化」：残差纯因子 IC

# 17. Alpha101 因子 + 自动因子挖掘（受 FinHack 启发）
python scripts/17_factor_mine.py  # 候选因子池(~50) → IC/ICIR + 多因子回归筛选有效因子

# 15. 时序 K 折交叉验证（模型泛化稳定性）
python scripts/15_cross_validate.py --splits 5  # 教材「K 折」：均值±标准差

# 16. 每日选股（寻找大盘优质股，每日推荐）
python scripts/16_daily_selection.py --n 12     # 沪深300 → 流动性+基本面+技术面 → topN
# 前端 Tab1「📋 今日选股」面板也可一键触发（POST /api/selection/run），收盘后自动更新

# 3. 训练（含分类指标：P/R/F1 + 样本不平衡 + 阈值调优）
python scripts/03_train.py              # 输出 acc/precision/recall/f1 + 混淆矩阵 + 最优阈值

# 6. 每日更新 + 纸面调仓（模拟「每天收盘后跑一次」）
python scripts/05_update.py

# 7. 启动 API + 打开 Web 看板
python -m uvicorn api.main:app --port 8001   # 或 python api/main.py
# 用浏览器打开 frontend/index.html 即可查看
```

## 🎯 买卖决策辅助 & 模拟炒股

看板提供三个 Tab，面向散户的使用场景：

### Tab 1 · 买卖决策
每日给出每只股票的明确建议（**买入 / 卖出 / 观望**）+ 可读理由：
- 模型看涨概率 + 买入/卖出阈值
- 技术面多空投票：均线趋势、RSI 超买超卖、MACD 金叉死叉、短期动量
- 结合你的手动持仓标注「持有中」

### Tab 2 · 自动纸面盘
系统自动调仓（`scripts/05_update.py`）的账户跟踪，验证「如果让系统全自动交易会怎样」。

### Tab 3 · 模拟炒股
独立虚拟账户（**10 万初始资金**，配置在 `config.yaml` 的 `manual` 段）：
- 你手动选择股票 / 方向 / 数量下单，真实行情 + 佣金滑点模拟撮合
- 账户 / 持仓 / 盈亏 / 成交记录实时更新
- 与自动纸面盘**完全独立**，互不影响

```
下单示例：POST /api/manual/order
{"symbol": "600519", "side": "buy", "shares": 100}
```

## ⏱️ 自动刷新（日线级）

只要 **API 服务保持运行**，系统就会自动"活着"：

| 层 | 机制 | 配置项 |
|----|------|--------|
| 数据层 | 每个工作日 `15:30` 后自动拉最新行情入库 | `auto_refresh.update_time` |
| 预测层 | 更新后自动重算信号 + 自动纸面盘调仓 | `auto_refresh.auto_rebalance` |
| 前端 | 看板每 60 秒自动轮询，header 显示上次更新时间 | 前端内置 |

调度器后台线程每 `check_interval_minutes` 检查一次，满足「工作日 + 已过更新时间 + 今天未更新」才执行，**周末自动跳过**。所有参数在 `config.yaml` 的 `auto_refresh` 段。

> 备选：也可用 Windows 计划任务每天定时运行 `python scripts/05_update.py` 实现同样的效果（服务无需常驻）。

## � 实时行情（阶段一：盘中实时看盘）

用**新浪 + 腾讯双源**（公开接口，免费无鉴权）实时抓取快照，看板 Tab1 顶部实时显示：

- **现价 / 涨跌幅 / 今开 / 最高最低 / 成交额 / 买卖一档**，每 10 秒自动刷新（红涨绿跌）
- 后台 `QuoteManager` 线程轮询缓存，失败自动回退另一源，不影响看板
- 与现有日线系统**完全独立**，两套共存

```
quant/realtime/
├── quoter.py          # 新浪/腾讯实时快照抓取 + 解析 + 双源回退
├── manager.py         # QuoteManager 后台轮询缓存（10秒）
├── minute.py          # 新浪分钟K线抓取（5/15/30/60）
├── minute_store.py    # MinuteStore 分钟K线 SQLite 增量存储
└── minute_manager.py  # MinuteManager 后台增量更新（盘中60秒）
API: GET /api/realtime    实时快照（5秒轮询）
     GET /api/minute/{symbol}  分钟K线蜡烛图（60秒刷新）
配置: config.yaml 的 realtime / minute 段
```

## 📊 大盘参考模块（阶段六：市场参考）

作为组合决策的**市场温度参考**，Tab1 顶部新增「📊 大盘参考」面板：

- **四大指数实时**（腾讯指数通道）：上证指数 / 深证成指 / 创业板指 / 沪深300 的点位 + 涨跌 + 涨跌幅
- **指数日 K 线**（新浪源 `ak.stock_zh_index_daily`，30 分钟缓存）：下拉切换四大指数，ECharts 蜡烛图（红涨绿跌）+ 底部缩放，默认显示最近 150 根
- **股票池市场温度**：上涨 / 下跌 / 平的家数（市场宽度）——大盘普跌时谨慎加仓、上涨家数多则模型信号更可信
- `quant/realtime/indices.py`：`IndexQuoter` 腾讯指数实时（10 秒短缓存）+ `fetch_index_daily` 新浪指数日线（30 分钟缓存，失败回退缓存）
- `GET /api/market/indices`：`{indices, breadth, last_update}`；`GET /api/market/indices/kline?code=sh000001&days=120`：日 K 线
- **腾讯指数格式坑**：`s_` 通道返回 `1~名称~代码~当前点位~涨跌额~涨跌幅%~量~额~...`（字段索引 3/4/5，不是普通股票行情格式）
- 测试：`tests/test_indices.py`（5 项：行解析、正涨跌、非法行、缓存、失败回退缓存）

```
quant/realtime/
├── indices.py         # IndexQuoter 腾讯指数实时 + fetch_index_daily 新浪指数日K线
```

### 🗺️ 演进路线（小步快跑，每阶段可验收）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 一 | 实时看盘（快照级，5~10秒刷新） | ✅ 已完成 |
| 二 | 分钟K线（实时拼接存储 + 蜡烛图看盘） | ✅ 已完成 |
| 三 | 扩展规模（40只沪深300 + 并发抓取） | ✅ 已完成 |
| 四 | 分钟级模型（单独训练 + 盘中信号） | ✅ 已完成 |
| 五 | 聚焦模拟炒股（交易时间限制 + 账户重置） | ✅ 已完成 |

### 🗺️ 阶段四：分钟级模型（盘中短周期信号）

- `scripts/08_train_minute.py`：按日切窗训练分钟 LSTM（默认 5 分钟线，窗口 30 → 预测未来 25 分钟）
- 分钟模型单独保存 `results/minute_model.pt`，`/api/minute/{symbol}` 返回 `minute_prob` / `minute_signal`
- **盘中信号仅在交易时段（9:30-11:30 / 13:00-15:00）计算**——收盘后"未来 25 分钟"不存在，返回提示而非信号
- 训练/推理一致性：训练时日末窗口因标签为空被过滤，模型对"收盘结尾窗口"无经验，故收盘后不预测

**分钟模型 v2：重训 + 概率校准（9/3，修复"恒判 sell"极端概率）**
- 病根诊断（`scripts/24_diagnose_minute.py`）：旧模型 sigmoid 过饱和（近 2 天 720 概率 98% sell、中位 0.003，而真实日内上涨 35%+）→ 训练按"行序乱切"的评估假象 + 过拟合单边行情段
- `scripts/25_retrain_minute_v2.py`：**严格按时间切分**（训练样本时间 < 验证）→ **AUC=0.72**（证明分钟数据确有真实盘中排序信号，0.5=无信息）；Platt 校准 + **logit 截断**（clip 到验证区间，防分布外极端外推），概率均值≈真实基率
- 预测端自动应用校准：`trainer.save_checkpoint` 支持存 `calib`，`predict.ModelPredictor` 加载后 clip+sigmoid(A·logit+B)（dayline 模型无 calib 不受影响）
- **时间轴扩大（9/3）**：分钟数据原是系统上线(7/31)起仅 ~25 天。免费源探测（`scripts/26_probe_minute_history.py`）：东财历史分钟仍被墙，腾讯 5 分钟可回溯 ~41 交易日（7/8 起）→ `scripts/27_backfill_minute_tencent.py` 用腾讯回填 79 只，库从 8.7 万根扩到 15.5 万根（覆盖 7/8~9/3）。41 天重训：样本 +74%、验证 F1 0.575→**0.628**、AUC≈0.69（验证段更长仍稳定有信号）、校准后中位 0.42 且 90 分位 0.53（开始出现 buy 信号）；盘中已见区分度（个股可到 0.27）
- **诚实结论**：免费源下 5 分钟历史上限即腾讯 ~41 天；要覆盖更多行情只能靠系统长期运行每日自动积累（可持续免费），或付费源。`minute_veto` 保持默认关闭，分钟信号作方向化 ±0.05 微调 + 参考列

```
python scripts/08_train_minute.py              # 训练分钟模型（含按日切窗，旧版）
python scripts/25_retrain_minute_v2.py         # v2：时间切分 + Platt 校准（推荐）
python scripts/01_fetch_data.py                # 补充日线数据
```

### 🗺️ 按波动率动态止盈止损

固定百分比（止损 8%/止盈 15%）对**高波动股太窄**（频繁被震出）、对**低波动股太宽**（回吐过多）。
改用 **ATR（平均真实波幅）自适应**：止损/止盈线 = 成本 ± ATR 倍数，每只股票按自身波动定宽窄。

- `quant/features/technical.py` `add_atr`：TR = max(最高-最低, |最高-前收|, |最低-前收|)，ATR = TR 的 20 日均值
- `quant/risk/volatility.py`：ATR 计算 + 「ATR → 止损/止盈百分比」换算（`dynamic_pcts`），被 paper/updater/API 共用，口径统一
- 默认参数（config.yaml risk 段）：止损 2.5×ATR、止盈 3.5×ATR、移动止损 2.5×ATR，止损 clamp [3%,15%]、止盈 clamp [5%,30%]，止盈恒 > 止损
- 实测：青岛啤酒（低波动 ATR 1.5%）止损 -3.8%，兆易创新（高波动 ATR 6.4%）止损 -15% —— 高波动宽、低波动窄
- 池外股票（无日线算 ATR）自动回退固定百分比；自动盘/手动盘/API 持仓展示（波动率列 + 动态 % 标签）全部接入
- 单测：`tests/test_volatility.py`（7 项全过：ATR 计算 / 高低波动宽窄 / clamp 上下限 / 止盈>止损 / 动态触发 / 固定回退 / 移动止损用高点）

### 🗺️ 阶段五：聚焦模拟炒股（试验模式）

- **目前只做模拟炒股**（手动下单），试验一段时间后再考虑实盘；自动纸面盘后台仍运行但前端 tab 已隐藏
- **交易时间限制**：闭市/休市禁止下单（`manual.enforce_trading_hours`，默认开）——后端返回 400 + 前端按钮禁用并提示交易时段
- 模拟炒股与自动纸面盘是**两个独立账户**（`paper/manual_account.db` 与 `paper/paper_account.db`），总资产不同属正常
- 账户重置：`python scripts/09_reset_accounts.py`（清空持仓/成交/净值，资金回初始 10 万；不动日线/分钟行情数据）
- **模拟炒股可视化**：账户净值曲线（折线，每个交易日自动补点）、资产构成环形图（现金 vs 持仓）、持仓盈亏分布条形图（绿盈红亏）
  - 后端 `_sync_manual_equity()`：手动盘查询时若最新净值早于最新交易日则补快照，保证曲线每日有点
- **实时估值**：账户/持仓市值按**盘中实时价**估算（`PaperBroker.live_summary` + `_live_prices`：10 秒实时行情快照优先、日线收盘回退）；前端每 15 秒同步一次，**总资产跟随行情实时波动**（净值历史曲线仍按日线）
- **实时表排序**：点击表头按该列升/降序排列（代码/名称/现价/涨跌幅/今开/最高/最低/成交额/买一/卖一），再点切换方向，箭头标识
- **净值曲线按小时**：盘中每小时记一个实时估值点（`_sync_manual_equity` 交易时段内整点快照，同小时不重复），收盘记日点——时间轴细化为 小时 + 日
- **A股真实交易规则**（让模拟炒股更真实）：
  - **T+1**：当日买入份额当日不可卖（`paper.py` sell 查询当日买入量限制可卖额度）
  - **整手**：买入须 100 股整数倍（`manual.lot_size`，手动盘强制；自动盘不受限）
  - **印花税**：卖出单边 0.05%（`backtest.stamp_tax`，卖出费用=佣金+印花税）
  - **涨跌停**：一字涨停买不进/一字跌停卖不出（`quant/trading/rules.py`，创业板/科创板 ±20%）

### 🗺️ 阶段五·补充：回测增强（3 策略对比）

把**ATR 动态止盈止损**与**逐日滚动择时**接进回测，对比 3 种策略 + 买入持有：

- `quant/backtest/engine.py` `run()` 新增 `risk_cfg`/`atr` 参数（持仓期按 ATR 止损/止盈/移动止损）、`stamp_tax`（回测也真实计费）
- `quant/timing/roll.py`：`roll_regime`（逐日市场状态）与 `roll_timing_signal`（逐日滚动择时综合分，**只用截至当天数据、无未来泄漏**）
- `scripts/04_backtest.py`：baseline（纯概率）/ stop（+动态止损）/ timing（概率看多 且 择时不看空 + 止损）/ buy&hold 四路对比，输出跨股票平均绩效 + 净值曲线图

**回测结论（40 只，诚实记录）**：
- 模型概率信号本身已含选时（做多占比低、频繁进出），**夏普 0.76 vs 买入持有 0.51，回撤 -27% vs -59%**——选时价值主要来自模型
- 叠加 ATR 止损/择时在全池平均**无显著提升**（个别股如茅台：夏普 0.89→0.90、回撤 -27.4%→-25.2%）
- 原因：signal 保持看多时，止损卖出后**次日即买回** → 止损只增加换手成本；**止损/冷却期的保护只有在信号同步转弱时才体现**——这是有价值的真实发现，也是后续可调优方向（止损后冷却期、更宽 ATR 倍数）

### 🗺️ 模型评价指标与样本不平衡（教材 2.4）

- `quant/models/metrics.py`：`confusion_counts`（TP/FP/TN/FN）、`precision_recall_f1`、`classification_report`、`best_threshold`（验证集 F1 最优阈值）
- 训练器 `train_model`：
  - **pos_weight 类别权重**：正样本占比低时放大正类损失（教材 2.4.3 样本不平衡）
  - 每 epoch 记录 val **precision / recall / F1**（不只 accuracy）
  - 最佳模型按 **F1** 保存（类别不平衡下更全面），早停仍按 val_loss 防过拟合
  - 训练后自动在验证集**调优阈值**（F1 最优）并存入 checkpoint（`best_threshold`）
- `ModelPredictor` 读取最优阈值，`make_signal` 默认使用
- `scripts/03_train.py`：输出完整分类报告（acc/P/R/F1 + 混淆矩阵 + 最优阈值）
- **实测（重训）**：acc 0.533 / P 0.541 / R 0.550 / F1 0.546；阈值调优到 **0.30 后 F1 → 0.646**（+0.10）；正样本 49.7% 接近平衡 → pos_weight≈1
- 测试：`tests/test_metrics.py`（3 项：P/R/F1+混淆矩阵、除零保护、阈值调优提升 F1）

### 🗺️ 风险调整指标（教材 2.4：β / Alpha / IR / Treynor / Sortino）

把回测绩效从「只看收益」升级为「**每单位风险赚多少**」，都以**基准**为参照：

- `quant/backtest/metrics.py` 新增（自动识别净值/收益序列，按共同日期对齐）：
  - `beta`：策略对基准的敏感度（β=1 同涨跌、>1 更激进、<1 更稳健）
  - `jensen_alpha`：α = R_p − [R_f + β·(R_m − R_f)]，**α>0 表示跑赢基准的风险补偿**
  - `treynor_ratio`：超额收益 / β（每单位系统性风险的回报）
  - `information_ratio`：年化主动收益 / 跟踪误差（>0.5 良好、>1 优秀）
  - `sortino_ratio`：只用下行波动率惩罚（夏普改进版）
  - `capture_ratio`：上行/下行捕获率（涨时涨得多、跌时跌得少）
  - `compare_risk_adjusted`：多策略对比表 + `risk_adjusted_summary` 汇总
- 基准 = **代理指数**（股票池等权收盘，与回测一致）；`config backtest.risk_free`（无风险利率，默认 0.02）
- `scripts/10_backtest_portfolio.py` / `scripts/04_backtest.py`：回测后追加风险调整指标表，保存 `backtest_portfolio_risk_adjusted.csv` / `backtest_risk_adjusted.csv`

**实测（40 只 · 组合 top5 每 5 天调仓，基准=代理指数）**：
| 指标 | 组合策略 | 等权全持 | 代理指数 |
|---|---|---|---|
| β | 0.57 | 0.69 | 1.00 |
| Jensen α | **1.38** | 0.11 | 0 |
| 信息比率 | **4.33** | 0.37 | 0 |
| Sortino | 6.07 | 1.13 | 0.75 |
| 上行/下行捕获 | 1.10 / **0.13** | 0.76 / 0.68 | 1 / 1 |

- **解读**：组合策略 β=0.57（比基准更稳，持仓分散+现金缓冲）、α 高达 1.38（显著跑赢）、下行捕获 0.13（下跌市只跌基准 13%）——验证了「分散+概率选时」确实创造超额收益，而非靠承担更高风险
- 代理指数对照行完全自洽（β=1、α=0、IR=0、捕获=1），校验公式正确
- 测试：`tests/test_risk_adjusted.py`（8 项：β=1/β=2、α>0、IR=0、Treynor 公式、Sortino 除零、捕获率、基准恒定保护）

### 🗺️ 多因子回归（教材：Fama-MacBeth + Pooling OLS）

比单因子 IC 更进一步：把全部因子放进**同一回归**，控制其他因子后看**边际显著性**（解决因子共线性干扰）：

- `quant/factors/regression.py`（纯 numpy 手写 OLS，无额外依赖）：
  - `fama_macbeth`：两步法——① 逐截面日期回归 y(t+h)~因子(t) 得到当日系数 β_t；② 对 β_t 求均值=因子长期载荷、t 统计量=mean/(std/√T)
  - `pooled_ols`：合并所有日期×股票做一次 OLS，给出整体 **R²/adj_R²** + 各系数标准误与显著性
  - `drop_collinear`：**自动剔除完全共线因子**（如 rev_5 ≡ -mom_5），避免 X'X 病态导致系数/SE 不可靠；被剔除的因子在报告中标注
  - `_normal_pvalue`：erf 正态近似双尾 p 值（不依赖 scipy）
- `scripts/13_factor_regression.py`：多预测期输出 FM + Pooling 两张表，保存 `factor_fm_h{5,20}.csv` / `factor_pooled_h{5,20}.csv`

**实测（40 只 · 2020-2026，诚实记录）**：
- `rev_5` 与 `mom_5` 完全共线（互为相反数）→ **自动剔除**，SE 恢复正常
- **Fama-MacBeth**（逐日截面，稳健）：h=5 仅 rsi_14 显著（系数 -0.0002 极小）；h=20 仅 vol_20 显著 → 与 IC 检验结论一致：价量因子的边际预测力弱
- **Pooling**（大样本）：6 万观测下 7 个因子 t 都显著，但 **R² 仅 0.003**（因子只能解释 0.3% 收益变异）→ 经典「**统计显著 ≠ 经济显著**」反例：样本大时 t 检验容易显著，要看系数量级与 R²
- 多因子联合解释力极低，印证组合策略的超额来自**分散+选时**而非因子 alpha
- 测试：`tests/test_factor_regression.py`（7 项：OLS 恢复系数、FM 恢复符号/量级、输出结构+p 值边界、Pooling 恢复系数、样本不足保护、p 值边界、共线剔除）

### 🗺️ 因子中性化（教材：剔除风格暴露）

一个因子 IC 显著，可能只是暴露在某个风格上（如高动量同时小市值/高波动）。**中性化** = 把因子对风格变量做横截面回归、**取残差**作为纯因子，再检验 IC：

- `quant/factors/neutralize.py`：
  - `build_style_panels`：风格面板 `size_proxy`（ln 20 日均成交额=close×volume，规模/流动性代理）+ `vol_20`（波动率风格）
  - `neutralize_factor`：逐日横截面回归 factor ~ 风格变量 → 残差面板（同 shape，样本不足日 NaN）
  - `neutralize_report`：每因子×预测期输出 `ic_raw / ic_neutralized / delta_ic / survive`（|残差 IC|≥0.03 判为保留独立预测力）
- `scripts/14_factor_neutralization.py`：报告存 `factor_neutralization.csv`
- **设计**：vol_20 既是风格也是候选因子 → 不作为被中性化对象（对自身回归得 0 残差无意义），作风格保留；其余 8 个因子全部中性化

**实测（40 只 · 2020-2026，诚实记录）**：
- 8 个价量因子中性化前后 IC 变化**极小**（Δ 均在 ±0.012 内），且原始 IC 本就 <0.03 → 全部 survive=False
- 解读：这些因子本身无显著预测力（与 IC/回归检验一致），谈不上「被风格解释」；同时 40 只沪深300 大盘股同质化，规模/波动风格区分度低
- 方法论价值：学会了**「检验因子是否为风格代理」**的完整流程（构建风格→回归残差→对比 IC），是简历可讲的中性化实战
- 测试：`tests/test_factor_neutralize.py`（5 项：风格面板构建、纯风格暴露因子被剔除（残差与风格相关≈0+方差下降）、独立因子不被破坏、面板 shape、报告结构）

### 🗺️ Alpha101 因子 + 自动因子挖掘（受 FinHack 启发）

FinHack（github.com/FinHackCN/finhack，1152⭐）内置世界经典 Alpha101/Alpha191 因子集。我们实现算子库 + Alpha101 子集 + 自动因子挖掘，全部接入已有因子检验流水线：

- `quant/factors/alpha101.py`：
  - **算子库**（Alpha101 核心）：`rank`(横截面)、`ts_rank/ts_sum/ts_max/ts_min/ts_argmax`、`sma/stddev/delay/delta/returns`、`ts_corr/ts_cov`、`scale/signedpower/decay_linear/sign`——纯 pandas/numpy，时序算子沿列滚动、横截面 rank 沿行
  - **Alpha101 子集**（22 个，仅依赖 OHLCV/amount，vwap=amount/volume）：alpha001~060 等原编号因子
  - **自动因子挖掘** `mine_factor_panels`：从 OHLCV 自动组合 ~30 个候选（动量/反转/风险调整动量/波动/振幅/上下影线/量价相关/均线乖离/随机位置/量比额比）
  - `build_all_candidate_panels`：合并 Alpha101 子集 + 挖掘因子 = 候选因子池
- `regression.py` 支持传入**外部因子面板**（`factor_panels` 参数），挖掘因子可直接进多因子回归
- `scripts/17_factor_mine.py`：候选池 → Rank IC/ICIR 检验 + 多因子回归（自动剔除共线）→ 报告存 `factor_mine_report.csv`

**实测（40 只 · ~50 因子，诚实记录）**：单因子 |IC| 仍普遍 <0.05（弱）；但多因子回归中 **Fama-MacBeth：`ma_dev_60`、`range_20` 显著**（60日均线乖离/20日区间对5日收益有微弱正向预测），Pooling 下动量/反转/乖离/量价族 t 显著（R² 仅 0.006）——印证「技术因子对 A 股短期收益解释力整体弱，但均线乖离/区间位置相对最强」，是继续挖掘的方向
- 测试：`tests/test_alpha101.py`（10 项：ts_sum/stddev/delay/delta/rank 横截面/signedpower/ts_rank/decay_linear 算子正确性 + Alpha101/挖掘/合并面板结构）

### 🗺️ 模型 v2：横截面相对标签 + 增强特征 + 多模型集成（修正任务错配与评估泄漏）

**动机（任务错配）**：旧 LSTM 学"单股未来 5 天绝对涨跌"→ accuracy≈0.5；但真实任务是"选股 = 横截面排序"，二者错配。v2 把标签/特征/评估全部对齐到选股任务：

- **相对标签**：y = 未来 horizon 是否**跑赢当日全池中位数**（`cross_dataset.relative_label_panel`）
- **增强特征 51 维**（`cross_dataset.build_enhanced_features`）：基础技术(20) + **当日截面分位** rank_close/rank_volume/rank_ret1/rank_ret5（0~1 不做滚动标准化，保留横截面语义）+ **Alpha101 精选 12 个** + **挖掘因子精选 15 个**（ma_dev/range_20 等此前回归显著的族）
- **多模型集成**（`cross_model.train_ensemble`）：LSTM + Transformer + LightGBM（缺失回退 sklearn HistGradientBoosting）soft-voting；LSTM/Transformer 吃 (window×F) 序列，GBM 吃窗口 tabular 汇总（末帧+均值+标准差+斜率）
- **无泄漏按日期切分** `split_by_date`：验证段 = 时间后 20%（旧 `temporal_split` 是"40 只拼接后按行切 80/20"= 最后 8 只股票，**不是时间外**——这是关键修复）
- 评估放弃 accuracy/F1（近随机无意义），改 **RankIC / ICIR / Top-Bottom 分层**（`_cross_metrics`）
- `scripts/18_train_ensemble.py`：训练 + 新旧对照 + 存 `results/model_v2/`（API 启动优先加载 v2，缺省回退旧模型；回退删 `model_v2/` 重启即可）；`scripts/20_ab_label_split.py`：绝对 vs 相对标签 A/B

**实测结论（40 只 · 验证段 2025-05~2026-08 · 316 天，诚实记录）**：
| 模型 | 真·日期切分 RankIC | 说明 |
|---|---|---|
| 旧 LSTM（声称） | **0.318** | ❌ **行序乱切假象**：验证段混入训练样本（样本内过拟合），实盘亏损与此吻合 |
| 旧 LSTM（A/B 复现 absolute） | **−0.038** | 同特征/日期切分，真时间外≈0（甚至略负）→ 证明旧指标不可信 |
| v2 LSTM（relative） | −0.019 | 真时间外≈0 |
| v2 集成 ensemble | **−0.011** | 真时间外≈0 |

- **核心教训**：① 时间序列评估必须**按日期切分**，行序切分会把验证段混进训练（旧模型所有"验证指标"都虚高）；② 5 日尺度 × 40 只大盘股 × 价量特征 → **真实外推截面信号≈0**（与此前 IC/回归/中性化检验结论一致），模型无法"增强到显著"，因为底层信号本就极弱
- **v2 的实际价值**：把评估体系修正为**无泄漏 + 与选股任务对齐**（相对语义、RankIC 度量），可作为后续任何模型改造的**可信基准**；旧"K 折 F1 0.66"等结论因乱序切分需谨慎解读
- 测试：`tests/test_cross_model.py`（6 项：相对标签/增强特征列/样本装配与日期切分/GBM/截面指标/端到端装配），全项目 **84 项 PASS**

### 🗺️ 更大横截面实证：40 → 600 只，定位真瓶颈（股票池，2026-09-03）

v2 之后继续做「为什么模型无信号」的归因实验，四轮离线因子扫描（脚本 19~22，真·日期切分）：

| 池子（风格） | 数量 | 低波动 vol20（h20 IC） | 60日反转 rev60（h20 IC） |
|---|---|---|---|
| 现池（大盘抱团蓝筹） | 40 | ≈0 | +0.008 |
| +沪深300 补抽 | 100 | ≈0 | +0.035 |
| +中证500（中盘） | 350 | −0.030 | +0.032 |
| +中证1000（小盘） | **600** | **−0.050** | **+0.045** |

- **结论**：h=5 无论多大池都无信号（太短）；h=20 且池子≥350 后，经典异象成片浮现（600 只时 17/34 因子 |IC|≥0.03），方向全部符合学术（高波动→跌、跌多了→反弹、乖离大→回归）
- **真正瓶颈 = 股票池**：原 40 只同质蓝筹没有价量 alpha（所以模型/因子 RankIC≈0 不是 bug，是池子问题）；但即便 600 只，最强因子也只到 |IC|≈0.05（扣成本后仍紧）
- **落地**：不重训模型（避免破坏 40 只实盘链路），改为把实证因子接进「每日选股」——候选宇宙扩到 600 只（`selection.universe: large`，用 22 扫描缓存），技术面权重按实证改为 动量25/趋势15/低波动30/60日反转30（上方章节）
- 脚本：`21_scan_larger_universe.py`（沪深300 追加）、`22_scan_mid_small_cap.py`（中证500/1000，带超时保护）、`23_preview_selection.py`（大池选股预览）；下载缓存 `results/large_scan_cache.pkl`
- ⚠️ 经验：akshare 请求默认**无超时**（曾致成分获取挂一整晚）→ 扫描脚本统一加 `socket.setdefaulttimeout` + 线程 deadline + 重试；抓中小盘基本面偶发 `py_mini_racer` native 崩溃（进程级，无法 try 捕获）——缓存热后重跑即可

### 🗺️ 600 只全量 ML 的决策闸门：LightGBM 离线验证（2026-09-03）

上一节发现 600 只 h20 有单因子信号（|IC|≈0.05）后，问「**把 ML 模型直接训到 600 只值不值**」——完整 v2 集成(LSTM+Transformer+GBM) 在 600 只上预估 5~7h CPU / ~15GB 内存，训完无超额就白烧。因此用**最便宜的代理 LightGBM**，在**完全相同的数据装配**（51 特征 + relative 标签 + 严格按日切分）上先跑决策闸门，并加**全池等权基准**（beta）来分辨「真超额 vs 只是跟涨」（脚本 `28_lgbm_offline_validation.py`）：

| 池 | horizon | RankIC | ICIR | 全池均值(基准) | Top20 | **超额** |
|---|---|---|---|---|---|---|
| 40（现生产范围） | 20 | −0.026 | −0.13 | — | — | 负 |
| 600 | 5 | +0.029 | +0.21 | +0.59%/5日 | +0.51% | **−0.08%** |
| 600 | 20 | +0.030 | +0.18 | +2.50%/20日 | +2.10% | **−0.40%** |

- **读法**：扩池确实让 RankIC 由负转正（40→600），证明池子对了；但验证段（~2025-05~2026-08，约 315 日）**全池等权本身就在大涨**（h20 未来 20 日平均 +2.5%），模型 Top 层收益 ≈ 等权全池甚至略低 → 这阶段 51 个价量因子产出的排序主要是 **beta 而非 alpha**
- **决策**：**暂不投入 5~7h 训 600 全量集成**。LightGBM 是最便宜的全特征代理，它都跑不赢等权基准时，更贵的深度集成大概率也不赢；要做出超额需换更本质的 alpha（基本面/资金流/行业内中性化），或等 beta 弱化时段（震荡/下跌）重跑验证再定
- ⚠️ 诚实记录：`--quick` 冒烟 60 只子集曾出 RankIC 0.093，是**排序偏差**（取最小代码 60 只恰好踩中验证段强势票），全量 600 只后回到 0.03——小池子结果不可外推
- ⚠️ 内存经验：600 只 × 30 窗口的序列数组 (866k,30,51)≈4.2GB，`tabular_features` 内滚动方差再要 3.9GB 中间数组 → OOM。解决：`make_tabular_samples` 用 pandas `rolling` 直接流式聚合 4 块特征（`shift(1)` 对齐「决策日不含当日」，std 的 ddof 差异可忽略），内存从 GB 级降到 MB 级
- 结果留存：`results/lgbm_offline_validation.json`

### 🗺️ 时序 K 折交叉验证（教材：K 折，模型泛化稳定性）

单次时序切分指标依赖切分点；**K 折**用多个切分点独立评估，输出均值±标准差，识别指标是否稳定：

- `quant/models/cross_validate.py`：
  - `time_series_split`：类 sklearn TimeSeriesSplit 的**顺序切分**（训练恒在验证之前，可选 gap 留空防泄漏）——随机打乱会泄漏未来，时序必须顺序切
  - `cross_validate`：每折独立训练 + 输出 acc/P/R/F1/最优阈值，汇总均值±标准差；**不保存任何模型**
- `trainer.py` 重构：抽出 `_fit_model(train_ds, val_ds, cfg, device)` 供 K 折复用（`train_model` 接口不变）
- `scripts/15_cross_validate.py --splits 5`：各折明细存 `cross_validate_folds.csv`

**实测（40 只 · 5 折，诚实记录 + 重要发现）**：
| fold | accuracy | precision | recall | f1 | threshold |
|---|---|---|---|---|---|
| 1 | 0.494 | 0.494 | 1.000 | 0.661 | 0.30 |
| 2 | 0.494 | 0.494 | 1.000 | 0.661 | 0.30 |
| 3 | 0.499 | 0.498 | 0.998 | 0.664 | 0.38 |
| 4 | 0.500 | 0.500 | 1.000 | 0.666 | 0.30 |
| 5 | 0.511 | 0.504 | 0.928 | 0.653 | 0.30 |
| **均值±std** | 0.499±0.007 | 0.498±0.004 | 0.985±0.032 | **0.661±0.005** | — |

- **核心发现**：accuracy≈0.50（=掷硬币）但 F1=0.66 —— 因为阈值调优在**类别平衡**（正类 49.7%）下退化成「几乎全预测为上涨」（recall→1、precision→0.5），**F1 虚高是假象**（recall=1 时 F1=2·p/(1+p)，p≈0.5 → F1≈0.667）
- **教训**：类别平衡时不能单看 F1/阈值调优，必须同时看 accuracy + precision/recall 是否都健康；std 小只是「稳定地等于随机」
- 真实结论：模型无显著分类能力（与此前 IC/回归/中性化检验一致），组合超额来自分散+选时
- 模块含自动诊断：`accuracy<0.52 且 recall>0.9` 时提示「F1 来自全预测正类」
- 测试：`tests/test_cross_validate.py`（5 项：切分无泄漏、训练扩张、小样本降级、gap、小规模 CV 跑通+结构）

### 🗺️ 因子有效性检验（Rank IC / ICIR，对应教材）

- `quant/factors/analysis.py`：横截面 **Rank IC**（Spearman 秩相关）、**平均 IC**、**ICIR**（教材 1.3.2-5）
  - `build_factor_panels`：动量(mom_5/20/60)、反转(rev_1/5)、波动(vol_20)、量比(volume_ratio_20)、均线偏离(ma_dev_20)、RSI(14) 九大因子面板
  - `rank_ic_series`：逐日截面 IC；`summarize_ic`/`judge_factor`：按教材阈值评级（|IC|>0.05 有效、>0.1 优秀、ICIR>0.5 高质量）
- `scripts/12_factor_ic.py`：输出全部因子 × 预测期(5/20 日) 报告，保存 `results/factor_ic_report.csv`

**诚实结果（40 只 · 2020-2026）**：9 因子 × 2 预测期全部 `|Mean IC|≈0`、`ICIR≈0`，判定**弱/无效**。
- 单日 |IC| 高达 0.16-0.22 但均值≈0 → 截面 IC 波动大、方向不稳定（风格切换快）
- 印证：模型验证集仅 52%（要捕捉的信号本就弱）、组合策略靠**分散+再平衡**而非因子 alpha
- 这是有价值的负结果：学会「检验因子是否有效」正是教材核心；后续可扩展**基本面因子历史序列(PE/ROE)、市值/行业中性化、资金流**等
- 测试：`tests/test_factor_ic.py`（5 项：+1/-1 秩相关、汇总、面板列、报告结构）

### 🗺️ 组合交易策略（多股票动态调仓）

把单标的信号升级为**多股票组合的动态管理**，回测结果远超单标的：

- `quant/backtest/portfolio.py`：`PortfolioBacktest` 组合回测引擎（纯内存、逐日模拟）——每 N 天调仓，按**上涨概率排名选 top N 等权持有**，卖掉出榜的、买入新进的，计入佣金+滑点+印花税
- 复用现有能力：`predict_probability` 生成全池概率面板、`roll_regime` 做市场状态
- `scripts/10_backtest_portfolio.py`：组合策略 vs 下跌空仓 vs 等权全持 vs 代理指数

**组合回测结果（40 只，top5·每 5 天调仓）**：

| 策略 | 年化 | 夏普 | 最大回撤 |
|------|------|------|---------|
| **组合 top5 调仓** | **57.3%** | **1.54** | **-24.3%** |
| 组合+下跌空仓 | 30.5% | 1.05 | -23.8% |
| 等权全持 | 20.5% | 0.79 | -38.0% |
| 代理指数 | 12.0% | 0.50 | -44.3% |

- **组合分散 + 概率排名 + 定期再平衡**带来夏普 1.54（单标的概率策略 0.76、买入持有 0.51）
- 诚实发现：叠加「下跌市空仓」反而降夏普（1.54→1.05，回撤基本持平）——概率排名已隐含选时，空仓错过反弹
- 组合净值曲线图：`results/backtest_portfolio_equity.png`
**已接入模拟炒股（组合调仓 · 核心功能）**：
- `GET /api/portfolio`：**融合模型概率 + 择时 + 技术面 + 盘中分钟**的 topN 目标——每只目标带 `综合分 = 概率 + 择时分×0.1 + 技术面修正(buy+0.05/sell-0.05) + 分钟修正`，含择时动作与技术面理由
- **盘中分钟信号**（解决「开盘前不挂单、盘中决策」痛点）：日线模型只能看昨日收盘信号，`_minute_signal()` 用当日最新 5 分钟 K 线跑分钟模型（未来 25 分钟），盘中（9:30-11:30/13:00-15:00）给每只目标附加 `minute_prob/minute_signal`，并按方向修正综合分（`trading.minute_adj`，默认 buy+0.05/sell-0.05）
  - **注意**：分钟模型概率偏极端（未校准，常出 0.002/0.997），故用**方向化固定修正**而非 `(prob-0.5)*k` 放大，避免极端概率压垮日线分；前端「盘中分钟」列展示分钟概率+信号（收盘后显示「收盘/无」）
- `POST /api/portfolio/apply`：一键调仓，**基于现有资金决策**——总资产×仓位比例等权分配到每只目标，已持有且不足则加仓、达标则持有、未持有则买入（整手100股）；实时价优先、回退日线
- 前端模拟炒股 tab「⭐ 组合调仓」面板：目标表（综合分/概率/择时/技术面）+ 建议买入·卖出清单 + **资金决策表**（当前市值/目标市值/建议买入股数/预估金额）+ 一键执行按钮（交易时段可用）+ 市场状态标签
- **这是后续买卖操作的主要入口**：一键按模型信号与现有资金完成整个组合的再平衡

**交易池升级（9/3）：组合/一键调仓 = 从「今日选股（600 只大池）」挑 topN 交易**
- 背景：40 只训练池同质化（实证无价量 alpha），但"每日选股"已能在 600 只大池里用基本面+低波动/60日反转找到候选
- `config trading.candidate_source`：`selection`（默认）= 组合目标取 **每日选股 topN**（`max_positions=5`，综合分=基本面0.6+技术0.4，0~100）；`model40` = 旧逻辑（40 池模型综合分），可随时回退
- 实现：`selector.select_daily` 每只候选带 `price`（最新收盘，买卖成交价用）；API 新增 `_portfolio_from_selection()`（池内候选叠加模型/择时/分钟作参考列，池外仍可交易）与 `_build_prices()`（实时快照→信号收盘→选股 price 兜底，覆盖池外）
- GET/apply 目标表统一：综合分 0~100 展示、池外标的打「池外」标、模型列池内才有值；一键调仓直接买/卖大池候选
- 前端「⭐ 组合调仓」标题动态显示「交易池=今日选股·600只大池」，目标表加「模型(池内)/池外」列

**风控增强（复盘 9/2 后，对应亏损根因）**：
- `config portfolio_risk` 段：`market_weak_threshold(-1.0)` / `weak_position_pct(0.5)` / `max_stock_pct(0.20)` / `minute_veto(false)`
- **大盘弱势降仓**：`_market_weakness()` 用沪深300/上证实时跌幅 + regime 判断，弱于阈值 → 买入仓位自动降到 50%（避免满仓挨打）
- **个股仓位上限**：单票目标市值 ≤ 总资产×20%（避免重仓单票，如 9/2 招商轮船占 18% 却 -3.75%）
- **盘中分钟否决（默认关闭，9/3 修复）**：`minute_veto: false`——分钟模型概率**极端未校准**（常 0.002~0.006 恒判 sell），若开启会让盘中所有买入目标 `skip`，导致一键调仓"卖出能成交、买入全被拦"（显示调仓完成却没买）。**待分钟模型校准/重训后再置 true**；期间靠大盘弱势降仓+单票上限兜底
- **未成交原因透明化（9/3）**：`apply` 中被风控暂停（skip）或资金不足一手的目标会进入执行结果 `executed[].error`（前端显示"⚠ 未成交 + 原因"），不再静默；`_portfolio_allocation` 对"等权预算买不起一手"的目标标记 `skip(资金不足以买入一手)` 而非 `buy 0股` 误导
- 前端组合调仓面板顶部**大盘弱势黄色 banner** + 资金决策表「暂停」标签
- 实测：9/2 沪深300 -1.38% → weak=True，目标每只从 ¥18,800 自动降到 ¥9,895（仓位 50%），已有超标持仓不再加仓
### 🗺️ 阶段三：扩展规模（40 只股票池）

- `quant/data/universe.py`：从沪深300成分股自动生成股票池（固定种子可复现）
- `scripts/07_build_universe.py`：生成 N 只股票池并写入 config.yaml（含名称映射）
- 分钟K线**并发抓取**（`minute.concurrency=4`），SQLite 批量写入 + 写锁 + WAL 解决并发锁冲突

### 🗺️ 每日选股（寻找大盘优质股 · 每日自动更新）

把选股从「一次性建池」升级为「**每日推荐**」——每天收盘后自动从沪深300寻找当前基本面好且技术面走强的优质股：

- `quant/data/selector.py` 的 `select_daily(n, basic_topk, tech_topk, universe)`：
  - **候选宇宙**（`config selection.universe`）：`large` = 现池 40 ∪ 中小盘缓存 ~560（600 只，实证低波动/反转所在；缓存由 `scripts/22_scan_mid_small_cap.py` 下载）；`hs300` = 仅沪深300（旧行为，无缓存自动回退）
  - **分层**：大池 → 排除 ST + 实时流动性过滤（分批实时快照，自带名称/成交额）→ 基本面 PE/ROE（**缓存复用**，24h 内不重复抓）→ 技术面 topK **并发拉日线**
  - **技术面（600 只横截面实证权重）**：`mom20` 20日动量 + `trend` 均线偏离 + `vol20` 低波动 + `rev60` **60日反转**（超跌加分）→ `_score_tech`（0~100：动量25 + 趋势15 + 低波动30 + 反转30）
  - **综合分** = 基本面(PE/ROE)×0.6 + 技术面×0.4；无技术面时回退基本面分
  - 返回明细 `{code,name,pe,roe,fund_score,tech_score,mom20,vol20,rev60,total_score,in_universe}`
- `scripts/16_daily_selection.py`：跑选股，保存 `results/daily_selection.json` + `.csv`；`scripts/23_preview_selection.py`：命令行预览大池选股
- API：`GET /api/selection`（读最近结果）、`POST /api/selection/run`（后台手动触发，约1-2分钟）；**收盘后随自动刷新自动更新**（`config selection.auto: true`）
- 前端 Tab1「📋 今日选股」面板：topN 列表（综合分/基本面/技术面/PE/ROE/动量20/**反转60**/是否在池）+ 立即选股按钮（轮询等结果）
- **设计**：每日选股**不修改训练股票池**（避免频繁换池重训模型），只做推荐；基本面用缓存提速，技术面权重来自 600 只横截面实证
- 测试：`tests/test_selection.py`（5 项：强/弱打分、分数边界、None 回退、上涨趋势技术面、日线失败跳过）

**实测（2026-09-03 · large 大池 600 只）top12 示例**：宝丰能源 综合79.8（基本面100）· 贵州茅台 76.4 · 中国铝业 76.3 · 新华保险 75.8 · 云铝股份 75.5 · 安克创新 74.5 · 江波龙 73.7（**反转60 +42%** 超跌加分）· 世纪华通 72.5 · **金安国纪 71.5（非池新面孔，技术面 60 最高）** · 兆易创新 71.0（反转 +29%）——基本面 60% 权重仍主导前排，实证的「低波动 + 60日反转」把超跌/低波动优质股提上来，并能命中池外中盘候选

**背景：为什么给选股加大池 + 这两个因子**（详见下方「更大横截面实证」）：原 40 只全是大盘抱团蓝筹，价量上无 alpha（模型/因子 RankIC≈0）；把横截面扩到 600 只后，**低波动（h20 IC≈-0.05）与 60日反转（+0.045）**成为最强也最稳的价量异象——于是把它们作为可解释因子接入选股打分，而非继续堆模型。
- 40 只 ≈ 7 万训练样本，LSTM 跨股票泛化性更好

```
python scripts/07_build_universe.py --n 40   # 生成股票池
python scripts/01_fetch_data.py               # 下载日线
python scripts/03_train.py                    # 重训模型
```

> ⚠️ 合规说明：公开接口仅限**个人学习、低频使用**（间隔≥5秒）。高频抓取/商用/实盘信号请改用正规数据源（Tushare/券商行情/付费数据商）。

## �💼 纸面交易与实盘落地

系统通过 **Broker 抽象层** 实现「研究 → 模拟 → 实盘」的无缝演进：

```
Broker（抽象接口）
 ├── PaperBroker   ✅ 已实现：真实行情 + 模拟撮合，状态持久化到 SQLite
 └── QMTBroker     🔜 未来：券商 miniQMT 官方 Python API
```

**每日流程**（`scripts/05_update.py`，可配计划任务定时执行）：
1. 拉取最新行情入库
2. 用训练好的模型生成每只股票的上涨概率与信号
3. 调仓引擎执行：选出概率 > 阈值的股票（最多 3 只）→ 卖出淘汰 → 买入补仓
4. 按当日收盘价成交（含佣金 + 滑点），记录成交与净值快照

**Web 看板**（`frontend/index.html`，Vue3 + ECharts 单文件，无需构建）：
- 账户总览：总资产 / 现金 / 累计收益 / 看多信号数
- 净值曲线、股票池预测概率、纸面持仓、成交记录
- 直接浏览器打开即可，配合 FastAPI 使用

**实盘落地的现实路径**（国内 A 股）：
| 方案 | 正规性 | 门槛 | 建议 |
|------|--------|------|------|
| miniQMT / QMT | ✅ 官方 | 券商开通（通常 50 万，部分券商活动降门槛） | **首选**，Python API |
| Ptrade | ✅ 官方 | 较高 | 大资金 |
| easytrader | ⚠️ 非官方 | 无 | 有封号风险，不推荐 |
| 券商 Open API | ✅ | 高 | 企业/白名单 |

拿到券商权限后，只需写一个 `QMTBroker(Broker)` 实现 buy/sell/query 映射到券商 API，
把 `TradingEngine` 的 broker 换成它即可实盘，**其余代码零改动**。

## 🧠 核心设计理念（面试可讲）

1. **防止未来信息泄漏**
   - 特征用**滚动 z-score**（只用窗口内数据求均值/方差），不用全局统计量
   - 训练/验证**按时间顺序切分**，不用随机切分
   - 回测用历史价格逐步预测，而非一次性预测

2. **预测目标可解释**
   - 不做"预测具体价格"（不现实），而是预测**未来 5 天涨跌方向**（二分类）
   - 输出上涨概率，由策略层决定是否买入

3. **成本真实化**
   - 回测计入佣金（万三）+ 滑点，避免"看起来很美"

4. **分层解耦**
   - 数据 / 特征 / 模型 / 策略 / 回测相互独立，任一层可替换升级

## 📚 学习路线图（跟着做，逐个理解）

| 阶段 | 你要理解的关键点 | 对应文件 |
|------|----------------|---------|
| 1. 数据 | akshare 接口、复权、SQLite 长表设计 | `quant/data/fetcher.py` |
| 2. 特征 | MA/RSI/MACD 公式含义、滚动标准化 | `quant/features/technical.py` |
| 3. 窗口 | 为什么用"过去30天预测未来5天"、标签构造 | `quant/features/pipeline.py` |
| 4. 模型 | LSTM 门控机制、Transformer 自注意力 | `quant/models/lstm.py` |
| 5. 训练 | 二分类损失、早停、时间序列切分 | `quant/models/trainer.py` |
| 6. 回测 | 净值曲线、夏普/回撤/胜率 | `quant/backtest/` |

## 🔧 常见改进方向（进阶练手）

- **多标的组合**：按预测概率给多只股票打分、轮动持仓
- **特征扩展**：加入成交额占比、北向资金、行业板块等
- **模型升级**：改为**回归**预测收益、加 Attention 门控、尝试 TFT
- **风险控制**：波动率目标仓位、止损止盈、最大回撤熔断
- **实盘模拟**：对接券商接口（需合规考虑）

## 🛠️ 技术栈

Python 3.12 · akshare · Pandas · NumPy · PyTorch · scikit-learn · FastAPI · SQLite · Matplotlib
