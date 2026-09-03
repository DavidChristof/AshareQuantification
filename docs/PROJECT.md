# 📘 A股量化预测系统 · 项目说明（协作者版）

> 本文档面向**加入本项目协作的朋友**：从 clone 到跑通全链路、理解架构、知道改哪里、以及遵守的约定。
> 技术演进细节 / 踩坑记录 / 教材对应，见根目录 `README.md`（作者长期维护的"主文档"）。

⚠️ **声明**：本项目仅用于学习与研究，不构成任何投资建议。模型预测存在大量噪声，请勿据此实盘交易。

---

## 1. 这是什么

一个**从零搭建的 A 股量化研究 + 模拟交易系统**，覆盖完整链路：

```
数据获取(akshare/腾讯/新浪) → SQLite 存储 → 特征/因子工程 → 深度学习+树模型预测
→ 回测/绩效 → 每日选股(600只大池) → 纸面交易(模拟盘) → FastAPI + Web 看板(Vue3)
```

**做过的关键实证结论**（读代码前先了解，能少走弯路，详见 README 对应小节）：

| 结论 | 含义 |
|---|---|
| 模型任务应是"横截面排序选股"，不是"预测单股涨跌" | v2 用相对标签 + 截面分位特征 + RankIC 评估 |
| **40 只同质大盘蓝筹上无价量信号**（RankIC≈0） | 瓶颈在"股票池"，不是模型 bug |
| 扩到 600 只后低波动/60日反转等异象浮现（h=20） | 每日选股候选宇宙 = 600 只大池 |
| 600 只全量 ML 暂无超额（LightGBM 决策闸门，9/3） | 暂不训练 600 全模型，改用可解释因子选股 |

---

## 2. 技术栈与运行环境

| 项 | 值 |
|---|---|
| Python | 3.12（作者本机 `D:/Python/Python3_12/python.exe`） |
| 数据 | akshare（东财接口**被墙不可用**；日常用 新浪/腾讯，免费源） |
| 数值/模型 | pandas≥2、numpy、torch(CPU 即可)、scikit-learn |
| 树模型 | **LightGBM**（`cross_model` 缺失时自动回退 sklearn HistGradientBoosting） |
| 服务 | FastAPI + uvicorn（**统一端口 8001**：`启动量化系统.bat` 与前端 `API_BASE` 均写死 8001，起服务请带 `--port 8001`） |
| 前端 | **单文件** `frontend/index.html`：Vue3(CDN) + ECharts，无构建步骤 |
| 存储 | SQLite：`data/market.db`(日线) + `data/minute.db`(5分钟线) |

> ⚠️ **依赖提示**：仓库根 `requirements.txt` 偏旧，**实际运行还需**：`pip install lightgbm scipy requests`（LightGBM 是 v2 集成 gbm 成员的默认实现；scipy 用于 RankIC/Fama-MacBeth）。建议朋友 clone 后顺手把这两行补进 requirements.txt 并提交。

---

## 3. 目录结构（clone 后能看到什么 / 看不到什么）

```
量化/
├── config/config.yaml        # ⭐ 全局配置（股票池/模型开关/风控/交易）——改行为先看这里
├── quant/                    # 核心库（无状态纯逻辑，应尽量与运行时数据解耦）
│   ├── config.py             # 配置加载
│   ├── data/                 # 数据层 fetcher/storage/loader/selector/universe
│   ├── features/             # 技术指标 + 旧版时序窗口流水线
│   ├── factors/              # 因子研究：analysis(IC)/regression/neutralize/alpha101
│   ├── models/               # LSTM/Transformer + cross_model(v2集成) + 训练/预测
│   ├── strategy/ timing/ risk/ advisor/   # 决策辅助层
│   ├── backtest/             # 回测引擎 + 绩效指标(含风险调整)
│   ├── trading/              # Broker 抽象 + 纸面账户 + 组合调仓引擎
│   └── realtime/             # 实时行情(新浪/腾讯)
├── api/main.py               # FastAPI（~1300 行，全部 HTTP 端点）
├── frontend/index.html       # Web 看板（Vue3 单文件，4 视图）
├── scripts/                  # 可复现脚本（见第 6 节脚本地图）
├── tests/                    # pytest 风格自测（14 个文件，直接 python 跑）
├── docs/PROJECT.md           # ← 本文档
├── data/    results/   paper/   logs/     # ❌ 运行生成，已被 .gitignore 排除
└── 启动量化系统.bat            # Windows 启动器（内含本机 python 路径，朋友需自行改）
```

**❌ clone 后没有、需要自己生成的运行时产物**（.gitignore 排除）：

| 产物 | 怎么生成 |
|---|---|
| `data/market.db` 日线库 | `python scripts/01_fetch_data.py` |
| `data/minute.db` 分钟库 | 系统运行自动积累；或脚本 `08`/`27` 补 |
| `results/model_v2/` 模型 | `python scripts/18_train_ensemble.py` |
| `results/minute_model.pt` | `python scripts/25_retrain_minute_v2.py` |
| `results/large_scan_cache.pkl`(600大池) | `21` → `22` 扫描下载（可选） |
| `results/daily_selection.json` 当日选股 | `scripts/16_daily_selection.py` 或 API `POST /api/selection/run` |
| `paper/*.db` 模拟账户 | 自动创建 |

---

## 4. 快速上手（第一次跑通）

```bash
# 1) 装依赖
pip install -r requirements.txt
pip install lightgbm scipy requests      # requirements 缺失的实际依赖

# 2) 抓日线（40 只股票池，2020 至今）→ 生成 data/market.db
python scripts/01_fetch_data.py

# 3) 训练主模型（v2 横截面集成 LSTM+Transformer+GBM）
python scripts/18_train_ensemble.py       # 约 10~30 分钟 CPU

# 4) 起 API（另开终端）
python -m uvicorn api.main:app --port 8001
#    或 Windows 双击「启动量化系统.bat」（记得先改里面的 python 路径）

# 5) 浏览器打开 frontend/index.html  →  看板

# 6) 跑一遍测试，确认环境 OK
python tests/test_cross_model.py          # 任一测试文件；全量见第 8 节
```

> 想跳过训练直接看前端界面？起 API 后看板仍可打开（模型缺失时会回退/报提示），但预测接口需要模型。

---

## 5. 架构与数据流（读懂代码入口）

### 5.1 主预测链路（日线 v2 横截面）

```
40只股票池日线(SQLite) ─┐
                        ├─► cross_dataset.build_enhanced_features
600只大池缓存(pkl,可选) ─┘        │  51 维特征
                                  ▼
          make_samples：X=(N,30天窗口,51特征)  y=相对标签(跑赢当日全池中位数)
                                  ▼
     cross_model.train_ensemble：LSTM + Transformer + LightGBM  软投票
                                  ▼
              CrossSectionalPredictor.make_signals_all（按天给全池打分排序）
```

关键文件：`quant/models/cross_dataset.py`（数据）、`cross_model.py`（模型+RankIC 评估）、
`api/main.py` 顶部加载 v2 → `results/model_v2/`。

### 5.2 每日选股 + 组合调仓链路（交易实际用这套）

```
600 只大池候选
  → 基本面：PE/ROE 质量分（实时快照 + 缓存 factor_cache.json）
  → 技术面（regime 门控，config selection.regime_gating 默认开）：按沪深300 近20日趋势
    判市场状态 → 切权重：上涨市动量主导 / 震荡低波+反转 / 下跌市反转+低波防守
     （源自 29 号实证「价量 alpha 是条件性的」：弱市才有真 alpha）
  → select_daily(n=12) → results/daily_selection.json（含 regime/tech_weights）
  → 一键调仓 portfolio/apply：候选=今日选股 topN，实时价撮合整手，风控(大盘弱势降仓/单票上限)
```

关键文件：`quant/data/selector.py`、`api/main.py` 的 `_portfolio_*`、配置 `selection:` / `trading.candidate_source` / `portfolio_risk:`。

### 5.3 盘中分钟信号（辅助，不否决）

5 分钟 K 线 → 分钟模型(未来25分钟涨跌二分类+Platt校准) → 给每只目标 `minute_prob` + 方向化 ±0.05 微调综合分。**`minute_veto` 默认 false**——分钟信号只做参考、不做买入否决（历史教训：未校准的极端概率曾把一键调仓买入全拦掉）。

### 5.4 数据流总览

```
每日 15:30 自动刷新(auto_refresh)：
  拉日线入库 → 重算特征 → 更新预测 → 跑选股 → (auto_rebalance) 自动纸面调仓
盘中每 10s(realtime) 快照 → 前端实时价格；每 60s(minute) 拉 5min 线 → 分钟信号
```

---

## 6. 脚本地图（`scripts/`，按编号）

| 分组 | 脚本 | 作用 |
|---|---|---|
| 数据 | `01_fetch_data` | 下载 40 池日线 → market.db |
| 数据 | `07_build_universe` | 重建/扩展股票池（universe） |
| 特征/因子 | `02_build_features` | 特征工程演示 |
| 因子研究 | `12_factor_ic` / `13_factor_regression` / `14_factor_neutralization` | RankIC / Fama-MacBeth / 中性化 |
| 因子研究 | `17_factor_mine` | Alpha101 + 自动因子挖掘 |
| 因子研究 | `15_cross_validate` | 时序 K 折交叉验证 |
| 训练 | `03_train` | 旧版单股 LSTM（history） |
| 训练 | `18_train_ensemble` | ⭐ v2 横截面集成（当前生产，`--members gbm` 可只训树） |
| 训练 | `11_retrain` | 拉新数据+重训主模型（自动备份旧模型） |
| 训练 | `08_train_minute` / `25_retrain_minute_v2` | 分钟模型（v2=时间切分+校准） |
| 实验/诊断 | `19_audit_rankic` / `20_ab_label_split` / `21`/`22_scan_*` / `24_diagnose_minute` / `26_probe_minute` / `28_lgbm_offline_validation` | 归因实验与扫描（想复现"池子才是瓶颈"的结论就看这些） |
| 回测 | `04_backtest` / `10_backtest_portfolio` | 单股 / 组合策略回测 |
| 每日更新 | `05_update` | 拉数据→预测→纸面调仓（模拟每日收盘跑） |
| 选股 | `16_daily_selection` / `23_preview_selection` | 生成/预览每日选股 |
| 账户 | `09_reset_accounts` | 重置模拟账户 |

---

## 7. API 一览（`api/main.py`）

| 端点 | 说明 |
|---|---|
| `GET /api/dashboard` / `/api/stocks` | 看板主数据 / 股票池状态 |
| `GET /api/predict/{symbol}` | 单股预测（概率+建议） |
| `GET /api/selection` · `POST /api/selection/run` | 查看 / 触发每日选股 |
| `GET /api/portfolio` · `POST /api/portfolio/apply` | 组合目标 / 一键调仓（候选来自选股 topN） |
| `GET /api/manual/account` · `POST /api/manual/order` | 手动模拟盘账户 / 下单 |
| `GET /api/minute/{symbol}` / `api/realtime` / `api/market/indices*` | 分钟信号 / 实时行情 / 大盘指数 |
| `GET /api/backtest/{symbol}` | 单股历史回测 |

---

## 8. 测试

```bash
# 单文件
python tests/test_cross_model.py

# 全量（作者环境：14 文件 84 PASS 0 FAIL）
$files = Get-ChildItem tests\test_*.py
$pass=0; $fail=0
foreach($f in $files){ $o = python $f 2>&1 | Out-String
  $pass += ([regex]::Matches($o,'PASS ')).Count
  $fail += ([regex]::Matches($o,'FAIL ')).Count }
"TOTAL PASS $pass FAIL $fail"
```

新增功能请配套加测试（现有覆盖：因子 IC / Alpha101 算子 / 回归 / 中性化 / 回测引擎 / 交易规则 / 组合调仓 / v2 模型 / 波动率）。

---

## 9. 协作约定

1. **永远不 push 大文件/运行时产物**：`.gitignore` 已排除 `data/ results/ paper/ logs/ *.db *.pt *.pkl *.csv *.bak`。加文件前先想：这是"代码/文档"还是"我机器上的运行数据"？
2. **分支流**：大改动开分支 `feat/xxx`，稳定后合 main。直接小修可 main 上 commit。
3. **commit 署名**：本项目 local 已配置 `DavidChristof <DavidChristof@users.noreply.github.com>`（别用全局 GitLab 邮箱 `13380231@gitlab.com`）。新 clone 后请自行 `git config user.name/user.email`。
4. **提交动作**：`git add -A && git commit -m "说明" && git push`（在项目根目录）。
5. **改行为先查 config**：`config/config.yaml` 是唯一全局开关（模型成员/选股宇宙/交易候选源/风控/分钟开关/自动刷新）。代码里尽量不写死业务参数。
6. **Windows/PowerShell 提示**：
   - git 推送时 stderr 会被 PowerShell 报成红字（`NativeCommandError`），**看到 `* [new branch]` 或 `git status -sb` 同步即成功**，非真错误。
   - Python 打印 `✓`/`²` 等字符在 GBK 控制台会崩 → 打印用 ASCII/中文。
   - 长日志用 `python -u`（否则管道下块缓冲看不到实时输出）。
7. **数据源现状**：东方财富接口不可用（被墙）；行情用 新浪/腾讯。免费 5 分钟历史上限 ≈ 腾讯 41 个交易日（想更长靠每日积累或付费源）。

---

## 10. 常见问题 FAQ

**Q：clone 后跑 `01_fetch_data.py` 很慢/失败？**
数据源走新浪等免费接口，可能限流；脚本一般有重试。网络需能访问国内行情源。

**Q：API 起来了但前端预测是空的？**
大概率 `results/model_v2/` 不存在（被 gitignore）→ 先 `python scripts/18_train_ensemble.py`。

**Q：一键调仓提示"资金不足以买入一手"或目标为空？**
交易候选来自"今日选股 topN"：先触发一次选股（`POST /api/selection/run` 或跑 `16`）。选股依赖实时快照/基本面缓存，首次较慢。

**Q：想改选股池 / 调仓风格？**
`config.yaml`：`selection.universe`（large=600 大池 / hs300）、`trading.candidate_source`（selection / model40 可回退）、`portfolio_risk`（风控）、`trading.max_positions`。

**Q：模型相关文件在哪学？**
v2：`quant/models/cross_dataset.py` + `cross_model.py` + `tests/test_cross_model.py`；
旧 LSTM：`quant/models/` 其余 + `03_train.py`（留作对照/教学）。

---

*文档维护：随代码演进更新。发现过时处请顺手改并提交 🙌*
