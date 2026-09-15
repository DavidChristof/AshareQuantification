# api 服务反复被 V8 崩溃打死 —— 指数日线从新浪（py_mini_racer）换到腾讯

**日期**：2026-09-15
**起因**：用户「帮我检查api服务」—— 当时 8001 无监听、系统里一个 python 进程都没有
**结论**：`quant/realtime/indices.py::fetch_index_daily` 是服务进程内**唯一**还在碰
`py_mini_racer`(V8) 的指数路径。换成腾讯源后，服务进程**不再加载 `mini_racer.dll`**。

---

## 1. 现象与硬证据

用户报障时服务已死。Windows 事件日志给出了确凿签名（不是我推测的）：

```
ProviderName : Application Error / Windows Error Reporting
事件名称     : APPCRASH
P1: python.exe   P4: mini_racer.dll   P7: 0x80000003

2026-09-15 09:26:46 / 09:28:10 / 09:29:37    <- 当天三次
```

拉近 6 天全部记录：**15 次，全部落在 `mini_racer.dll`**。

```
09-15 09:29:37, 09:28:10, 09:26:46
09-14 17:40:02, 17:26:05, 17:07:05, 16:58:55, 16:45:10, 09:36:06, 09:29:53, 09:28:04
09-12 16:40:57, 16:39:04
09-10 16:37:25, 14:35:32
```

`0x80000003` = `STATUS_BREAKPOINT`，即 V8 的 `Check failed:` 断点，对应此前在
`api.err.log` 里抓到的 `[FATAL:partition_address_space.cc(243)] Check failed:
!IsConfigurablePoolInitialized()`。

当天早上是**崩溃循环**：09:26 崩、重启、09:28 又崩、再重启、09:29 再崩 —— 三次启动都在
1~2 分钟内被杀掉。

---

## 2. 根因：谁在服务进程里建 V8

`fetch_index_daily`（原实现）直接调 `akshare.stock_zh_index_daily`。而 akshare 的**新浪**日线
源码里就是：

```python
js_code = py_mini_racer.MiniRacer()
js_code.eval(hk_js_decode)          # 跑 JS 解密
```

即 **每调用一次就新建一个 V8 isolate**。而 `fetch_index_daily` 是**在 api 服务进程内**被调用的：

| 调用点 | 经由 |
|---|---|
| `api/main.py:925` | `/api/market/indices/kline` |
| `api/main.py:2225` | `_market_trend_gate()` → 被 `/api/timing`、`/api/portfolio`、`/api/advisor` 共用 |
| `quant/data/selector.py:322` | `fetch_market_regime()`（选股，但在子进程里跑） |
| `scripts/33/36/37/40/41` | 离线回测（独立进程） |

⇒ 服务**一被请求就会把 V8 加载进来**，之后进程随时可能被 native 崩溃带走，
**整个服务连日志都来不及写就没了**（所以 `api.err.log` 里常常只有启动那几行）。

### ⚠️ 一处需要更正我自己的判断

我最初给出的机制是「同进程出现第 2 个 isolate 就会崩」。**这个说法被探针否掉了**：

```
4 次串行 ak.stock_zh_index_daily(...)          -> 存活
4 线程并发 ak.stock_zh_index_daily(...)        -> 存活
4 次串行 fetch_daily(...)                      -> 存活
12 次串行 fetch_daily(...)                     -> 存活
```

**在干净的裸进程里，串行和并发都复现不出这个崩溃。** 所以：
- 「内存紧张触发」——之前也这么写过，同样是错的（实测 5GB 可用时照样崩）；
- 「第 2 个 isolate」——本次探针不支持。

**确切触发条件仍未复现。** 但这不影响修法：崩溃**必然**要经由 `mini_racer.dll`，
把服务进程里的 V8 全部拿掉，这条路径就**不可达**——下面的验证正是按这个思路做的。

---

## 3. 修法：换腾讯源

`fetch_index_daily` 改用腾讯 K 线接口（**纯 requests，不碰 V8**）：

```
https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=sh000300,day,,,2000,
```

选腾讯而不是做子进程隔离，理由：**`indices.py` 的实时报价本来就是腾讯源**（`qt.gtimg.cn`），
换过去是同族归一，改动只落在一个函数里，不引入新进程开销。

### 3.1 换源**不能换数据** —— 逐值核对

6 个指数（`sh000300/sh000001/sz399001/sz399006/sh000905/sh000852`）× 2000 根重叠 bar
与新网源逐值比对：

| 字段 | 最大绝对误差 | 最大相对误差 | 差 >0.01 的根数 |
|---|---|---|---|
| open / high / low / close | 0.0050 | 1.2e-06 | **0** |
| volume | **0.000000** | 0 | **0** |

误差量级 = 腾讯保留 2 位小数造成的取整。**成交量在 ×100 之后与新浪完全相等**
（比值 min = max = 1.000000）。

### 3.2 源间三处差异（已在实现里逐条抹平）

1. **字段顺序不同** —— 腾讯一行是 `[date, open, close, high, low, volume]`，
   新浪是 `open/high/low/close`。**照搬新浪的顺序会把 `high` 当成 `close` 用**，
   且不会报错，只会让所有均线/趋势判断悄悄失真。已加不变量测试兜底
   （`high >= max(open, close)`、`low <= min(open, close)`）。
2. **腾讯盘中就返回当天的实时 bar**，新浪日线盘中只到昨天。不处理的话，
   `fetch_market_regime` / `_market_trend_gate` 会在**盘中**把「还没走完的当日 bar」
   当成收盘价 —— 正是 `docs/2026-09-14-trend-gate-semantics.md` 修掉的那类口径漂移。
   ⇒ 默认**盘中剔除当天、收盘后保留**（`_intraday()`，以 15:00 为界），与新浪源逐日一致。
3. **成交量单位不同** —— 腾讯给「手」、新浪给「股」，×100 等值。

### 3.3 已知限制（写在这里，别当成 bug）

腾讯单次上限 **2000 根**（实测 2500 起 `data[code]` 直接返回 `None`）⇒ 只能回溯到
**约 2018-06**（新浪能到 2002）。本项目所有回测起点都是 2020-01-01，**够用**。
若将来要更早的历史，得改用带起止日期的形式 `param=code,day,<start>,<end>,<count>,` 分批拼。
函数参数里留了 `count`，签名可扩展。

另外顺手修了一个 docstring 与实现不符的老问题：原 docstring 写着「失败时回退到上次缓存」，
但原代码**没有实现**这个回退（akshare 抛异常就直接往外抛）。现在按 docstring 兑现了。

---

## 4. 验证

| 项 | 结果 |
|---|---|
| `tests/test_indices.py` | 5 → **13/13** |
| 全量测试 | **27/27 文件通过** |
| 服务进程是否加载 V8 | `mini_racer/v8 NOT loaded` —— **崩溃源不可达** |
| 并发压测（6 个指数代码 × 2 轮 + 闸门端点并发） | 全部 200，**0 次 FATAL** |
| 闸门输出是否变化 | **逐字节一致**：`level 4480.08 / ma 4580.08 / below true / as_of 2026-09-14` |

最后一条是本次改动的核心验收：**只换源，不改行为**。闸门在同一个交易日报出的数字和理由
与换源前完全相同。

新增的 8 条测试里有两条是**根因护栏**，不是功能测试：

- `test_kline_row_order_is_tencent_not_sina` —— 锁住字段顺序（错了不会报错，只会静默失真）
- `test_kline_no_akshare_import` —— 用 **AST** 检查 `indices.py` 不再 import
  akshare / py_mini_racer（用 AST 而非搜字符串，这样 docstring 里解释「为什么离开 akshare」
  不会误报）

---

## 5. 未解决：服务进程里还有另一条 V8 路径

`_run_auto_update`（16:45 定时）→ `refresh_market_data` → `fetch_universe`
→ **每只股票一次 `fetch_daily`，而 `fetch_daily` 的源 1 就是新浪**（`quant/data/fetcher.py`）。
40 只 = 服务进程内 40 次 MiniRacer。**这条路径本次没动。**

不过必须说明：12 次串行 `fetch_daily` 的探针**存活**，所以**没有证据**表明它就是崩溃源；
下午那批崩溃（16:37~17:40）时间上吻合，但同样没能复现。下一步建议先做**复现**再动手，
可选方向：子进程隔离（照抄 `_run_daily_selection` → `scripts/30` 的模式），或给 `fetch_daily`
加一个非 V8 的源。

---

## 6. 教训

1. **native 崩溃要靠操作系统日志定位，不能只读应用日志**。服务被 native 崩溃带走时，
   应用日志里往往**只有启动那几行**（本次 `api.err.log` 就恒为 202 字节）。
   真正的签名在 `Get-WinEvent -LogName Application` 里。
2. **我给的机制解释必须能被探针验证**。这次「第二个 isolate 就崩」听起来很合理、
   也和现象吻合，但一测就不成立。**先写能证伪的实验，再下结论**；
   否则会把一个错误的机制写进文档和记忆里，比不写更糟。
3. **换数据源时，"值一样" 不等于 "语义一样"**。本次三处差异（字段顺序、当日 bar、单位）
   没有一处会导致程序报错，但两处会让结果悄悄错。**换源要按「逐值核对 + 语义对齐清单」走**。
4. **删掉一个依赖，要证明它真的没了**。这次除了测试，还直接查了服务进程的模块表
   （`mini_racer/v8 NOT loaded`）—— 这是唯一能证明「崩溃源不可达」的证据。

---

## 相关

- `docs/2026-09-14-trend-gate-semantics.md` —— 当日 bar 口径问题（本次差异 #2 正是同一类坑）
- `docs/2026-09-14-auto-open-fix.md`
- `quant/realtime/indices.py`、`tests/test_indices.py`
