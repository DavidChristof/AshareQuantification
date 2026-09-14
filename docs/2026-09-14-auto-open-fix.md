# 事故：开盘自动调仓静默失败 —— 线程启动竞态 + 失败也写 marker

**日期**：2026-09-14（周一）
**影响**：当日 09:31 的开盘自动组合调仓未执行（**已于 09:41 补跑完成**，结果为「按趋势闸门不开新仓」，见文末）
**性质**：两个独立缺陷叠加，一个让调仓崩、一个把失败伪装成成功

---

## 1. 现象

用户 09:29 双击 `启动量化系统.bat`，服务没起来。检查：

```
端口 8001: 只有 SYN_SENT，无 LISTEN
tasklist:  没有任何 python 进程
logs/api.log: 0 字节，时间戳 09:29
```

`api.log` 是 0 字节是因为 bat 里写的是

```bat
start "..." /min "%PYTHON%" -u -m uvicorn api.main:app ... > "%LOG_FILE%" 2>&1
```

**重定向挂在 `start` 上而不是子进程上**，所以该文件只能反映 `start` 自身的输出，子进程日志不在这里。（bat 另有最多 6 分钟的等待循环。）

用 `python -c "import api.main"` 抓真实报错，得到：

```
[auto-open] 执行异常: name 'portfolio_apply' is not defined
  File "api/main.py", line 337, in _auto_open_execute_worker
    summary = portfolio_apply(force_open_ref=True)
NameError: name 'portfolio_apply' is not defined
```

---

## 2. 缺陷一：线程启动竞态（NameError 的根因）

| 位置 | 内容 |
|---|---|
| `api/main.py:351`（改前） | `threading.Thread(target=_auto_open_execute_worker).start()` ← **模块 import 途中就启动** |
| `api/main.py:1791`（改前） | `def portfolio_apply(...)` ← **定义在 1400 行之后** |

worker 的触发条件是「**现在已过 09:31 且距触发点不超过 grace_min(25) 分钟**」。于是：

> 服务只要**恰好在 09:31~09:56 之间启动**，worker 就会在模块还没执行到 1791 行时抢跑，
> 此时 `portfolio_apply` 尚未绑定 → NameError。

**这正是当天命中的情况**：09:29 启动，模块加载需一两分钟，09:31 触发时函数还没定义。

同类竞态不止一处 —— `_scheduler`(原 300 行)、`_shadow_ab_worker`(原 506 行) 同样是**在定义处立即启动**，
它们引用的后半文件函数理论上同样可能未绑定。三个线程本次**一并**移到文件末尾。

### 为什么以前没暴露

2026-09-11 那次自动调仓是**成功**的（marker = 2026-09-11）。因为那天服务是**前一天晚上就在跑**的，
到 09:31 时模块早已加载完毕，不存在竞态。**只有「在 09:31~09:56 窗口内冷启动服务」才会触发。**

---

## 3. 缺陷二：失败也写 marker（把失败伪装成成功）

改前代码：

```python
try:
    summary = portfolio_apply(force_open_ref=True)
    logger.info("[auto-open] 完成: %s", summary)
except HTTPException as exc:
    logger.info("[auto-open] 跳过（%s）", exc.detail)
except Exception as exc:
    logger.error("[auto-open] 执行异常: %s", exc, exc_info=True)
marker.parent.mkdir(parents=True, exist_ok=True)
marker.write_text(today.isoformat(), encoding="utf-8")   # ← 在 try 之外，无条件执行
```

`marker.write_text()` **写在 try/except 之外**，所以**即使调仓抛异常（什么都没执行），marker 照样被写上**，
当日被永久标记为「已自动执行」，不会再补跑 —— 异常被吞进日志，表面上一切正常。

### 同类缺陷的第 3 次

memory 记录 2026-09-11 影子流水线修过一模一样的问题（「**必须校验通过才写** —— 曾因无条件写 marker，
导致 15:45 那次没拿到当日日线时兜底定时误判已跑而跳过补跑」，见 `docs/2026-09-11-shadow-retry.md`）。
那次只修了影子流水线，**auto-open 这条路径漏掉了**。以后凡是「写 marker 防重复」的地方，
一律按「确认处理过才写」实现。

### 本次修法

```python
handled = False
try:
    summary = portfolio_apply(force_open_ref=True)
    handled = True
except HTTPException as exc:          # 业务性跳过（非交易日/非时段/已调仓）：算已处理，不重试
    logger.info("[auto-open] 跳过（%s）", exc.detail)
    handled = True
except Exception as exc:              # 真异常：**不写 marker**，grace 窗口内重启可重试
    logger.error("[auto-open] 执行异常（不写 marker，grace 窗口内重启可重试）: %s",
                 exc, exc_info=True)
if handled:
    marker.write_text(today.isoformat(), encoding="utf-8")
```

---

## 4. 改动

`api/main.py`：

1. 三处 `threading.Thread(...).start()` 从**定义处**移到**文件末尾**统一启动（新增「后台线程统一启动」段）。
   保证任何 worker 触发时模块已全部执行完、所有函数已绑定。
2. `_auto_open_execute_worker` 的 marker 写入加 `handled` 条件。

**没有改动**任何业务逻辑、阈值、配置；`portfolio_apply` 本身一行未动。

> 顺带修掉一个隐性好处：**`import api.main` 不再有副作用**（以前一 import 就会拉起三个交易相关线程，
> 任何工具脚本/测试只要 import 就会启动它们）。

---

## 5. 当日的补跑

marker 被检查动作误写、且补跑窗口（09:56）有限，处理方式：

1. 预置 marker，使服务启动时 worker 主动跳过（避开与浏览器轮询并发）；
2. 用**独立进程**调用 `portfolio_apply(force_open_ref=True)`（无并发 akshare 调用）；
3. 再启动服务。

结果（`portfolio_apply` 返回）：

```
大盘趋势闸门：大盘 4510.15 跌破 20 日均线 4593.13（低 1.81%）→ 当日不开新仓
被挡下的买入：002709 天赐材料、000737 北方铜业
大盘弱势（沪深300 -0.59%，阈值 -1.0%）→ 买入仓位降至 50%
账户：cash 75913.0 / equity 98780.52 / 累计 -1.22%
```

**当日无任何成交是正确结果**：持仓仍在 topN 内无需卖出，新买入被 2026-09-11 上线的**大盘趋势闸门**拦下。
因此 marker = `2026-09-14` 名副其实。

---

## 6. 排查中发现，但**本次未处理**的问题

### 6.1 `py_mini_racer` V8 段错误（重要，未解决）

第一次尝试直接用正常路径起服务时，进程**原生崩溃**：

```
INFO:     Application startup complete.
[FATAL:partition_address_space.cc(243)] Check failed: !IsConfigurablePoolInitialized().
  ...
  #0..#9  py_mini_racer\mini_racer.pyd
  #10..#15 _ctypes.pyd / libffi-8.dll
```

`py_mini_racer` 在**同一进程内并发创建多个 V8 实例**时会崩在 V8 的 partition 地址空间初始化上。
akshare 的新浪行情/日线接口每次调用都会新建一个 `MiniRacer`，所以**只要有两条并发路径同时打新浪**就会触发。

memory 里记过同源事故（`26_refresh_largepool.py --workers 4` 在低内存下秒崩）。本次疑似是
「auto-open worker 拉日线」与「浏览器轮询 `/api/realtime`」并发。

**规避是有效的**：改用独立进程补跑、并让 worker 跳过之后，服务至今 0 次 FATAL。
**但根因未修** —— 建议后续给 akshare 的 sina 调用加**全局串行锁**，或换用不需要 V8 的数据源。
在修好之前，**不要在服务刚启动、模型还在预热时同时打开看板大量轮询**。

### 6.2 `启动量化系统.bat` 的日志重定向无效

`start ... > logs\api.log 2>&1` 的重定向作用于 `start` 而非子进程，导致：

- `logs/api.log` 会是 0 字节（不能用来判断服务是否起来）；
- 真正的子进程输出只在那条**最小化的控制台窗口**里。

本次已改用 `Start-Process -RedirectStandardOutput/-RedirectStandardError`（memory 里记录的启动手法）。
建议把 bat 也改成同样的写法。

---

## 7. 复现 / 验证

```bash
PY=.venv/Scripts/python.exe

# 语法与线程位置
$PY -m py_compile api/main.py
grep -n "threading.Thread(target=.*daemon=True).start()" api/main.py   # 三处应在文件末尾

# 起服务（可抓到真实 stderr，替代 bat）
powershell -NoProfile -Command "Start-Process -FilePath '.venv\Scripts\python.exe' \
  -ArgumentList '-u','-m','uvicorn','api.main:app','--host','127.0.0.1','--port','8001' \
  -WindowStyle Minimized -WorkingDirectory '<repo>' \
  -RedirectStandardOutput 'logs\api.log' -RedirectStandardError 'logs\api.err.log'"

# 健康检查
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8001/api/stocks   # 期望 200
grep -c FATAL logs/api.err.log                                             # 期望 0
cat logs/auto_open_execute_date                                            # 期望当日日期
```

**回归**：全量测试文件通过（见提交信息）。

---

## 8. 教训

1. **daemon 线程不要在其定义处启动** —— 模块 import 是自上而下的，定义处启动 = 与文件后半部分赛跑。
   要么放文件末尾，要么改用 `@app.on_event("startup")`。
2. **防重复的 marker 必须在「确认处理过」之后才写** —— 这条已经是第 3 次踩（影子流水线 → auto-open）。
   写 marker 前先问：*失败路径会不会也走到这里？*
3. **`start ... > log` 在 bat 里是无效重定向** —— 别用它判断服务状态。
4. **检查动作本身可能有副作用** —— 本次 `import api.main` 就直接触发了调仓 worker 并写下 marker。
   排查「导入即启动线程」的模块时，先读代码确认副作用，或加 `--dry-run` 开关。
