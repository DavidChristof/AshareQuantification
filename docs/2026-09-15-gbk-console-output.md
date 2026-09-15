# `scripts/28` 的结论永远打不出来 —— GBK 控制台吞掉输出

**日期**：2026-09-15
**起因**：跑例行兜底 `26→27→28`，`28_shadow_check.py` 崩在 `UnicodeEncodeError`
**结论**：本机控制台是 **GBK**，而 `28` 用 `✓`/`✗` 打印判定表，这两个字符不在 GBK 里。
更糟的是——**这张表挡在 `STABLE`/`NOT_YET` 判定之前**，所以崩在这里 = 结论永远不输出。

---

## 1. 事故本身

```
[28] 累积：OOS 日子 6（其中已有收益 1）
[28] 前向 OOS（> 2026-09-07，1 天）：
      live40     平均RankIC=0.0204  ICIR=0.000
      shadow600  平均RankIC=0.0325  ICIR=0.000
Traceback (most recent call last):
  File "scripts/28_shadow_check.py", line 91, in main
    print(f"      [{('✓' if v else '✗')}] {k}", flush=True)
UnicodeEncodeError: 'gbk' codec can't encode character '✗'
```

**为什么以前没暴露**：这段检查表只有走到「有 OOS 可评日子」才会执行。
此前前向 OOS 一直是 0 天，`28` 在更早的分支就 `return` 了，**根本没走到这里**。
2026-09-15 首次攒够 1 个可评日 → 首次走到检查表 → 首次崩。

**后果**：兜底定时任务（以及任何调用 `28` 的地方）**看不到 `STABLE`/`NOT_YET`**。
讽刺的是，这个 bug 恰好把「该不该上线 600 池」的判定给屏蔽了——**最需要它说话的时候它闭嘴**。

**修法**：`✓/✗` → `OK/--`（纯 ASCII）。中文本身在 GBK 里，不需要动。

---

## 2. 这是系统性问题，不止一个文件

顺手全仓扫了一遍「字符串字面量里的非 GBK 字符」。判定方式：
AST 取出所有 `str` 常量（含 docstring），逐个试 `encode('gbk')`。

**25 个文件的字符串里存在非 GBK 字符**，涉及的主要字符：

| 字符 | 码位 | 出现处（部分） |
|---|---|---|
| `⇒` | U+21D2 | scripts/33/34/35/37/38/39/40、quant 多个模块 |
| `⚠` `️` | U+26A0 U+FE0F | scripts/38/40、quant/trading/*、api/main.py |
| `¥` | U+00A5 | scripts/09、quant/trading/fill.py、real_advice.py |
| `−` | U+2212 | quant/risk/drawdown.py、market_trend.py |
| `²` | U+00B2 | quant/factors/regression.py |
| `✓` | U+2713 | scripts/23 |
| `🛑` `🟢` `⏸` `⇓` | U+1F6D1 U+1F7E2 U+23F8 U+21E9 | api/main.py |

⚠️ **这是上界，不是确数**：扫描会把 docstring 也算进「字符串字面量」，
而 docstring 永远不会被 `print`。真正会炸的只是**被打印/记录的那部分字符串**，
且**只有走到那个分支才炸** —— 所以这是一个「平时全绿、关键时刻翻车」的雷区。

**没有本次就动手大扫除**，因为：25 个文件的改动面远大于本次任务的边界，
且需要逐个判断「这个字符到底会不会被打印」。**留给用户决定**。

### 建议的两个方向（择一即可）

1. **一行顺手护栏**：在每个入口 `sys.stdout.reconfigure(errors="replace")`。
   不可编码的字符退化成 `?`，**不崩**。改动小、覆盖全，缺点是不好看。
2. **扫干净**：把那几个符号换成 ASCII 等价物。彻底，但 25 个文件、要逐个核。

无论选哪个，都建议补一条**测试护栏**：遍历 `scripts/*.py` + `quant/**/*.py`，
对「会进入 print/logger 的字符串常量」断言 GBK 可编码。
本次这个 bug 用一条测试就能永久挡住。

---

## 3. 验证

```
$ .venv/Scripts/python.exe scripts/28_shadow_check.py     # 不设 PYTHONIOENCODING
      [--] OOS天数够
      [OK] 600有正IC
      [OK] 600>40(平均IC)
      [--] 600>40(ICIR)

NOT_YET：尚未稳压。预计还需 ≥9 个 OOS 可评日子；继续每天跑 26→27。
EXIT=1
```

**刻意不设 `PYTHONIOENCODING=utf-8` 复现**：正是这个环境变量会让 bug 藏起来
（它把 stdout 变成 UTF-8，`✗` 就能编码了）。**判定「脚本在真实控制台下能不能跑」时
不能加它**，否则等于把问题遮住。

（本次排查中我自己的诊断脚本也因为打印这些字符崩了两次——同一个坑。）

---

## 4. 教训

1. **「平时没报错」不等于「没问题」**：这条 `print` 一直在代码里，只是从没被执行到。
2. **结论性输出必须用最保守的字符集**。日志/判定表是给机器和定时任务读的，
   它们的编码环境往往比交互终端更差——**别用 ✓✗⚠⇒ 这些"好看"的符号**。
3. **不要用 `PYTHONIOENCODING=utf-8` 掩盖问题**：它让当前会话舒服了，
   却让真实控制台下的崩溃推迟到某个更关键的时刻才爆。

---

## 相关

- `docs/2026-09-15-indices-tencent-source.md` —— 同日的 V8 崩溃修复
- `scripts/28_shadow_check.py`
