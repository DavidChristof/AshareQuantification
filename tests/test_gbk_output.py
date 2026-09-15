"""控制台输出编码 单元测试：仓库里的 .py 必须能被 **GBK** 编码。

运行：python -m pytest tests/test_gbk_output.py -v  或  python tests/test_gbk_output.py

## 为什么需要这条测试

本机（中文 Windows）控制台是 **GBK**。非 GBK 字符只要落在**会被执行到的** print 里，
就会 `UnicodeEncodeError` 把整个脚本打死。而且它的发作方式是
「**平时全绿、关键时刻翻车**」——只有走到那个分支才炸：

    2026-09-15 实例：scripts/28_shadow_check.py 的判定表用了 checkmark/ballot-X
    （码位 U+2713 / U+2717），而那张表**挡在 STABLE/NOT_YET 判定之前**。
    前向 OOS 一直是 0 天，脚本在更早分支就 return 了；攒够第 1 个可评日的那天
    才第一次走到、也才第一次崩 => **结论永远打不出来**。
    见 docs/2026-09-15-gbk-console-output.md。

所以「文件能被 GBK 编码」是一条**便宜且无假阴性**的不变量：
不需要去判断「这个字符串到底会不会被打印」（那件事无法可靠判定，28 就是这么漏的）。

## 豁免

`api/main.py` 里保留了 4 个符号（码位 U+1F7E2 / U+23F8 / U+1F6D1 / U+21E9），
它们是**给浏览器渲染的 UI 文案**（JSON 的 text 字段），不是控制台输出；
换成 ASCII 只会让看板变难看。它由
`sys.stdout/stderr.reconfigure(errors="replace")` 兜底，故在此豁免。
**豁免名单是显式的** —— 加文件必须是刻意的，见 test_exempt_list_is_deliberate。
（本文件刻意只写码位、不写那些符号本身，否则它自己就通不过本测试。）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]

# 显式豁免：只允许这些文件含非 GBK 字符，且理由必须写在上面
EXEMPT = {"api/main.py"}

_SKIP_DIRS = {".venv", "__pycache__", ".git", "node_modules"}


def _python_files():
    for p in ROOT.rglob("*.py"):
        if _SKIP_DIRS & set(p.parts):
            continue
        if p.name.startswith("_"):        # 临时工具脚本不纳入
            continue
        yield p


def _first_bad_char(src: str):
    try:
        src.encode("gbk")
        return None
    except UnicodeEncodeError as exc:
        return exc.object[exc.start:exc.end]


def test_all_python_files_are_gbk_encodable():
    """除了显式豁免，所有 .py 都必须能被 GBK 编码。"""
    offenders = []
    for p in _python_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel in EXEMPT:
            continue
        bad = _first_bad_char(p.read_text(encoding="utf-8"))
        if bad:
            offenders.append(f"{rel}: {bad!r} (U+{ord(bad[0]):04X})")
    assert not offenders, (
        "这些文件含 GBK 编不出的字符，落到 print 里会崩掉整个脚本：\n  "
        + "\n  ".join(offenders)
        + "\n\n改成 ASCII 等价物（如 => - ^2 [!] OK X），或加进 EXEMPT 并写明理由。"
    )


def test_scripts_and_tests_are_strictly_clean():
    """控制台程序（scripts/ tests/）**一个豁免都没有** —— 它们的输出直接落 GBK 控制台。"""
    offenders = []
    for p in _python_files():
        rel = p.relative_to(ROOT).as_posix()
        if not rel.startswith(("scripts/", "tests/")):
            continue
        bad = _first_bad_char(p.read_text(encoding="utf-8"))
        if bad:
            offenders.append(f"{rel}: U+{ord(bad[0]):04X}")
    assert not offenders, "scripts/ 与 tests/ 不允许有任何非 GBK 字符：\n  " + "\n  ".join(offenders)


def test_exempt_list_is_deliberate():
    """豁免只能是「真的还含非 GBK 字符」的文件；已清干净的要及时移出。

    防的是「豁免名单变成垃圾抽屉」——它一旦只增不减，这条测试就废了。
    """
    still_bad = set()
    for rel in EXEMPT:
        p = ROOT / rel
        if p.exists() and _first_bad_char(p.read_text(encoding="utf-8")):
            still_bad.add(rel)
    stale = EXEMPT - still_bad
    assert not stale, (
        f"这些文件已不含非 GBK 字符，应从 EXEMPT 移除：{sorted(stale)}\n"
        "（豁免名单只该留真正需要的）"
    )


def test_the_crash_that_motivated_this_is_fixed():
    """回归：scripts/28 的判定表不得再用 checkmark/ballot-X（码位 U+2713/U+2717）。"""
    src = (ROOT / "scripts" / "28_shadow_check.py").read_text(encoding="utf-8")
    assert not any(ord(c) in (0x2713, 0x2717) for c in src), \
        "28 的判定表又用回了 GBK 编不出的符号 -> STABLE/NOT_YET 会再次打不出来"
    assert '{\'OK\' if v else \'--\'}' in src, "28 的判定表应当用 ASCII 的 OK/--"


if __name__ == "__main__":
    tests = [test_all_python_files_are_gbk_encodable,
             test_scripts_and_tests_are_strictly_clean,
             test_exempt_list_is_deliberate,
             test_the_crash_that_motivated_this_is_fixed]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
