"""大盘趋势闸门单元测试（quant/risk/market_trend.py）。纯函数：无网络。

运行：python tests/test_market_trend.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.risk.market_trend import completed_closes, trend_gate      # noqa: E402


def test_below_ma_triggers():
    # 20 个 100，随后跌到 90 → 跌破 MA20
    g = trend_gate([100.0] * 19 + [90.0], ma_days=20)
    assert g.below is True
    assert "跌破" in g.reason and "不开新仓" in g.reason
    assert g.ma == (100.0 * 19 + 90.0) / 20
    assert g.level == 90.0


def test_above_ma_passes():
    g = trend_gate([100.0] * 19 + [110.0], ma_days=20)
    assert g.below is False
    assert "上方" in g.reason


def test_exactly_equal_is_not_below():
    g = trend_gate([100.0] * 20, ma_days=20)
    assert g.below is False


def test_insufficient_data_does_not_intervene():
    g = trend_gate([100.0] * 5, ma_days=20)
    assert g.below is False and g.ma == 0.0
    assert "数据不足" in g.reason
    assert trend_gate([], ma_days=20).below is False


def test_ignores_bad_values():
    seq = [float("nan"), None, "abc"] + [100.0] * 19 + [95.0]
    g = trend_gate(seq, ma_days=20)
    assert g.level == 95.0 and g.below is True


def test_to_dict_shape():
    d = trend_gate([100.0] * 20, ma_days=20).to_dict()
    assert set(d) == {"level", "ma", "ma_days", "below", "reason"}


# ---------- completed_closes：闸门对「今天」的口径必须唯一 ----------
def test_same_day_is_identical_before_and_after_close():
    """**核心回归**：同一交易日，盘中与收盘后必须得到同一闸门结论。

    日线接口盘中不含当天、收盘后含当天。旧实现直接取 `closes[-1]`，于是：
        盘中  → 用 D-1 的收盘
        收盘后 → 用 D   的收盘
    同一天两个结论（实测 12.9% 的交易日相反）。这里模拟"盘中"与"收盘后"两份数据，
    断言过滤后完全一致。
    """
    D = "2026-09-14"
    dts = ["2026-09-10", "2026-09-11"]
    cls = [5000.0, 4500.0]
    intraday = completed_closes(dts, cls, D)                        # 盘中：还没有今天
    after = completed_closes(dts + [D], cls + [4300.0], D)          # 收盘后：今天出现了
    assert intraday == after == [5000.0, 4500.0], (intraday, after)
    # 结论也必须一致（用一个会被"今天大跌"翻转的构造）
    assert trend_gate(intraday, 20).below == trend_gate(after, 20).below


def test_advances_only_after_the_day_passes():
    """到了次日，才把 D 的收盘纳入 —— 这是唯一正确的"前进一根"。"""
    D, D1 = "2026-09-14", "2026-09-15"
    dts = ["2026-09-10", "2026-09-11", D]
    cls = [5000.0, 4500.0, 4300.0]
    assert completed_closes(dts, cls, D1) == [5000.0, 4500.0, 4300.0]
    assert completed_closes(dts, cls, D) == [5000.0, 4500.0]


def test_no_lookahead_drops_future_dates():
    """`today` 之后的数据（未来）同样剔除 —— 结构上不可能有未来函数。"""
    dts = ["2026-09-10", "2026-09-11", "2026-09-14", "2026-09-15"]
    cls = [5000.0, 4500.0, 4300.0, 4200.0]
    assert completed_closes(dts, cls, "2026-09-14") == [5000.0, 4500.0]


def test_accepts_timestamp_like_dates():
    """pandas Timestamp 也要能正确比较（str(Timestamp)[:10] 就是日期部分）。"""
    class _Ts:
        def __init__(self, s):
            self.s = s

        def __str__(self):
            return self.s + " 00:00:00"

    dts = [_Ts("2026-09-11"), _Ts("2026-09-14")]
    assert completed_closes(dts, [4500.0, 4300.0], "2026-09-14") == [4500.0]


def test_drops_bad_values_and_handles_empty():
    dts = ["2026-09-10", "2026-09-11", "2026-09-14"]
    cls = [float("nan"), 4500.0, 4300.0]
    assert completed_closes(dts, cls, "2026-09-14") == [4500.0]
    assert completed_closes([], [], "2026-09-14") == []
    assert completed_closes(["2026-09-14"], [4300.0], "2026-09-14") == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
