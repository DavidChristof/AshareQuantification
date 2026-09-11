"""账户回撤熔断单元测试（quant/risk/drawdown.py）。

纯函数：无网络、无数据库。

运行：python tests/test_drawdown.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.risk.drawdown import (                                   # noqa: E402
    drawdown_pct, evaluate,
)


def test_drawdown_pct_basic():
    assert drawdown_pct([100, 105, 110], 60) == 0.0          # 在涨 → 无回撤
    assert abs(drawdown_pct([100, 120, 108], 60) - 10.0) < 1e-9   # 120 → 108 = -10%
    assert drawdown_pct([], 60) == 0.0
    assert drawdown_pct([0, 0], 60) == 0.0                    # 非法净值被忽略


def test_drawdown_window_uses_recent_peak():
    # 很久以前的高点不进入 3 日窗口 → 按近 3 点算
    curve = [200, 100, 100, 95, 90]
    assert abs(drawdown_pct(curve, window_days=3) - 10.0) < 1e-9   # 近3点峰值100 → 90
    assert abs(drawdown_pct(curve, window_days=99) - 55.0) < 1e-9  # 全历史峰值200 → 90


def test_accepts_equity_history_shape():
    hist = [{"date": "2026-09-08", "equity": 100.0},
            {"date": "2026-09-09", "equity": 95.0},
            {"date": "2026-09-10", "equity": 91.0}]
    assert abs(drawdown_pct(hist, 60) - 9.0) < 1e-9


def test_trip_threshold():
    flat = [100, 100, 100]
    assert evaluate(flat, 8.0, 4.0, 60).tripped is False
    dd7 = [100, 100, 93.0]                                    # -7% < 8% → 不触发
    assert evaluate(dd7, 8.0, 4.0, 60).tripped is False
    dd9 = [100, 100, 91.0]                                    # -9% ≥ 8% → 触发
    st = evaluate(dd9, 8.0, 4.0, 60)
    assert st.tripped is True and st.dd_pct >= 8.0
    assert "熔断" in st.reason


def test_hysteresis_keeps_tripped_in_band():
    """滞回：已触发时，回撤收窄到 [release, trip) 区间内仍保持触发。"""
    band = [100, 100, 94.0]                                   # -6%：在 [4%,8%) 区间
    assert evaluate(band, 8.0, 4.0, 60, tripped_before=False).tripped is False  # 未触发→不触发
    assert evaluate(band, 8.0, 4.0, 60, tripped_before=True).tripped is True    # 已触发→维持
    recovered = [100, 100, 97.0]                              # -3% < 4% → 解除
    st = evaluate(recovered, 8.0, 4.0, 60, tripped_before=True)
    assert st.tripped is False and "解除" in st.reason


def test_evaluate_handles_short_and_bad_input():
    st = evaluate([], 8.0, 4.0, 60)
    assert st.tripped is False and st.equity == 0.0
    st2 = evaluate([None, "x", 100.0], 8.0, 4.0, 60)
    assert st2.equity == 100.0 and st2.tripped is False


def test_to_dict_rounds():
    d = evaluate([100, 91.0], 8.0, 4.0, 60).to_dict()
    assert set(d) == {"equity", "peak", "dd_pct", "window_days", "tripped", "reason"}
    assert d["dd_pct"] == 9.0 and d["tripped"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
