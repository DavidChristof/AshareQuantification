"""大盘趋势闸门单元测试（quant/risk/market_trend.py）。纯函数：无网络。

运行：python tests/test_market_trend.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.risk.market_trend import trend_gate                        # noqa: E402


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


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
