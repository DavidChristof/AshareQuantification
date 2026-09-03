"""每日选股 单元测试：技术面打分 + 技术面计算（网络 mock 掉）。

运行：python -m pytest tests/test_selection.py -v  或  python tests/test_selection.py
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.data.selector import _daily_tech, _score_tech  # noqa: E402


def test_score_tech_strong_weak():
    """强动量+强趋势+低波动 → 高分；反之低分。"""
    strong = _score_tech({"mom20": 0.20, "trend": 0.05, "vol20": 0.01})
    weak = _score_tech({"mom20": -0.10, "trend": -0.08, "vol20": 0.05})
    assert strong is not None and weak is not None
    assert strong > 70
    assert weak < 30
    assert strong > weak


def test_score_tech_bounds():
    """打分离散在 0~100（含极端输入不越界）。"""
    s = _score_tech({"mom20": 9.9, "trend": 9.9, "vol20": 0.0})
    assert 0 <= s <= 100
    s2 = _score_tech({"mom20": -9.9, "trend": -9.9, "vol20": 9.9})
    assert 0 <= s2 <= 100


def test_score_tech_none():
    """无技术数据 → None（调用方回退基本面分）。"""
    assert _score_tech(None) is None


def test_daily_tech_uptrend():
    """持续上涨的日线 → 动量>0、趋势>0、波动小。"""
    idx = pd.date_range("2026-01-01", periods=30, freq="B")
    close = np.linspace(100, 120, 30)
    df = pd.DataFrame({
        "date": idx.strftime("%Y-%m-%d"), "open": close, "high": close * 1.01,
        "low": close * 0.99, "close": close, "volume": np.full(30, 1e6),
    })
    with mock.patch("quant.data.selector.fetch_daily", return_value=df):
        t = _daily_tech("600000")
    assert t is not None
    assert t["mom20"] > 0.05
    assert t["trend"] > 0
    assert t["vol20"] < 0.01


def test_daily_tech_failure():
    """日线拉取失败/数据不足 → None（不崩溃，跳过该股）。"""
    with mock.patch("quant.data.selector.fetch_daily", side_effect=RuntimeError("net")):
        assert _daily_tech("600000") is None
    short = pd.DataFrame({"close": [1, 2, 3]})   # 不足 25 根
    with mock.patch("quant.data.selector.fetch_daily", return_value=short):
        assert _daily_tech("600000") is None


if __name__ == "__main__":
    tests = [test_score_tech_strong_weak, test_score_tech_bounds,
             test_score_tech_none, test_daily_tech_uptrend,
             test_daily_tech_failure]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
