"""每日选股 单元测试：技术面打分 + 技术面计算（网络 mock 掉）。

运行：python -m pytest tests/test_selection.py -v  或  python tests/test_selection.py
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from quant.data.selector import (                               # noqa: E402
    DEFAULT_TECH_WEIGHTS, TECH_REGIME_WEIGHTS, _daily_tech, _score_tech,
    pick_tech_weights,
)


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


def test_pick_tech_weights_regime_direction():
    """regime 门控方向：下跌市反转/低波权重最高，上涨市动量/趋势权重最高。"""
    down = pick_tech_weights("downtrend")
    up = pick_tech_weights("uptrend")
    rng = pick_tech_weights("range")
    assert down["rev"] > down["mom"]                 # 防守：反转 > 动量
    assert up["mom"] > up["rev"]                     # 追涨：动量 > 反转
    assert up["mom"] > rng["mom"] > down["mom"]     # 动量权重 单调随市况
    assert down["vol"] >= up["vol"]                  # 弱市低波权重大于强市


def test_pick_tech_weights_default_and_sum():
    """未知 regime → 默认权重；所有 regime 权重分量和为 100。"""
    assert pick_tech_weights(None) == DEFAULT_TECH_WEIGHTS
    assert pick_tech_weights("unknown") == DEFAULT_TECH_WEIGHTS
    for regime, w in TECH_REGIME_WEIGHTS.items():
        assert abs(sum(pick_tech_weights(regime).values()) - 100.0) < 1e-6


def test_score_tech_regime_weights_change_ranking():
    """同一股票在不同 regime 权重下得分方向合理：
    弱动量高反转低波(超跌防御票) → 下跌市权重下更高；强动量 → 上涨市权重下更高。"""
    defensive = {"mom20": -0.05, "trend": -0.02, "vol20": 0.015, "rev60": 0.15}
    momentum = {"mom20": 0.15, "trend": 0.04, "vol20": 0.025, "rev60": -0.05}
    s_def_down = _score_tech(defensive, pick_tech_weights("downtrend"))
    s_def_up = _score_tech(defensive, pick_tech_weights("uptrend"))
    assert s_def_down is not None and s_def_up is not None
    assert s_def_down > s_def_up + 5          # 防御票在弱市权重下明显更受青睐
    s_mom_up = _score_tech(momentum, pick_tech_weights("uptrend"))
    s_mom_down = _score_tech(momentum, pick_tech_weights("downtrend"))
    assert s_mom_up is not None and s_mom_down is not None
    assert s_mom_up > s_mom_down + 5          # 动量票在强市权重下明显更受青睐
    # 向后兼容：无 weights → 默认权重，旧行为不变
    assert _score_tech(defensive) is not None
    assert _score_tech(None) is None


if __name__ == "__main__":
    tests = [test_score_tech_strong_weak, test_score_tech_bounds,
             test_score_tech_none, test_daily_tech_uptrend,
             test_daily_tech_failure,
             test_pick_tech_weights_regime_direction,
             test_pick_tech_weights_default_and_sum,
             test_score_tech_regime_weights_change_ranking]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
