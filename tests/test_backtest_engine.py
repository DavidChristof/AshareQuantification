"""回测增强 单元测试：ATR 动态止盈止损进回测 + 逐日滚动择时。

运行：python -m pytest tests/test_backtest_engine.py -v  或  python tests/test_backtest_engine.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quant.backtest.engine import BacktestEngine          # noqa: E402
from quant.risk.volatility import vol_cfg_from_risk       # noqa: E402
from quant.timing.roll import roll_regime, roll_timing_signal  # noqa: E402


def _sig_table(close_seq, signal=None, start="2026-01-01"):
    n = len(close_seq)
    idx = pd.date_range(start, periods=n, freq="B")
    sig = signal if signal is not None else np.ones(n, dtype=int)
    return pd.DataFrame({"close": close_seq, "signal": sig}, index=idx)


def _bars(close_seq, start="2026-01-01"):
    n = len(close_seq)
    idx = pd.date_range(start, periods=n, freq="B")
    return pd.DataFrame({
        "date": idx,
        "open": close_seq, "close": close_seq,
        "high": [c * 1.02 for c in close_seq],
        "low": [c * 0.98 for c in close_seq],
        "volume": [1e6] * n,
    }, index=idx)


def test_engine_dynamic_stop_triggers():
    """ATR 动态止损：价格跌破止损位时即使信号仍看多也卖出，之后不再追。"""
    close = [100.0] * 5 + [96.0, 90.0, 85.0, 80.0] * 3   # 先平后跌（17 天）
    signal = [1] * 6 + [0] * 11                            # 第 6 天触发止损，之后空仓信号
    sig = _sig_table(close, signal=signal)
    n = len(close)
    atr = pd.Series(np.full(n, 1.0), index=sig.index)      # ATR 恒定 1.0 → 止损 3%（clamp 下限）
    risk_cfg = vol_cfg_from_risk({})
    engine = BacktestEngine(initial_capital=100_000.0)

    base = engine.run(sig)
    stop = engine.run(sig, risk_cfg=risk_cfg, atr=atr)

    # stop 策略触发了止损（比 base 更早、更高价位离场）
    assert stop["stop_reason"].notna().any()
    # 止损后空仓（signal=0 不再买回）
    assert stop["holdings"].iloc[-1] == 0
    # 止损策略在深跌行情中净值更高（保护效果：止损在第 6 天 96 离场，base 第 7 天 90 离场）
    assert stop["equity"].iloc[-1] > base["equity"].iloc[-1]


def test_engine_no_risk_cfg_fallback():
    """不传 risk_cfg → 与原来一样（无止损，全程持有）。"""
    close = [100.0] * 5 + [80.0] * 10
    sig = _sig_table(close)
    engine = BacktestEngine()
    r = engine.run(sig)
    assert r["stop_reason"].isna().all()
    assert r["holdings"].iloc[-1] > 0     # 一直持有


def test_engine_take_profit_and_signal_exit():
    """止盈触发卖出；signal=0 也正常清仓。"""
    close = [100.0] * 5 + [115.0] * 5     # 涨到 115
    sig = _sig_table(close)
    n = len(close)
    atr = pd.Series(np.full(n, 1.0), index=sig.index)   # 止盈 3.5*1=3.5% → 止盈位 ~103.5
    engine = BacktestEngine()
    r = engine.run(sig, risk_cfg=vol_cfg_from_risk({}), atr=atr)
    assert r["stop_reason"].notna().any()
    assert "止盈" in str(r["stop_reason"].dropna().iloc[0])

    # signal=0 清仓
    sig2 = _sig_table(close, signal=[1] * 3 + [0] * 7)
    r2 = engine.run(sig2)
    assert r2["holdings"].iloc[-1] == 0


def test_engine_stamp_tax_on_sell():
    """回测卖出计入印花税：现金回收 = 收入 - 佣金 - 印花税。"""
    close = [100.0] * 10
    sig = _sig_table(close, signal=[1] * 5 + [0] * 5)
    engine = BacktestEngine(commission=0.0003, stamp_tax=0.0005)
    r = engine.run(sig)
    # 第 5 天持有份额
    shares = r["holdings"].iloc[4] / 100.0
    sell_price = 100 * (1 - 0.0002)                       # 滑点
    proceeds = shares * sell_price
    expected_cash_in = proceeds * (1 - 0.0003 - 0.0005)   # 佣金 + 印花税
    delta = r["cash"].iloc[5] - r["cash"].iloc[4]       # 卖出那天的现金增量
    assert abs(delta - expected_cash_in) < 1e-6


def test_roll_regime_updown():
    """上升序列 → 主要 uptrend；下降序列 → 主要 downtrend。"""
    up = pd.Series(100 * (1.005 ** np.arange(80)), index=pd.date_range("2026-01-01", periods=80, freq="B"))
    ru = roll_regime(up)
    assert (ru == "uptrend").mean() > 0.5

    down = pd.Series(100 * (0.995 ** np.arange(80)), index=pd.date_range("2026-01-01", periods=80, freq="B"))
    rd = roll_regime(down)
    assert (rd == "downtrend").mean() > 0.5


def test_roll_timing_signal_shape():
    """择时滚动信号输出形状正确、分值有界。"""
    n = 60
    close = 100 + np.sin(np.arange(n) / 5) * 3
    bars = _bars(list(close))
    prob = pd.Series(np.full(n, 0.6), index=bars.index)
    regime = pd.Series(["uptrend"] * n, index=bars.index)
    scores, actions = roll_timing_signal(bars, prob, regime)
    assert len(scores) == n and len(actions) == n
    assert scores.between(-1, 1).all()
    assert set(actions.unique()).issubset({"buy", "sell", "hold"})


if __name__ == "__main__":
    tests = [test_engine_dynamic_stop_triggers, test_engine_no_risk_cfg_fallback,
             test_engine_take_profit_and_signal_exit, test_engine_stamp_tax_on_sell,
             test_roll_regime_updown, test_roll_timing_signal_shape]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
