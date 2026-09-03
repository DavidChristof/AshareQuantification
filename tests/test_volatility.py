"""按波动率动态止盈止损 单元测试。

运行：python -m pytest tests/test_volatility.py -v  或  python tests/test_volatility.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quant.features.technical import compute_technical
from quant.risk.volatility import (
    build_vol_map, dynamic_pcts, latest_atr, vol_cfg_from_risk,
)
from quant.trading.paper import PaperBroker


def _bars(close_seq, high_seq=None, low_seq=None, start="2026-01-01"):
    n = len(close_seq)
    high_seq = high_seq or [c * 1.02 for c in close_seq]
    low_seq = low_seq or [c * 0.98 for c in close_seq]
    return pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D"),
        "open": close_seq, "close": close_seq,
        "high": high_seq, "low": low_seq,
        "volume": [1e6] * n,
    })


def test_add_atr_in_technical():
    close = [100.0 + (i % 5) for i in range(30)]
    feat = compute_technical(_bars(close), ["atr"])
    atr_last = feat["atr"].iloc[-1]
    assert np.isfinite(atr_last) and atr_last > 0
    # 波动范围约 4 元（2% 高低），ATR 应落在合理区间
    assert 1.0 < atr_last < 10.0


def test_build_vol_map():
    data = {
        "HV": _bars([100.0 + (i % 5) for i in range(30)]),
        "LV": _bars([100.0 + 0.1 * (i % 5) for i in range(30)]),
    }
    vm = build_vol_map(data, window=20)
    assert "HV" in vm and "LV" in vm
    assert vm["HV"]["atr"] > vm["LV"]["atr"]          # 高波动 ATR 更大
    assert vm["HV"]["close"] > 0 and vm["HV"]["atr_pct"] > 0


def test_dynamic_pcts_high_vol_wider():
    """高波动股止损/止盈百分比更大（线更宽）。"""
    cfg = dict(stop_mult=2.5, take_mult=3.5, trail_mult=2.5,
               min_pct=0.02, max_pct=0.15, take_min_pct=0.04, take_max_pct=0.30)
    dh = dynamic_pcts({"atr": 6.0}, 100.0, **cfg)     # ATR 6% 
    dl = dynamic_pcts({"atr": 1.0}, 100.0, **cfg)     # ATR 1%
    assert dh["stop_pct"] > dl["stop_pct"]
    assert dh["take_pct"] > dl["take_pct"]
    assert dh["stop_price"] < dl["stop_price"]
    assert dh["take_price"] > dl["take_price"]


def test_dynamic_pcts_clamped_and_ordered():
    """极值被 clamp，且止盈始终 > 止损。"""
    cfg = dict(stop_mult=2.5, take_mult=3.5, trail_mult=2.5,
               min_pct=0.03, max_pct=0.15, take_min_pct=0.05, take_max_pct=0.30)
    # 极端高波动 → 止损 clamp 到 0.15
    d = dynamic_pcts({"atr": 100.0}, 100.0, **cfg)
    assert d["stop_pct"] == 0.15
    assert d["take_pct"] <= 0.30 and d["take_pct"] > d["stop_pct"]
    # 极端低波动 → 止损 clamp 到 0.03
    d2 = dynamic_pcts({"atr": 0.001}, 100.0, **cfg)
    assert d2["stop_pct"] == 0.03
    assert d2["take_pct"] == 0.05
    # 无有效波动率 → None
    d3 = dynamic_pcts({"atr": 0.0}, 100.0, **cfg)
    assert d3["stop_pct"] is None


def _broker(tmp_name):
    return PaperBroker(tmp_name, initial_capital=100_000.0)


def _rm(tmp):
    try:
        os.remove(tmp)
    except OSError:
        pass


import itertools
_TMP_UID = itertools.count()


def _tmp_db():
    """为每个测试生成唯一 db 文件名（避免 id(object()) 复用导致测试间文件冲突）。"""
    return f"paper/_test_vol_{os.getpid()}_{next(_TMP_UID)}.db"


def test_apply_stop_rules_dynamic():
    """高波动股止损更宽（94 不触发）；低波动股止损更窄（94 触发）。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("HV", 100, 100.0, "2026-08-01")
        b.buy("LV", 100, 100.0, "2026-08-01")
        # HV: ATR12→stop_pct=clamp(0.30)=0.15→止损位85 → 现价94不触发
        # LV: ATR1 →stop_pct=clamp(0.025)=0.03→止损位97 → 现价94触发
        vol = {
            "HV": {"atr": 12.0, "close": 100.0, "atr_pct": 0.12},
            "LV": {"atr": 1.0, "close": 100.0, "atr_pct": 0.01},
        }
        vol_cfg = dict(atr_stop_mult=2.5, atr_take_mult=3.5, atr_trailing_mult=2.5,
                       vol_min_pct=0.03, vol_max_pct=0.15,
                       take_min_pct=0.05, take_max_pct=0.30, trailing_enabled=True)
        triggered = b.apply_stop_rules("2026-08-10", {"HV": 94.0, "LV": 94.0},
                                       vol=vol, vol_cfg=vol_cfg)
        symbols = {t["symbol"] for t in triggered}
        assert "LV" in symbols and "HV" not in symbols
    finally:
        _rm(tmp)


def test_apply_stop_rules_fixed_fallback():
    """无 vol → 回退固定百分比（8% 止损）。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("X", 100, 100.0, "2026-08-01")   # 成本≈100.02
        assert b.apply_stop_rules("2026-08-10", {"X": 94.0}) == []   # 94 > 92.02
        tr = b.apply_stop_rules("2026-08-10", {"X": 90.0})           # 90 ≤ 92.02
        assert tr and tr[0]["symbol"] == "X" and "止损" in tr[0]["reason"]
    finally:
        _rm(tmp)


def test_dynamic_trailing_uses_high():
    """动态移动止损基于持仓最高价：从高点回撤 ATR 倍数。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("T", 100, 100.0, "2026-08-01")     # 成本≈100.02, max_price=100.02
        vol = {"T": {"atr": 4.0, "close": 100.0, "atr_pct": 0.04}}   # 4%
        vol_cfg = dict(atr_stop_mult=2.5, atr_take_mult=3.5, atr_trailing_mult=2.5,
                       vol_min_pct=0.03, vol_max_pct=0.15,
                       take_min_pct=0.05, take_max_pct=0.30, trailing_enabled=True)
        # 先涨到 112（更新 max_price），再从高点回撤 2.5×4%=10% → ≤100.8 触发移动止损
        b.apply_stop_rules("2026-08-10", {"T": 112.0}, vol=vol, vol_cfg=vol_cfg)
        tr = b.apply_stop_rules("2026-08-11", {"T": 100.0}, vol=vol, vol_cfg=vol_cfg)
        assert tr and "移动止损" in tr[0]["reason"]
    finally:
        _rm(tmp)


if __name__ == "__main__":
    tests = [test_add_atr_in_technical, test_build_vol_map,
             test_dynamic_pcts_high_vol_wider, test_dynamic_pcts_clamped_and_ordered,
             test_apply_stop_rules_dynamic, test_apply_stop_rules_fixed_fallback,
             test_dynamic_trailing_uses_high]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
