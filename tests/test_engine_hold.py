"""TradingEngine「多日持有·双阈值」单元测试。

运行：python tests/test_engine_hold.py   （或 pytest -v）

三种语义：
  - 掉出核心榜但概率仍 ≥ 清仓线 → 减持到观察仓（不清仓）
  - 概率跌破清仓线 → 全仓清出
  - classic 模式（clear_threshold=None）保持「掉榜即清」旧行为
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.engine import TradingEngine     # noqa: E402
from quant.trading.paper import PaperBroker        # noqa: E402

TH = 0.55       # 买入/核心阈值
CLEAR = 0.45    # 清仓线
POS_PCT = 0.95
MAX_POS = 2


def _make_broker(tmp_dir):
    """零佣金/零滑点纸面账户，便于精确断言。"""
    return PaperBroker(
        str(Path(tmp_dir) / "test.db"),
        initial_capital=100_000.0, commission=0.0, slippage=0.0,
    )


def _positions(broker):
    return {p.symbol: p.shares for p in broker.query_positions()}


def _engine(broker, clear_threshold=CLEAR):
    return TradingEngine(broker, threshold=TH, position_pct=POS_PCT,
                         max_positions=MAX_POS, clear_threshold=clear_threshold,
                         hold_trim_pct=0.5)


def test_out_of_core_trim_not_sold():
    """B 掉出核心榜但概率 0.50 ≥ 0.45 → 只减到半仓观察，不清仓；C 新进照买。"""
    tmp = tempfile.mkdtemp()
    try:
        b = _make_broker(tmp)
        eng = _engine(b)
        # day1：核心 A、B（概率最高两只），等权满仓
        eng.rebalance("2026-09-01",
                      {"A": 0.60, "B": 0.58},
                      {"A": 100.0, "B": 50.0})
        day1 = _positions(b)
        assert day1.get("A", 0) > 0 and day1.get("B", 0) > 0
        b_shares_day1 = day1["B"]

        # day2：A 仍核心；C 以 0.56 挤进核心；B 掉出但未破清仓线
        eng.rebalance("2026-09-02",
                      {"A": 0.60, "C": 0.56, "B": 0.50},
                      {"A": 100.0, "C": 40.0, "B": 50.0})
        p2 = _positions(b)
        # B 保留（未清仓），且被减持约一半
        assert p2.get("B", 0) > 0, "掉出核心但未破线不应清仓"
        assert p2["B"] < b_shares_day1 * 0.9, "B 应被减持到半仓观察"
        # A 保留、C 新进
        assert p2.get("A", 0) == day1["A"]
        assert p2.get("C", 0) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_out_of_core_trim_not_sold")


def test_below_clear_full_exit():
    """B 概率跌破清仓线 0.45 → 全部卖出。"""
    tmp = tempfile.mkdtemp()
    try:
        b = _make_broker(tmp)
        eng = _engine(b)
        eng.rebalance("2026-09-01",
                      {"A": 0.60, "B": 0.58},
                      {"A": 100.0, "B": 50.0})
        eng.rebalance("2026-09-02",
                      {"A": 0.60, "B": 0.40},
                      {"A": 100.0, "B": 50.0})
        p2 = _positions(b)
        assert p2.get("B", 0) == 0, "跌破清仓线应清仓"
        assert p2.get("A", 0) > 0, "仍为核心应保留/补仓"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_below_clear_full_exit")


def test_classic_mode_unchanged():
    """clear_threshold=None → 旧语义：掉出核心榜即使概率尚可也全清。"""
    tmp = tempfile.mkdtemp()
    try:
        b = _make_broker(tmp)
        eng = _engine(b, clear_threshold=None)
        eng.rebalance("2026-09-01",
                      {"A": 0.60, "B": 0.58},
                      {"A": 100.0, "B": 50.0})
        # B 掉出（0.50 < 0.55 阈值），但若在 dual 下会保留半仓——classic 应清掉
        eng.rebalance("2026-09-02",
                      {"A": 0.60, "C": 0.56, "B": 0.50},
                      {"A": 100.0, "C": 40.0, "B": 50.0})
        p2 = _positions(b)
        assert p2.get("B", 0) == 0, "classic 模式掉出核心榜应清仓"
        assert p2.get("A", 0) > 0 and p2.get("C", 0) > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS test_classic_mode_unchanged")


if __name__ == "__main__":
    tests = [test_out_of_core_trim_not_sold,
             test_below_clear_full_exit,
             test_classic_mode_unchanged]
    for fn in tests:
        fn()
    print("全部通过")
