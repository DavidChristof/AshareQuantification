"""A股交易规则 单元测试：T+1 / 整手买入 / 卖出印花税。

运行：python -m pytest tests/test_trading_rules.py -v  或  python tests/test_trading_rules.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker      # noqa: E402
from quant.trading.rules import limit_prices     # noqa: E402


def _broker(tmp_name, **kw):
    kw.setdefault("commission", 0.0003)
    kw.setdefault("slippage", 0.0002)
    return PaperBroker(tmp_name, initial_capital=100_000.0, **kw)


def _rm(tmp):
    try:
        os.remove(tmp)
    except OSError:
        pass


import itertools
_TMP_UID = itertools.count()


def _tmp_db():
    """为每个测试生成唯一 db 文件名（避免 id(object()) 复用导致测试间文件冲突）。"""
    return f"paper/_test_rules_{os.getpid()}_{next(_TMP_UID)}.db"


def test_t1_today_buy_cannot_sell():
    """T+1：今天买入的今天不能卖。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("X", 100, 100.0, "2026-08-31")          # 今天买
        r = b.sell("X", 100, 101.0, "2026-08-31")     # 今天卖 → 拒绝
        assert not r.success and "T+1" in r.message
        r2 = b.sell("X", 100, 101.0, "2026-09-01")    # 次日卖 → 成功
        assert r2.success
    finally:
        _rm(tmp)


def test_t1_partial_frozen_shares():
    """T+1：昨天买 100 + 今天买 100 = 200，今天最多卖 100（昨日份额）。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("X", 100, 100.0, "2026-08-28")
        b.buy("X", 100, 100.0, "2026-08-31")
        r = b.sell("X", 200, 101.0, "2026-08-31")    # 想卖 200 → 拒绝（今天只能卖 100）
        assert not r.success and "T+1" in r.message
        r2 = b.sell("X", 100, 101.0, "2026-08-31")   # 卖 100（昨日份额）→ 成功
        assert r2.success
        r3 = b.sell("X", 100, 101.0, "2026-08-31")   # 再卖 100（今日份额）→ 拒绝
        assert not r3.success and "T+1" in r3.message
    finally:
        _rm(tmp)


def test_lot_size_buy():
    """整手：lot_size=100 时买入必须 100 整数倍。"""
    tmp = _tmp_db()
    b = _broker(tmp, lot_size=100)
    try:
        r = b.buy("X", 150, 100.0, "2026-08-31")     # 非整手 → 拒绝
        assert not r.success and "整数倍" in r.message
        r2 = b.buy("X", 100, 100.0, "2026-08-31")    # 整手 → 成功
        assert r2.success
        # 卖出不强制整手（可卖零股）
        r3 = b.sell("X", 50, 101.0, "2026-09-01")
        assert r3.success
    finally:
        _rm(tmp)


def test_stamp_tax_on_sell():
    """印花税：卖出费用 = 佣金 + 印花税（单边）。"""
    tmp = _tmp_db()
    b = _broker(tmp, commission=0.0003, stamp_tax=0.0005)
    try:
        b.buy("X", 100, 100.0, "2026-08-01")
        r = b.sell("X", 100, 100.0, "2026-08-03")
        proceeds = 100 * 100 * (1 - 0.0002)          # 滑点
        expected_fee = proceeds * (0.0003 + 0.0005)  # 佣金+印花税
        assert r.success
        assert abs(r.fee - expected_fee) < 1e-6
        # 卖出费用高于纯佣金（含印花税）
        assert r.fee > proceeds * 0.0003 + 1e-9
    finally:
        _rm(tmp)


def test_stamp_tax_zero_when_not_configured():
    """默认（未配置印花税）卖出费用 = 佣金。"""
    tmp = _tmp_db()
    b = _broker(tmp, commission=0.0003, stamp_tax=0.0)
    try:
        b.buy("X", 100, 100.0, "2026-08-01")
        r = b.sell("X", 100, 100.0, "2026-08-03")
        proceeds = 100 * 100 * (1 - 0.0002)
        assert abs(r.fee - proceeds * 0.0003) < 1e-6
    finally:
        _rm(tmp)


def test_live_summary_uses_prices():
    """实时估值：总资产随传入价格同步，缺失价格回退成本。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("X", 100, 100.0, "2026-08-01")
        s_flat = b.live_summary({"X": 100.0})
        s_up = b.live_summary({"X": 120.0})     # 价格涨 20%
        print("DBG s_up:", s_up)
        assert s_up["equity"] > s_flat["equity"]
        # 市值 = 100 股 × 120
        assert abs(s_up["market_value"] - 100 * 120) < 1e-6
        # 缺失价格 → 回退成本价（≈100.02）
        s_no = b.live_summary({})
        assert abs(s_no["market_value"] - 100 * 100.02) < 5
        # 不写历史快照
        assert len(b.equity_history()) == 0
    finally:
        _rm(tmp)


def test_limit_prices_board():
    """涨跌停：主板 ±10%，创业板/科创板 ±20%。"""
    # 主板 600519：前收 100 → 涨停 110 / 跌停 90
    up, down = limit_prices(100.0, "600519")
    assert up == 110.0 and down == 90.0
    # 创业板 300750：前收 100 → 涨停 120 / 跌停 80
    up, down = limit_prices(100.0, "300750")
    assert up == 120.0 and down == 80.0
    # 科创板 688111
    up, down = limit_prices(100.0, "688111")
    assert up == 120.0 and down == 80.0
    # 深主板 000333
    up, down = limit_prices(100.0, "000333")
    assert up == 110.0 and down == 90.0


def test_limit_prices_rounding():
    """涨跌停价按 0.01 取整。"""
    up, down = limit_prices(13.21, "002532")
    assert up == round(13.21 * 1.10, 2) == 14.53
    assert down == round(13.21 * 0.90, 2) == 11.89


if __name__ == "__main__":
    tests = [test_t1_today_buy_cannot_sell, test_t1_partial_frozen_shares,
             test_lot_size_buy, test_stamp_tax_on_sell, test_stamp_tax_zero_when_not_configured,
             test_limit_prices_board, test_limit_prices_rounding,
             test_live_summary_uses_prices]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
