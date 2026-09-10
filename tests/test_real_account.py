"""实盘账本 RealBroker 单元测试（记账 / 回报手续费 / 未成交留痕 / 纠错重建）。

运行：python tests/test_real_account.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.real_account import RealBroker

_TMP_UID = [0]
REAL_KW = dict(initial_capital=3000.0, commission=0.0003, slippage=0.0,
               stamp_tax=0.0005, lot_size=100, min_commission=5.0,
               transfer_fee=0.00001)


def _tmp_db():
    _TMP_UID[0] += 1
    return f"paper/_test_real_{os.getpid()}_{_TMP_UID[0]}.db"


def _rm(tmp):
    try:
        os.remove(tmp)
    except OSError:
        pass


def _broker(tmp):
    return RealBroker(tmp, **REAL_KW)


def test_record_buy_uses_reported_fee():
    tmp = _tmp_db()
    b = _broker(tmp)
    r = b.record_execution("603993", "buy", 100, 18.00, "2026-09-10", fee=5.00)
    assert r.success and r.fee == 5.00
    assert abs(b.query_cash() - (3000.0 - 1800.0 - 5.0)) < 1e-6
    assert [(p.symbol, p.shares) for p in b.query_positions()] == [("603993", 100.0)]
    # 成交也进订单留痕（status=filled）
    orders = b.orders()
    assert len(orders) == 1 and orders[0]["status"] == "filled"
    _rm(tmp)


def test_unfilled_order_leaves_cash_untouched():
    tmp = _tmp_db()
    b = _broker(tmp)
    oid = b.log_order("300750", "buy", 100, 12.00, "2026-09-10", status="unfilled",
                      reason="涨停封板", advice_price=11.98, advice_status="hard")
    assert b.query_cash() == 3000.0 and b.query_positions() == []
    assert b.orders()[0]["id"] == oid and b.orders()[0]["status"] == "unfilled"
    assert b.orders()[0]["reason"] == "涨停封板"
    # 作废留痕不影响资金
    assert b.void_order(oid) is True
    assert b.orders()[0]["status"] == "void"
    assert b.query_cash() == 3000.0
    _rm(tmp)


def test_sell_with_reported_fee_and_t1():
    tmp = _tmp_db()
    b = _broker(tmp)
    b.record_execution("603993", "buy", 100, 18.00, "2026-09-09", fee=5.0)
    assert b.sellable_shares("603993", "2026-09-09") == 0.0      # 当日买入不可卖
    assert b.sellable_shares("603993", "2026-09-10") == 100.0
    r = b.sell("603993", 100, 18.50, "2026-09-10", fee_override=5.93)
    assert r.success and abs(r.fee - 5.93) < 1e-9
    assert b.query_positions() == []
    _rm(tmp)


def test_delete_trade_and_rebuild_is_self_consistent():
    tmp = _tmp_db()
    b = _broker(tmp)
    b.record_execution("603993", "buy", 100, 18.00, "2026-09-09", fee=5.0)
    b.record_execution("600160", "buy", 100, 10.00, "2026-09-10", fee=5.0)
    hist = b.trade_history_with_id()
    assert len(hist) == 2 and all("id" in h for h in hist)
    bad_id = hist[0]["id"]                                     # 最新那笔（600160）
    assert b.delete_trade(bad_id) is True
    out = b.rebuild_from_trades()
    assert out["positions"] == {"603993": 100.0}
    assert abs(b.query_cash() - (3000.0 - 1800.0 - 5.0)) < 1e-6
    # 重放结果必须与直接记账一致
    assert [p.symbol for p in b.query_positions()] == ["603993"]
    _rm(tmp)


def test_buy_enforces_lot_and_rejects_bad_price():
    tmp = _tmp_db()
    b = _broker(tmp)
    assert b.record_execution("603993", "buy", 150, 18.0, "2026-09-10").success is False
    assert b.record_execution("603993", "buy", 100, float("nan"), "2026-09-10").success is False
    assert b.query_cash() == 3000.0 and b.query_positions() == []
    _rm(tmp)


def test_insufficient_cash_rejected():
    tmp = _tmp_db()
    b = _broker(tmp)
    # 3 手 18 元 = 5400 > 3000
    r = b.record_execution("603993", "buy", 300, 18.0, "2026-09-10")
    assert r.success is False and "资金不足" in r.message
    assert b.query_cash() == 3000.0
    _rm(tmp)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
