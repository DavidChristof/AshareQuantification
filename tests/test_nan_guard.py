"""NaN 价格防御单元测试（2026-09-09 事故回归）。

事故：自动纸面盘某日对 300750 下了一笔「价格 NaN」的买单——
`updater.rebalance_auto` 从信号表取 `close.iloc[-1]`，个别行为 NaN；
`engine` 里 `if not price:` 对 NaN 为 False（NaN 是 truthy），
`paper.buy` 里 `total_cost > cash` 对 NaN 也恒 False → 单子成交，
`cash - NaN = NaN` 被 SQLite 写成 NULL → `query_cash()` 的 `float(None)` 抛错
→ `/`、`/api/dashboard`、`/api/account` 全部 500。

修复：
- quant/trading/paper.py:_valid_price（非 None/非 NaN/inf 且 > 0）；buy/sell 拒无效价；
  query_cash / account_summary 容忍 NULL；snapshot_equity 跳过无效价。
- quant/trading/engine.py：所有 `if not price` 改 `if not _valid_price(price)`；市值只算有效价。
- quant/trading/updater.py:rebalance_auto：价格映射剔除无效收盘价。

运行：python tests/test_nan_guard.py
"""
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.engine import TradingEngine
from quant.trading.paper import PaperBroker, _valid_price

_TMP_UID = [0]
NAN = float("nan")


def _tmp_db():
    _TMP_UID[0] += 1
    return f"paper/_test_nan_{os.getpid()}_{_TMP_UID[0]}.db"


def _broker(tmp):
    # 自动纸面盘用的就是默认 lot_size=1（允许零股），与 api 里 BROKER 一致
    return PaperBroker(tmp, initial_capital=100_000.0)


def _rm(tmp):
    try:
        os.remove(tmp)
    except OSError:
        pass


def test_valid_price_helper():
    assert _valid_price(10.0) and _valid_price("3.5")
    assert not _valid_price(NAN)
    assert not _valid_price(float("inf"))
    assert not _valid_price(None)
    assert not _valid_price(0)
    assert not _valid_price(-1.0)
    assert not _valid_price("abc")


def test_buy_with_nan_price_rejected_cash_intact():
    """NaN 价的买单被拒 → 现金不变、无成交、无持仓（不再写 NULL）。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    r = b.buy("300750", 100, NAN, "2026-09-09")
    assert r.success is False
    assert b.query_cash() == 100_000.0            # 现金未被 NaN 污染
    assert b.query_positions() == []
    assert b.trade_history() == []
    _rm(tmp)


def test_buy_with_nan_shares_rejected():
    tmp = _tmp_db()
    b = _broker(tmp)
    assert b.buy("300750", NAN, 10.0, "2026-09-09").success is False
    assert b.query_cash() == 100_000.0
    _rm(tmp)


def test_engine_all_nan_prices_no_trade():
    """核心榜票价格 NaN：引擎跳过该票，不下 NaN 单、现金/总资产不变。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    eng = TradingEngine(b, threshold=0.55, position_pct=0.95, max_positions=2)
    out = eng.rebalance("2026-09-09", {"300750": 0.90}, {"300750": NAN})
    assert b.query_cash() == 100_000.0
    assert b.query_positions() == []
    assert math.isfinite(out["equity"]) and out["equity"] == 100_000.0
    _rm(tmp)


def test_engine_mixed_prices_only_valid_bought():
    """NaN 价票不买；有效价票正常买，且所有成交 shares/price 均有限（无 NaN 落库）。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    eng = TradingEngine(b, threshold=0.55, position_pct=0.95, max_positions=2)
    out = eng.rebalance("2026-09-09", {"300750": 0.90, "600519": 0.80},
                        {"300750": NAN, "600519": 1500.0})
    syms = [p.symbol for p in b.query_positions()]
    assert "300750" not in syms                   # NaN 价票没被买入
    assert "600519" in syms                       # 有效价票正常建仓
    assert math.isfinite(out["equity"])           # 总资产未被 NaN 污染
    for t in b.trade_history():
        assert t["shares"] is not None and t["price"] is not None
        assert math.isfinite(t["shares"]) and math.isfinite(t["price"])
    _rm(tmp)


def test_query_cash_and_summary_tolerate_null():
    """历史坏数据（cash=NULL）不再让 account_summary 抛错（接口 500 的直接原因）。"""
    import sqlite3
    tmp = _tmp_db()
    b = _broker(tmp)
    b.buy("600519", 100, 10.0, "2026-09-08")
    with sqlite3.connect(tmp) as c:
        c.execute("UPDATE paper_account SET value=NULL WHERE key='cash'")
    assert b.query_cash() == 0.0                  # float(None) 不再抛
    s = b.account_summary()                       # 不抛即通过
    assert s["cash"] == 0.0
    _rm(tmp)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
