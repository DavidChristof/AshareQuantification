"""成交可行性判定单元测试（quant/trading/fill.py）。

覆盖「实盘能不能成交」的逐分支：非交易时段、无行情、停牌、涨停/跌停封板、
价格漂移（hard/missed）、当日区间未触及、盘口深度、整手、资金不足、T+1、持仓不足，
以及费用（最低佣金/印花税/过户费）与回本涨幅的数值。

纯函数测试：无网络、无数据库、无 FastAPI。

运行：python tests/test_fill.py
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.fill import (                                    # noqa: E402
    BLOCKED, FILLABLE, HARD, LIKELY, MISSED, UNKNOWN, FillConfig,
    assess_buy, assess_sell, breakeven_pct, breakeven_price, buy_fees,
    is_suspended, limit_band, max_affordable_shares, round_lot_down,
    sell_fees, tick_round, touched_range,
)

CFG = FillConfig(min_commission=5.0, transfer_fee=0.00001)


def _quote(price, low=None, high=None, ask=None, bid=None, volume=1e6):
    return {"price": price, "low": low if low is not None else price,
            "high": high if high is not None else price,
            "ask": ask or [], "bid": bid or [], "volume": volume}


# ------------------------------------------------------------ 原语
def test_tick_round():
    assert tick_round(10.001, "buy") == 10.01          # 买入向上
    assert tick_round(10.009, "sell") == 10.0          # 卖出向下
    assert tick_round(10.1, "buy") == 10.1             # 浮点安全
    assert tick_round(0, "buy") == 0.0
    assert tick_round(float("nan"), "buy") == 0.0


def test_round_lot_down():
    assert round_lot_down(250, 100) == 200
    assert round_lot_down(99, 100) == 0
    assert round_lot_down(100, 100) == 100
    assert round_lot_down(0, 100) == 0


def test_limit_band():
    up, down = limit_band(10.0, "600519")              # 主板 ±10%
    assert (up, down) == (11.0, 9.0)
    up, down = limit_band(10.0, "300750")              # 创业板 ±20%
    assert (up, down) == (12.0, 8.0)
    assert limit_band(None, "600519") == (None, None)
    assert limit_band(0, "600519") == (None, None)


def test_fees_min_commission_floor_and_rate():
    # 小额触发最低佣金 ¥5；大额按费率
    f = buy_fees(1000.0, CFG)
    assert f["commission"] == 5.0
    assert f["fee"] == 5.0 + 1000.0 * 0.00001
    assert f["cash_needed"] == 1000.0 + f["fee"]
    f2 = buy_fees(1_000_000.0, CFG)
    assert abs(f2["commission"] - 300.0) < 1e-9          # 万3
    # 卖出含印花税
    s = sell_fees(1000.0, CFG)
    assert abs(s["stamp_tax"] - 0.5) < 1e-9
    assert s["net"] == 1000.0 - s["fee"]
    assert buy_fees(0, CFG)["fee"] == 0.0


def test_max_affordable_shares_never_overspends():
    cash, price = 3000.0, 14.0
    sh = max_affordable_shares(cash, price, CFG)
    assert sh == 200.0                                    # 3 手 4200 买不起
    assert buy_fees(sh * price, CFG)["cash_needed"] <= cash + 1e-9
    assert buy_fees((sh + 100) * price, CFG)["cash_needed"] > cash
    assert max_affordable_shares(300.0, 14.0, CFG) == 0.0  # 1 手 1400 买不起


def test_breakeven_math():
    # ¥14 一手的往返费用约 0.8%
    be14 = breakeven_pct(14.0, 100, CFG)
    assert 0.006 < be14 < 0.010, be14
    # ¥3 低价股费用占比高（≈3.4%）
    be3 = breakeven_pct(3.0, 100, CFG)
    assert 0.030 < be3 < 0.040, be3
    p = breakeven_price(14.0, 100, CFG)
    # 回本价必须 > 买价，且卖出净收入恰好覆盖买入支出
    assert p > 14.0
    target = buy_fees(14.0 * 100, CFG)["cash_needed"]
    assert abs(sell_fees(p * 100, CFG)["net"] - target) < 0.05


def test_is_suspended_and_touched():
    assert is_suspended({"price": 0, "high": 0, "low": 0, "volume": 0})
    assert is_suspended({"price": 10, "high": 0, "low": 0, "volume": 0})
    assert not is_suspended({"price": 10, "high": 10.5, "low": 9.9, "volume": 1000})
    assert not is_suspended(None)                          # 无行情 ≠ 停牌
    assert touched_range({"low": 9.5, "high": 10.5}) == (9.5, 10.5)
    assert touched_range(None, [{"low": 9.0, "high": 9.8}, {"low": 9.2, "high": 10.1}]) \
        == (9.0, 10.1)


def test_fill_config_from_config():
    c = FillConfig.from_config({
        "lot_size": 100, "commission": 0.0003, "min_commission": 5.0,
        "stamp_tax": 0.0005, "transfer_fee": 0.00001,
        "fill": {"drift_tol_pct": 0.8, "miss_tol_pct": 1.8, "depth_ratio": 2.0},
    })
    assert c.min_commission == 5.0 and c.drift_tol_pct == 0.8
    assert c.depth_ratio == 2.0 and c.lot_size == 100


# ------------------------------------------------------------ 买入判定
def test_buy_unknown_without_quote():
    a = assess_buy("603993", quote=None, cfg=CFG, cash=3000, shares=100,
                   prev_close=18.0)
    assert a.status == UNKNOWN and not a.ok
    assert "无实时行情" in " ".join(a.reasons)


def test_buy_blocked_when_closed_or_preopen():
    q = _quote(10.0, ask=[(10.01, 10000)])
    a = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, session="closed")
    assert a.status == BLOCKED
    b = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, session="pre")
    assert b.status == BLOCKED and b.suggested_price > 0   # 仍给计划价


def test_buy_blocked_when_suspended():
    q = _quote(10.0, low=0, high=0, volume=0)
    a = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0)
    assert a.status == BLOCKED and "停牌" in " ".join(a.reasons)


def test_buy_limit_up_is_hard():
    # 一字涨停：现价=涨停价，卖一无挂单
    q = _quote(11.0, low=11.0, high=11.0, ask=[(11.0, 0)])
    a = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0)
    assert a.status == HARD
    assert "涨停" in " ".join(a.reasons)
    # 涨停但卖一有挂单 → 仍 hard（排队）
    q2 = _quote(11.0, low=11.0, high=11.0, ask=[(11.0, 100000)])
    assert assess_buy("600519", quote=q2, cfg=CFG, cash=3000, shares=100,
                      prev_close=10.0).status == HARD


def test_buy_drift_hard_then_missed():
    hi = _quote(10.15, low=10.0, high=10.2, ask=[(10.16, 10000)])
    a = assess_buy("600519", quote=hi, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, reference=10.0)
    assert a.status == HARD                                  # +1.5% 需改价
    far = _quote(10.25, low=10.0, high=10.3, ask=[(10.26, 10000)])
    b = assess_buy("600519", quote=far, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, reference=10.0)
    assert b.status == MISSED                                # +2.5% 已错过


def test_buy_ask_above_today_high_is_hard():
    # 委托价（卖一）高于今日最高价 → 需价格上抬
    q = _quote(10.0, low=9.5, high=10.0, ask=[(10.50, 10000)])
    a = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, reference=10.0)
    assert a.status == HARD and a.touched is False


def test_buy_depth_insufficient_is_likely():
    q = _quote(10.0, low=9.9, high=10.1, ask=[(10.01, 50)])
    a = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, reference=10.0)
    assert a.status == LIKELY and a.depth_ok is False


def test_buy_lot_and_cash_blocked():
    q = _quote(10.0, low=9.9, high=10.1, ask=[(10.01, 10000)])
    bad_lot = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=150,
                         prev_close=10.0, reference=10.0)
    assert bad_lot.status == BLOCKED and "整手" in " ".join(bad_lot.reasons)
    poor = assess_buy("600519", quote=q, cfg=CFG, cash=100, shares=100,
                      prev_close=10.0, reference=10.0)
    assert poor.status == BLOCKED and "资金不足" in " ".join(poor.reasons)


def test_buy_happy_path_fillable():
    q = _quote(10.0, low=9.8, high=10.2, ask=[(10.01, 10000)])
    a = assess_buy("600519", quote=q, cfg=CFG, cash=3000, shares=100,
                   prev_close=10.0, reference=10.0)
    assert a.status == FILLABLE and a.ok
    assert a.suggested_price == 10.01
    assert a.est_cash_delta < 0                       # 买入现金流出
    assert 0 < a.breakeven_pct < 0.02


# ------------------------------------------------------------ 卖出判定
def test_sell_t1_and_insufficient_blocked():
    q = _quote(10.0, low=9.9, high=10.1, bid=[(9.99, 10000)])
    t1 = assess_sell("600519", quote=q, cfg=CFG, shares=100, held=100, sellable=0,
                     prev_close=10.0)
    assert t1.status == BLOCKED and "T+1" in " ".join(t1.reasons)
    short = assess_sell("600519", quote=q, cfg=CFG, shares=100, held=50, sellable=50,
                        prev_close=10.0)
    assert short.status == BLOCKED and "持仓不足" in " ".join(short.reasons)


def test_sell_limit_down_is_hard():
    q = _quote(9.0, low=9.0, high=9.0, bid=[(9.0, 0)])       # 一字跌停，买一无量
    a = assess_sell("600519", quote=q, cfg=CFG, shares=100, held=100, sellable=100,
                    prev_close=10.0)
    assert a.status == HARD and "跌停" in " ".join(a.reasons)


def test_sell_depth_insufficient_is_likely():
    q = _quote(10.0, low=9.9, high=10.1, bid=[(9.99, 50)])
    a = assess_sell("600519", quote=q, cfg=CFG, shares=100, held=100, sellable=100,
                    prev_close=10.0)
    assert a.status == LIKELY and a.depth_ok is False


def test_sell_happy_path_fillable():
    q = _quote(10.0, low=9.8, high=10.2, bid=[(9.99, 10000)])
    a = assess_sell("600519", quote=q, cfg=CFG, shares=100, held=100, sellable=100,
                    prev_close=10.0)
    assert a.status == FILLABLE and a.ok
    assert a.est_cash_delta > 0                       # 卖出净收入为正


def test_sell_unknown_without_quote():
    a = assess_sell("603993", quote=None, cfg=CFG, shares=100, held=100, sellable=100,
                    prev_close=18.0)
    assert a.status == UNKNOWN


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
