"""实盘建议引擎（¥3000：2 仓位 / 1 手优先 / 含费可负担）单元测试。

纯函数测试：无网络、无账本 —— `plan_real_portfolio` 结构上就无法下单。

运行：python tests/test_real_advice.py
"""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.fill import FillConfig                                  # noqa: E402
from quant.trading.real_advice import AdviceInput, capacity_band, plan_real_portfolio

REAL_CFG = {
    "max_positions": 2, "lot_size": 100, "target_position_pct": 0.95,
    "max_stock_pct": 0.5, "single_position_pct": 0.9,
    "max_breakeven_pct": 1.5, "min_order_amount": 500.0,
    "advice": {"n": 3},
}
FC = FillConfig(min_commission=5.0, transfer_fee=0.00001)


def _q(price, ask=None, bid=None):
    return {"price": price, "low": price * 0.98, "high": price * 1.02,
            "ask": ask or [(round(price + 0.01, 2), 100000)],
            "bid": bid or [(round(price - 0.01, 2), 100000)], "volume": 1e6}


def _row(code, price, score=90.0):
    return {"code": code, "name": f"N{code}", "price": price, "total_score": score}


def test_capacity_band_for_3000():
    cap = capacity_band(3000.0, REAL_CFG, FC)
    # 2 仓位 × 100 股 → 每股上限 ≈ ¥14.2
    assert 14.0 < cap["price_ceiling"] <= 14.25, cap
    # 价格下限由「回本涨幅 ≤1.5%」和「单笔 ≥ ¥500」共同决定 → 约 ¥6.5–7.5
    assert 6.0 < cap["price_floor"] < 7.5, cap
    assert cap["price_floor"] < cap["price_ceiling"]
    assert cap["per_slot"] == 1425.0 and cap["max_positions"] == 2
    assert "可买价格带" in cap["band_text"]


def test_slots_zero_when_two_positions_held():
    inp = AdviceInput(
        rows=[_row("600160", 10.0)], cash=0.0, prices={"600160": 10.0, "601600": 9.0},
        positions=[{"symbol": "600160", "shares": 100, "avg_cost": 10.0, "sellable": 100},
                   {"symbol": "601600", "shares": 100, "avg_cost": 9.0, "sellable": 100}],
        cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert out["buy"] == [] and out["summary"]["slots"] == 0
    assert any("上限" in n for n in out["notes"])


def test_one_lot_first_then_second_when_affordable():
    # ¥7.2：2 手 = ¥1440 ≤ 单票上限 ¥1500 → 2 手
    inp = AdviceInput(rows=[_row("600160", 7.2)], cash=3000.0, prices={"600160": 7.2},
                      prev_closes={"600160": 7.2}, quotes={"600160": _q(7.2)},
                      cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert len(out["buy"]) == 1 and out["buy"][0]["est_shares"] == 200.0
    # ¥10：2 手 ¥2000 > 上限 → 只买 1 手
    inp2 = AdviceInput(rows=[_row("600160", 10.0)], cash=3000.0, prices={"600160": 10.0},
                       prev_closes={"600160": 10.0}, quotes={"600160": _q(10.0)},
                       cfg=REAL_CFG, fill_cfg=FC)
    out2 = plan_real_portfolio(inp2)
    assert out2["buy"][0]["est_shares"] == 100.0


def test_above_price_ceiling_skipped():
    # 已有 1 只持仓（现金 ¥1500 + 持仓 ¥1000 → 权益 ¥2500，单票上限 ¥1250）：
    # ¥20 一手需 ~¥2005 > 上限 → 跳过（且因已持仓，不会触发「单只放宽」例外）
    inp = AdviceInput(rows=[_row("600519", 20.0)], cash=1500.0,
                      prices={"600519": 20.0, "600160": 10.0},
                      positions=[{"symbol": "600160", "shares": 100,
                                  "avg_cost": 10.0, "sellable": 100}],
                      prev_closes={"600519": 20.0}, quotes={"600519": _q(20.0)},
                      cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert out["buy"] == []
    assert any("超单票上限" in s["reason"] for s in out["skipped"])


def test_single_position_exception_when_nothing_else_affordable():
    # 空仓 + 只有 ¥20 的票：2 仓位政策下配不齐，放宽到单只 90% 买 1 手（并注明例外）
    inp = AdviceInput(rows=[_row("600519", 20.0)], cash=3000.0, prices={"600519": 20.0},
                      prev_closes={"600519": 20.0}, quotes={"600519": _q(20.0)},
                      cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert len(out["buy"]) == 1 and out["buy"][0]["est_shares"] == 100.0
    assert any("例外" in n for n in out["notes"])


def test_uneconomic_low_price_skipped():
    # ¥3 一手：往返费用占比 ≈3.4% > 1.5% → 不经济
    inp = AdviceInput(rows=[_row("000001", 3.0)], cash=3000.0, prices={"000001": 3.0},
                      prev_closes={"000001": 3.0}, quotes={"000001": _q(3.0)},
                      cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert out["buy"] == []
    assert any("费用占比" in s["reason"] for s in out["skipped"])


def test_blocked_and_guard_are_skipped():
    inp = AdviceInput(rows=[_row("600160", 10.0), _row("600489", 9.0)],
                      cash=3000.0, prices={"600160": 10.0, "600489": 9.0},
                      prev_closes={"600160": 10.0, "600489": 9.0},
                      quotes={"600160": _q(10.0), "600489": _q(9.0)},
                      blocked={"600160"}, guards={"600489": (False, "涨幅 3.0% 追高")},
                      cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert out["buy"] == []
    reasons = " ".join(s["reason"] for s in out["skipped"])
    assert "止损卖出" in reasons and "追高" in reasons


def test_fill_status_propagates_to_action():
    # 现价较参考价 +5% → 已错过 → action=wait（不是 buy）
    inp = AdviceInput(rows=[_row("600160", 10.5)], cash=3000.0, prices={"600160": 10.5},
                      prev_closes={"600160": 10.0}, quotes={"600160": _q(10.5)},
                      cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    assert out["buy"] == [] and len(out["pending"]) == 1
    assert out["pending"][0]["action"] == "wait"
    assert out["pending"][0]["fill"]["status"] == "missed"


def test_sell_advice_with_stop_rule():
    inp = AdviceInput(
        rows=[], cash=0.0, prices={"600160": 9.0},
        positions=[{"symbol": "600160", "shares": 100, "avg_cost": 10.0, "sellable": 100}],
        prev_closes={"600160": 10.0}, quotes={"600160": _q(9.0)},
        risk_lines={"600160": {"stop_price": 9.2, "take_price": 11.0}},
        sell_rules={"600160": "止损：现价 9.00 跌破 9.20"},
        cfg=REAL_CFG, fill_cfg=FC)
    out = plan_real_portfolio(inp)
    s = out["sell"][0]
    assert s["action"] == "sell" and s["urgency"] == "high"
    assert s["pnl_pct"] < 0 and s["breakeven_price"] > 10.0      # 回本价高于成本（含费）
    assert s["fill"]["status"] in ("fillable", "likely", "hard", "missed", "blocked")


def test_plan_does_not_mutate_input():
    rows = [_row("600160", 10.0)]
    inp = AdviceInput(rows=rows, cash=3000.0, prices={"600160": 10.0},
                      prev_closes={"600160": 10.0}, quotes={"600160": _q(10.0)},
                      cfg=REAL_CFG, fill_cfg=FC)
    snapshot = (copy.deepcopy(inp.rows), copy.deepcopy(inp.prices),
                copy.deepcopy(inp.positions), inp.cash)
    plan_real_portfolio(inp)
    assert (inp.rows, inp.prices, inp.positions, inp.cash) == snapshot


def test_entry_gate_downgrades_buy_to_wait():
    """入场择时：现价在 5 日均线上方 → 不买，降级为「等回踩」并给出触发价（不改选谁）。"""
    base = dict(rows=[_row("600160", 10.0)], cash=3000.0, prices={"600160": 10.0},
                prev_closes={"600160": 10.0}, quotes={"600160": _q(10.0)},
                cfg=REAL_CFG, fill_cfg=FC)
    # 闸门放行（现价 ≤ MA5）→ 正常买入
    ok = plan_real_portfolio(AdviceInput(**base, entry_gate={"600160": {"ma5": 10.2, "ok": True}}))
    assert len(ok["buy"]) == 1 and ok["buy"][0]["action"] == "buy"
    # 闸门不放行（现价 > MA5）→ 降级为等回踩，触发价 = MA5
    wait = plan_real_portfolio(AdviceInput(**base, entry_gate={"600160": {"ma5": 9.7, "ok": False}}))
    assert wait["buy"] == [] and len(wait["pending"]) == 1
    row = wait["pending"][0]
    assert row["action"] == "wait" and row["trigger_price"] == 9.7
    assert "回踩" in row["reason"]
    # 不给闸门数据 → 不干预（向后兼容）
    plain = plan_real_portfolio(AdviceInput(**base))
    assert len(plain["buy"]) == 1


def test_risk_off_blocks_new_buys_and_halves_budget():
    """账户回撤熔断：暂停开新仓；且目标仓位上限下调（per_slot 减半）。"""
    base = dict(rows=[_row("600160", 10.0)], cash=3000.0, prices={"600160": 10.0},
                prev_closes={"600160": 10.0}, quotes={"600160": _q(10.0)},
                cfg=REAL_CFG, fill_cfg=FC)
    normal = plan_real_portfolio(AdviceInput(**base))
    assert len(normal["buy"]) == 1
    off = plan_real_portfolio(AdviceInput(**base, risk_off={
        "tripped": True, "position_pct": 0.5, "block_new_buys": True,
        "reason": "账户自近60日高点回撤 9.1% ≥ 8%"}))
    assert off["buy"] == []
    assert any("熔断" in s["reason"] for s in off["skipped"])
    assert off["capacity"]["per_slot"] < normal["capacity"]["per_slot"]


def test_shipped_config_gate_filters_inefficient_orders():
    """仓库 config 的 max_breakeven_pct（2026-09-11 由 1.5 收紧到 1.2）：
    ¥9.9 一手（名义 ¥990，费用≈1.06%）可过；¥7 一手（名义 ¥700，费用≈1.45%）被费用闸门滤掉。
    """
    from quant.config import load_config
    from quant.trading.fill import breakeven_pct
    cfg = load_config()["real"]
    fcfg = FillConfig.from_config(cfg)
    cap = float(cfg["max_breakeven_pct"])
    assert cap == 1.2
    assert breakeven_pct(9.9, 100, fcfg) * 100 <= cap
    assert breakeven_pct(7.0, 100, fcfg) * 100 > cap


def test_plan_has_no_broker_parameter():
    """安全断言：本模块不接收账本对象 → 结构上不可能下单。"""
    import inspect
    sig = inspect.signature(plan_real_portfolio)
    assert list(sig.parameters) == ["inp"]
    src = Path(__file__).resolve().parents[1] / "quant/trading/real_advice.py"
    text = src.read_text(encoding="utf-8")
    assert ".buy(" not in text and ".sell(" not in text


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
