"""实盘建议（¥3000 本金：最多 2 只、1 手起配、含费可负担、成交可行性）。

纯函数模块：**不接收 broker、不触碰任何账本** → 结构上不可能下单（这也是它可测的原因）。
调用方（api/main.py）负责把实时行情、止损止盈线、追高闸门结果喂进来。

¥3000 的硬算术（本模块第一件要告诉用户的事）：
    2 仓位 × 100 股/手 → 每股上限 ≈ 3000/2/100 = ¥15；
    再计入单笔最低佣金 ¥5，可买价格带约 **¥7 – ¥14**（低价股费用占比过高、不经济）。
    所以建议经常只有 0–1 只 —— 必须把原因讲清楚，而不是给张空表。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .fill import (
    OK_STATUSES, FillConfig, assess_buy, assess_sell, breakeven_pct,
    breakeven_price, buy_fees, round_lot_down, sell_fees, tick_round,
)


def _num(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


@dataclass
class AdviceInput:
    """一次实盘建议的全部输入（由 api 层装配）。"""
    rows: list[dict] = field(default_factory=list)          # 选股候选（含 code/name/price/total_score）
    positions: list[dict] = field(default_factory=list)     # [{symbol,shares,avg_cost,sellable}]
    cash: float = 0.0
    prices: dict[str, float] = field(default_factory=dict)
    prev_closes: dict[str, float] = field(default_factory=dict)
    quotes: dict[str, dict] = field(default_factory=dict)
    risk_lines: dict[str, dict] = field(default_factory=dict)
    guards: dict[str, tuple] = field(default_factory=dict)  # {symbol: (ok, reason)}
    # 入场择时：{symbol: {"ma5": float, "ok": bool}}；ok=False（现价在 5 日均线上方）→ 降级为"等回踩"
    entry_gate: dict = field(default_factory=dict)
    # 账户回撤熔断/弱势减仓：{"tripped":bool,"position_pct":float,"block_new_buys":bool,"reason":str}
    risk_off: dict = field(default_factory=dict)
    # 大盘趋势闸门：{"below":bool,"reason":str} —— 跌破 MA20 → 当日不开新仓（保留持仓）
    market_gate: dict = field(default_factory=dict)
    blocked: set = field(default_factory=set)               # 今日已建议止损卖出 → 不买回
    sell_rules: dict[str, str] = field(default_factory=dict)  # {symbol: 触发原因}
    session: str = "open"
    opened_minutes: float | None = None
    market_weak: Mapping | None = None
    cfg: Mapping = field(default_factory=dict)              # cfg["real"]
    fill_cfg: FillConfig = field(default_factory=FillConfig)


def capacity_band(equity: float, cfg: Mapping, fill_cfg: FillConfig,
                  pos_pct_override: float | None = None) -> dict:
    """本账户「买得起的价格带」—— ¥3000 的硬约束，必须算给用户看。

    pos_pct_override: 回撤熔断等场景下传入下调后的总仓位上限（默认取配置值）。
    """
    n = max(int(cfg.get("max_positions", 2) or 2), 1)
    lot = max(int(cfg.get("lot_size", 100) or 100), 1)
    pos_pct = float(pos_pct_override if pos_pct_override is not None
                    else (cfg.get("target_position_pct", 0.95) or 0.95))
    cap_pct = float(cfg.get("max_stock_pct", 0.5) or 0.5)
    equity = max(_num(equity), 0.0)
    cap_value = equity * cap_pct
    per_slot = min(equity * pos_pct / n, cap_value) if equity > 0 else 0.0

    # 每股上限：满足「1 手含费 ≤ 每仓预算」的最高股价
    ceiling = 0.0
    if per_slot > 0:
        p = tick_round(per_slot / lot, "sell")
        while p > 0 and buy_fees(p * lot, fill_cfg)["cash_needed"] > per_slot + 1e-9:
            p = tick_round(p - 0.01, "sell")
        ceiling = p
    # 价格下限：回本涨幅 ≤ 上限（最低佣金被摊薄）且单笔金额 ≥ 下限
    floor = 0.0
    if ceiling > 0:
        be_cap = float(cfg.get("max_breakeven_pct", 1.5) or 1.5)
        min_amt = float(cfg.get("min_order_amount", 0.0) or 0.0)
        p = tick_round(max(min_amt / lot, 0.01), "buy")
        while p <= ceiling and breakeven_pct(p, lot, fill_cfg) * 100 > be_cap:
            p = tick_round(p + 0.01, "buy")
        floor = p if p <= ceiling else ceiling

    text = (f"可买价格带 ¥{floor:.2f} – ¥{ceiling:.2f}"
            f"（¥{equity:.0f} ÷ {n} 仓位 ÷ {lot} 股 = 每股上限 ≈ ¥{per_slot / lot:.2f}；"
            f"再计单笔最低佣金 ¥{fill_cfg.min_commission:.0f}）") if ceiling > 0 \
        else "账户无可买能力（现金/市值过低）"
    return {
        "max_positions": n, "lot_size": lot,
        "per_slot": round(per_slot, 2), "max_stock_value": round(cap_value, 2),
        "price_floor": round(floor, 2), "price_ceiling": round(ceiling, 2),
        "band_text": text,
        "one_lot_hint": (f"每仓约 ¥{per_slot:.0f}，约 {max(int(per_slot // max(ceiling * lot, 1)), 1)} 手"
                         if ceiling > 0 else ""),
    }


def plan_real_portfolio(inp: AdviceInput) -> dict:
    """产出 ¥3000 的建仓/卖出建议（含成交判定）。无副作用、不接触账本。"""
    cfg, fcfg = inp.cfg or {}, inp.fill_cfg
    n = max(int(cfg.get("max_positions", 2) or 2), 1)
    lot = max(int(cfg.get("lot_size", 100) or 100), 1)
    pos_pct = float(cfg.get("target_position_pct", 0.95) or 0.95)
    cap_pct = float(cfg.get("max_stock_pct", 0.5) or 0.5)
    n_out = max(int((cfg.get("advice") or {}).get("n", 3) or 3), 1)

    positions = list(inp.positions or [])
    held_syms = {str(p.get("symbol")) for p in positions}

    # ---- 资金 ----
    held_value = 0.0
    for p in positions:
        sym = str(p.get("symbol"))
        px = _num(inp.prices.get(sym)) or _num(p.get("avg_cost"))
        held_value += px * _num(p.get("shares"))
    equity = _num(inp.cash) + held_value
    # 账户回撤熔断：触发期间总仓位上限下调（与组合层同口径）
    risk_off = dict(inp.risk_off or {})
    brake_on = bool(risk_off.get("tripped"))
    mkt_gate = dict(inp.market_gate or {})
    gate_below = bool(mkt_gate.get("below"))
    gate_reason = str(mkt_gate.get("reason") or "指数跌破均线")
    if brake_on:
        pos_pct = min(pos_pct, float(risk_off.get("position_pct", 0.5) or 0.5))
    cap_value = equity * cap_pct
    per_slot = min(equity * pos_pct / n, cap_value) if equity > 0 else 0.0
    slots = max(n - len(positions), 0)

    capacity = capacity_band(equity, cfg, fcfg, pos_pct_override=pos_pct)
    notes: list[str] = []
    skipped: list[dict] = []
    buy: list[dict] = []
    pending: list[dict] = []       # 暂不可成交（难成交/已错过）——等改价或回落
    backup: list[dict] = []

    if slots <= 0:
        notes.append(f"已持有 {len(positions)}/{n} 只达到上限 —— 先卖出腾出仓位再买")

    # ---- 买入扫描（按综合分降序）----
    rows = sorted(list(inp.rows or []),
                  key=lambda r: -_num(r.get("total_score")))
    cash_left = _num(inp.cash)
    for r in rows:
        sym = str(r.get("code"))
        name = r.get("name", sym)
        price = _num(inp.prices.get(sym)) or _num(r.get("price"))
        base = {"symbol": sym, "name": name, "score": round(_num(r.get("total_score")), 2),
                "price": round(price, 3) if price else None}

        if price <= 0:
            skipped.append({**base, "reason": "无有效价格"})
            continue
        if sym in inp.blocked:
            skipped.append({**base, "reason": "今日已建议止损卖出，不买回"})
            continue
        if brake_on and risk_off.get("block_new_buys", True):
            skipped.append({**base, "reason": (
                f"账户回撤熔断：暂停开新仓（{risk_off.get('reason', '')}）")})
            continue
        if gate_below:
            skipped.append({**base, "reason": (
                f"大盘趋势闸门：{gate_reason}")})
            continue

        one = buy_fees(price * lot, fcfg)
        # 已持有 → 加仓 / 持有
        if sym in held_syms:
            pos = next(p for p in positions if str(p.get("symbol")) == sym)
            cur_val = _num(pos.get("shares")) * price
            add_cost = one["cash_needed"]
            if (cur_val + price * lot <= cap_value + 1e-6 and cash_left >= add_cost
                    and len(buy) < slots):
                a = assess_buy(sym, quote=inp.quotes.get(sym), cfg=fcfg, cash=cash_left,
                               shares=lot, prev_close=inp.prev_closes.get(sym),
                               reference=inp.prev_closes.get(sym), session=inp.session,
                               opened_minutes=inp.opened_minutes)
                row = {**base, "action": "add", "est_shares": lot,
                       "est_amount": round(price * lot, 2), "est_fee": round(one["fee"], 2),
                       "est_cash_needed": round(add_cost, 2),
                       "reason": f"加仓 1 手（当前市值 ¥{cur_val:.0f}，距单票上限 ¥{cap_value:.0f} 尚有空间）",
                       "fill": a.to_dict()}
                buy.append(row)
                cash_left -= add_cost
            continue

        # 名额已满 → 只作备选展示
        as_backup = len(buy) >= slots
        if as_backup and len(buy) + len(pending) + len(backup) >= n_out:
            skipped.append({**base, "reason": "名额已满（备选已足）"})
            continue

        # 单票上限
        if one["cash_needed"] > cap_value + 1e-6:
            skipped.append({**base, "reason": (
                f"1 手 ¥{one['cash_needed']:.0f} 超单票上限 ¥{cap_value:.0f}"
                f"（本账户可买价格上限 ≈ ¥{capacity['price_ceiling']:.2f}）")})
            continue
        # 现金
        if one["cash_needed"] > cash_left + 1e-6:
            skipped.append({**base, "reason": (
                f"现金不足：1 手需 ¥{one['cash_needed']:.2f}，可用 ¥{cash_left:.2f}")})
            continue
        # 经济性（低价股费用占比过高）
        be = breakeven_pct(price, lot, fcfg) * 100
        if be > float(cfg.get("max_breakeven_pct", 1.5) or 1.5):
            rt = one["fee"] + sell_fees(price * lot, fcfg)["fee"]
            skipped.append({**base, "reason": (
                f"费用占比过高：往返约 ¥{rt:.2f}（佣金最低 ¥{fcfg.min_commission:.0f} 起），"
                f"需涨 {be:.2f}% 才回本 → 不经济")})
            continue
        if price * lot < float(cfg.get("min_order_amount", 0.0) or 0.0):
            skipped.append({**base, "reason": f"单笔金额 ¥{price * lot:.0f} 过低"})
            continue
        # 追高/接飞刀（由 api 层用同一套 _guard_check 判定后注入）
        ok, why = inp.guards.get(sym, (True, ""))
        if not ok:
            skipped.append({**base, "reason": f"追高保护：{why}"})
            continue

        # 1 手优先；付得起且不破上限才加第 2 手
        shares = float(lot)
        if (cash_left - one["cash_needed"] >= one["cash_needed"]
                and price * lot * 2 <= cap_value + 1e-6):
            shares = float(2 * lot)

        a = assess_buy(sym, quote=inp.quotes.get(sym), cfg=fcfg, cash=cash_left,
                       shares=shares, prev_close=inp.prev_closes.get(sym),
                       reference=inp.prev_closes.get(sym), session=inp.session,
                       opened_minutes=inp.opened_minutes)
        need = buy_fees(shares * (a.suggested_price or price), fcfg)["cash_needed"]
        row = {
            **base, "suggested_price": a.suggested_price,
            "est_shares": shares, "est_amount": round(price * shares, 2),
            "est_fee": round(a.est_fee, 2), "est_cash_needed": round(need, 2),
            "breakeven_pct": round(be / 100, 6),
            "breakeven_price": round(breakeven_price(price, shares, fcfg), 3),
            "fill": a.to_dict(),
            "reference_price": inp.prev_closes.get(sym),
            "in_universe": bool(r.get("in_universe", False)),
        }
        if a.status in OK_STATUSES:
            gate = inp.entry_gate.get(sym) or {}
            if gate and gate.get("ok") is False:
                # 入场择时：现价在 5 日均线上方 → 不追，降级为"等回踩"（不改选谁，只改何时下手）
                ma5 = _num(gate.get("ma5"))
                row["action"] = "wait"
                row["trigger_price"] = round(ma5, 3) if ma5 > 0 else None
                row["reason"] = (f"未回踩：现价在 5 日均线 ¥{ma5:.2f} 上方 → 不追高，"
                                 f"等回到 ¥{ma5:.2f} 下方再买（实证：回踩入场 5 日收益更优）")
                pending.append(row)
                continue
            row["action"] = "buy"
            row["reason"] = "按建议委托价下单（人工）"
            if as_backup:
                row["action"] = "backup"
                row["reason"] = "备选：前序建议无法执行时可用"
                backup.append(row)
                continue
            buy.append(row)
            cash_left -= need
        elif a.status in ("hard", "missed"):
            row["action"] = "wait"
            row["trigger_price"] = tick_round(_num(inp.prev_closes.get(sym)) * 1.0, "buy") \
                if a.status == "hard" else None
            row["reason"] = "暂难成交/已错过 —— 按提示改价或等回落"
            pending.append(row)
        else:
            skipped.append({**base, "reason": "；".join(a.reasons[:1]) or a.label})

    # ---- 空仓且两手都配不齐 → 放宽到单只（例外，注明）----
    # ⚠️ 例外通道只放宽「单票上限」，**其余闸门一条都不能绕**（熔断/blocked/追高/入场择时/费用）。
    #    历史上这里先后漏过 追高保护、入场择时，现补熔断 —— 新增闸门时务必同步此处。
    brake_blocking = brake_on and risk_off.get("block_new_buys", True)
    no_new = brake_blocking or gate_below        # 熔断 / 大盘跌破均线 → 连例外通道也关闭
    if not buy and not positions and slots > 0 and rows and not no_new:
        single_pct = float(cfg.get("single_position_pct", 0.0) or 0.0)
        if single_pct > cap_pct:
            cap2 = equity * single_pct
            for r in rows:
                sym = str(r.get("code"))
                price = _num(inp.prices.get(sym)) or _num(r.get("price"))
                if price <= 0 or sym in inp.blocked:
                    continue
                # 例外通道同样必须过闸门与费用关，否则等于绕过追高保护/经济性筛选/入场择时
                if not inp.guards.get(sym, (True, ""))[0]:
                    continue
                if (inp.entry_gate.get(sym) or {}).get("ok") is False:
                    continue
                if price * lot < float(cfg.get("min_order_amount", 0.0) or 0.0):
                    continue
                if breakeven_pct(price, lot, fcfg) * 100 > \
                        float(cfg.get("max_breakeven_pct", 1.5) or 1.5):
                    continue
                one = buy_fees(price * lot, fcfg)
                if one["cash_needed"] <= cap2 + 1e-6 and one["cash_needed"] <= _num(inp.cash) + 1e-6:
                    a = assess_buy(sym, quote=inp.quotes.get(sym), cfg=fcfg,
                                   cash=_num(inp.cash), shares=float(lot),
                                   prev_close=inp.prev_closes.get(sym),
                                   reference=inp.prev_closes.get(sym),
                                   session=inp.session, opened_minutes=inp.opened_minutes)
                    if a.status in OK_STATUSES:
                        buy.append({
                            "symbol": sym, "name": r.get("name", sym), "action": "buy",
                            "score": round(_num(r.get("total_score")), 2),
                            "price": round(price, 3),
                            "suggested_price": a.suggested_price,
                            "est_shares": float(lot), "est_amount": round(price * lot, 2),
                            "est_fee": round(a.est_fee, 2),
                            "est_cash_needed": round(one["cash_needed"], 2),
                            "breakeven_pct": round(breakeven_pct(price, lot, fcfg), 6),
                            "breakeven_price": round(breakeven_price(price, lot, fcfg), 3),
                            "fill": a.to_dict(),
                            "reason": f"仅此 1 只可负担，已放宽至单票 {single_pct:.0%}（2 仓位政策下的例外）",
                        })
                        notes.append(f"⚠️ 例外：无第二只可负担，已放宽单只上限至 "
                                     f"{single_pct:.0%}（¥{cap2:.0f}）买 {sym} 1 手")
                        break

    # ---- 卖出/持有建议（实盘只提醒，绝不自动卖）----
    sell: list[dict] = []
    for p in positions:
        sym = str(p.get("symbol"))
        shares = _num(p.get("shares"))
        avg_cost = _num(p.get("avg_cost"))
        price = _num(inp.prices.get(sym)) or avg_cost
        rl = inp.risk_lines.get(sym, {}) or {}
        reason = inp.sell_rules.get(sym)
        urgency = "high" if (reason and "止损" in reason) else \
            ("mid" if (reason and "止盈" in reason) else "low")
        sellable = _num(p.get("sellable"), shares)
        ref = rl.get("stop_price") or rl.get("take_price")
        a = assess_sell(sym, quote=inp.quotes.get(sym), cfg=fcfg, shares=shares,
                        held=shares, sellable=sellable, prev_close=inp.prev_closes.get(sym),
                        reference=ref, session=inp.session,
                        opened_minutes=inp.opened_minutes, cost=avg_cost,
                        reason=reason or "")
        sell.append({
            "symbol": sym, "name": inp.prices.get(f"__name__{sym}", sym),
            "action": "sell" if reason else "hold",
            "urgency": urgency if reason else None,
            "reason": reason or "未触发卖出条件，继续持有",
            "shares": shares, "sellable_shares": sellable,
            "t1_locked": sellable < shares - 1e-6,
            "avg_cost": round(avg_cost, 3), "price": round(price, 3),
            "pnl_pct": round(price / avg_cost - 1, 4) if avg_cost > 0 else None,
            "stop_price": rl.get("stop_price"), "take_price": rl.get("take_price"),
            "suggested_price": a.suggested_price,
            "est_net_proceeds": a.est_cash_delta,
            "realized_pnl": (round(a.est_cash_delta - shares * avg_cost, 2)
                             if reason and avg_cost > 0 else None),
            "breakeven_price": round(breakeven_price(avg_cost, shares, fcfg), 3)
            if avg_cost > 0 else None,
            "fill": a.to_dict(),
        })

    # ---- 零结果要诚实说明原因 ----
    if not buy and slots > 0:
        notes.append(f"今日无「可立即成交」的买入标的：{capacity['band_text']}；"
                     f"共 {len(skipped)} 只候选因价格/费用/限额被跳过 —— 建议空仓观望，"
                     f"不要为了满仓去买不符合条件的票")
    if not positions:
        notes.append(f"当前空仓，现金 ¥{_num(inp.cash):.2f}；单票预算约 ¥{per_slot:.0f}")

    return {
        "capacity": capacity,
        "buy": buy, "pending": pending, "backup": backup,
        "sell": sell, "skipped": skipped, "notes": notes,
        "summary": {
            "slots": slots, "buy": len(buy), "pending": len(pending),
            "backup": len(backup), "sell": sum(1 for s in sell if s["action"] == "sell"),
            "hold": sum(1 for s in sell if s["action"] == "hold"),
            "equity": round(equity, 2), "cash": round(_num(inp.cash), 2),
            "position_count": len(positions), "max_positions": n,
        },
    }
