"""成交可行性判定（纯函数；不 import fastapi / requests / 行情层）。

实盘痛点：模型给出的「建议价」≠ 你能成交的价。T+1、涨跌停封板、整手、最低佣金、
盘中价格漂移，都会让纸面上完美的建议在实盘落空。本模块把「这单能不能成交」
变成可计算、可穷举测试的判定，供 /api/real/advice、/api/real/check 与前端徽章使用。

状态（严重度递增）：
    fillable 可成交 → likely 大概率（如盘口深度不足，可能部分成交）
    → hard 难成交（涨停/跌停封板、委托价高于今日最高、需改价）
    → missed 已错过（价格已跑远）
    → blocked 不可下单（非交易时段/停牌/T+1/整手/资金不足）
    另有 unknown 无行情（信息不足，不能瞎判——600 池候选常无实时行情）。

用法：
    cfg = FillConfig.from_config(cfg["real"])
    a = assess_buy("603993", quote=q, prev_close=18.20, cash=3000, shares=100, cfg=cfg)
    a.status, a.label, a.reasons, a.suggested_price
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .rules import limit_prices

FILLABLE, LIKELY, HARD, MISSED, BLOCKED, UNKNOWN = (
    "fillable", "likely", "hard", "missed", "blocked", "unknown")

STATUS_LABELS = {
    FILLABLE: "可成交", LIKELY: "大概率", HARD: "难成交",
    MISSED: "已错过", BLOCKED: "不可下单", UNKNOWN: "无行情",
}
OK_STATUSES = (FILLABLE, LIKELY)
_SEVERITY = {FILLABLE: 0, LIKELY: 1, HARD: 2, MISSED: 3, BLOCKED: 4}


def _worse(a: str, b: str) -> str:
    """取更坏（更严重）的状态。"""
    return a if _SEVERITY.get(a, 0) >= _SEVERITY.get(b, 0) else b


def _num(v: Any, default: float = 0.0) -> float:
    """安全转 float：None/NaN/inf/字符串 → default。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ---------------------------------------------------------------- 配置
@dataclass(frozen=True)
class FillConfig:
    """成交判定与费用的全部可调项（来源：config real: 段）。"""
    lot_size: int = 100
    commission: float = 0.0003
    min_commission: float = 5.0
    stamp_tax: float = 0.0005
    transfer_fee: float = 0.00001
    slippage: float = 0.0
    # 判定阈值
    drift_tol_pct: float = 1.0      # 漂移 > 此值 → hard（需改价）
    miss_tol_pct: float = 2.0       # 漂移 > 此值 → missed（已错过）
    touch_tol_pct: float = 0.3      # 贴近当日高点容忍（%）
    near_limit_pct: float = 0.5     # 距涨跌停多近算「封板风险」
    depth_ratio: float = 1.0        # 盘口量/建议股数 < 此值 → likely
    open_grace_min: float = 5.0     # 开盘后 N 分钟内漂移阈值放宽 1 倍
    max_breakeven_pct: float = 1.5  # 回本涨幅上限（经济性提示）
    min_order_amount: float = 500.0

    @classmethod
    def from_config(cls, real_cfg: Mapping | None) -> "FillConfig":
        c = real_cfg or {}
        f = c.get("fill") or {}
        return cls(
            lot_size=int(c.get("lot_size", 100) or 100),
            commission=float(c.get("commission", 0.0003)),
            min_commission=float(c.get("min_commission", 0.0) or 0.0),
            stamp_tax=float(c.get("stamp_tax", 0.0005)),
            transfer_fee=float(c.get("transfer_fee", 0.0) or 0.0),
            slippage=float(c.get("slippage", 0.0) or 0.0),
            drift_tol_pct=float(f.get("drift_tol_pct", 1.0)),
            miss_tol_pct=float(f.get("miss_tol_pct", 2.0)),
            touch_tol_pct=float(f.get("touch_tol_pct", 0.3)),
            near_limit_pct=float(f.get("near_limit_pct", 0.5)),
            depth_ratio=float(f.get("depth_ratio", 1.0)),
            open_grace_min=float(f.get("open_grace_min", 5.0)),
            max_breakeven_pct=float(c.get("max_breakeven_pct", 1.5)),
            min_order_amount=float(c.get("min_order_amount", 500.0)),
        )


# ---------------------------------------------------------------- 结果
@dataclass
class FillAssessment:
    """一次「这单能不能成交」的判定结果（可直接 to_dict 给前端）。"""
    symbol: str
    side: str                      # buy / sell
    status: str
    label: str
    ok: bool
    reasons: list[str] = field(default_factory=list)
    rule_warnings: list[str] = field(default_factory=list)
    suggested_price: float = 0.0   # 建议委托价（已 tick 取整，不越涨跌停）
    price: float | None = None     # 判定基准现价
    reference_price: float | None = None
    drift_pct: float | None = None
    limit_up: float | None = None
    limit_down: float | None = None
    shares: float = 0.0
    est_amount: float = 0.0
    est_fee: float = 0.0
    est_cash_delta: float = 0.0    # 买 = -(金额+费)；卖 = +(金额-费)
    breakeven_pct: float | None = None
    breakeven_price: float | None = None
    touched: bool | None = None
    depth_ok: bool | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side,
            "status": self.status, "label": self.label, "ok": self.ok,
            "reasons": self.reasons, "rule_warnings": self.rule_warnings,
            "suggested_price": self.suggested_price,
            "price": self.price, "reference_price": self.reference_price,
            "drift_pct": self.drift_pct,
            "limit_up": self.limit_up, "limit_down": self.limit_down,
            "shares": self.shares, "est_amount": self.est_amount,
            "est_fee": self.est_fee, "est_cash_delta": self.est_cash_delta,
            "breakeven_pct": self.breakeven_pct,
            "breakeven_price": self.breakeven_price,
            "touched": self.touched, "depth_ok": self.depth_ok,
            "note": self.note,
        }


# ---------------------------------------------------------------- 原语
def tick_round(price: float, side: str = "buy") -> float:
    """A股最小变动 0.01：买入向上取整（更易成交），卖出向下取整。"""
    p = _num(price)
    if p <= 0:
        return 0.0
    if side == "buy":
        return math.ceil(p * 100 - 1e-9) / 100
    return math.floor(p * 100 + 1e-9) / 100


def round_lot_down(shares: float, lot_size: int = 100) -> float:
    """向下取整到整手（绝不超出可负担股数）。"""
    lot = max(int(lot_size or 1), 1)
    s = _num(shares)
    if s <= 0:
        return 0.0
    return float(int(s // lot) * lot)


def limit_band(prev_close: float | None, symbol: str) -> tuple[float | None, float | None]:
    """(涨停价, 跌停价)；前收缺失/非法 → (None, None)。"""
    pc = _num(prev_close)
    if pc <= 0:
        return None, None
    return limit_prices(pc, symbol)


def buy_fees(amount: float, cfg: FillConfig) -> dict:
    """买入费用 = 佣金(≥最低佣金) + 过户费。"""
    gross = max(_num(amount), 0.0)
    if gross <= 0:
        return {"gross": 0.0, "commission": 0.0, "transfer_fee": 0.0,
                "fee": 0.0, "cash_needed": 0.0}
    commission = max(gross * cfg.commission, cfg.min_commission)
    transfer = gross * cfg.transfer_fee
    fee = commission + transfer
    return {"gross": gross, "commission": commission, "transfer_fee": transfer,
            "fee": fee, "cash_needed": gross + fee}


def sell_fees(amount: float, cfg: FillConfig) -> dict:
    """卖出费用 = 佣金(≥最低佣金) + 印花税(单边) + 过户费。"""
    gross = max(_num(amount), 0.0)
    if gross <= 0:
        return {"gross": 0.0, "commission": 0.0, "stamp_tax": 0.0,
                "transfer_fee": 0.0, "fee": 0.0, "net": 0.0}
    commission = max(gross * cfg.commission, cfg.min_commission)
    stamp = gross * cfg.stamp_tax
    transfer = gross * cfg.transfer_fee
    fee = commission + stamp + transfer
    return {"gross": gross, "commission": commission, "stamp_tax": stamp,
            "transfer_fee": transfer, "fee": fee, "net": gross - fee}


def max_affordable_shares(cash: float, price: float, cfg: FillConfig,
                          lot_size: int | None = None) -> float:
    """含最低佣金的**精确**可买股数（向下整手）。

    朴素 `cash/(price*(1+c))` 在有最低佣金时会高估 → 必须逐手回退校验。
    """
    lot = max(int(lot_size or cfg.lot_size or 1), 1)
    cash = _num(cash)
    price = _num(price)
    if cash <= 0 or price <= 0:
        return 0.0
    est = cash / (price * (1 + cfg.commission + cfg.transfer_fee))
    shares = round_lot_down(est, lot)
    while shares >= lot and buy_fees(shares * price, cfg)["cash_needed"] > cash + 1e-9:
        shares -= lot
    return max(shares, 0.0)


def breakeven_price(buy_price: float, shares: float, cfg: FillConfig) -> float:
    """含双边费用的回本价：卖出净收入 == 买入总支出 时的卖出价。

    卖出费用随价格线性单调 → 不动点迭代 10 次即收敛（无需讨论是否触发最低佣金）。
    """
    bp, sh = _num(buy_price), _num(shares)
    if bp <= 0 or sh <= 0:
        return bp
    target = buy_fees(bp * sh, cfg)["cash_needed"]
    p = target / sh
    for _ in range(10):
        p = (target + sell_fees(p * sh, cfg)["fee"]) / sh
    return p


def breakeven_pct(buy_price: float, shares: float, cfg: FillConfig) -> float:
    """回本所需涨幅（小数，如 0.0077 = +0.77%）。"""
    bp = _num(buy_price)
    if bp <= 0:
        return 0.0
    return breakeven_price(bp, shares, cfg) / bp - 1.0


def is_suspended(quote: Mapping | None) -> bool:
    """疑似停牌：现价≤0，或今日高=低=0 且成交量为 0。"""
    if not quote:
        return False
    price = _num(quote.get("price"))
    high, low = _num(quote.get("high")), _num(quote.get("low"))
    vol = _num(quote.get("volume"))
    if price <= 0:
        return True
    return high <= 0 and low <= 0 and vol <= 0


def touched_range(quote: Mapping | None,
                  bars: Sequence[Mapping] | None = None) -> tuple[float, float] | None:
    """当日已触及区间 (low, high)：优先实时快照，缺失时用分钟K线聚出。"""
    if quote:
        low, high = _num(quote.get("low")), _num(quote.get("high"))
        if low > 0 and high > 0:
            return low, high
    if bars:
        lows = [_num(b.get("low")) for b in bars]
        highs = [_num(b.get("high")) for b in bars]
        lows = [v for v in lows if v > 0]
        highs = [v for v in highs if v > 0]
        if lows and highs:
            return min(lows), max(highs)
    return None


def _best_level(levels: Any) -> tuple[float, float]:
    """盘口一档 → (价, 量)；新浪/腾讯格式为 [(price, qty), ...]。"""
    try:
        if levels:
            p, q = levels[0][0], levels[0][1]
            return _num(p), _num(q)
    except (TypeError, IndexError, KeyError):
        pass
    return 0.0, 0.0


def suggest_order_price(side: str, quote: Mapping | None, fallback: float | None,
                        limit_up: float | None, cfg: FillConfig) -> float:
    """建议委托价：买取卖一价、卖取买一价；无盘口退回现价/参考价，再 tick 取整。"""
    cand = 0.0
    if quote:
        if side == "buy":
            cand, _ = _best_level(quote.get("ask"))
        else:
            cand, _ = _best_level(quote.get("bid"))
        if cand <= 0:
            cand = _num(quote.get("price"))
    if cand <= 0:
        cand = _num(fallback)
    if cand <= 0:
        return 0.0
    p = tick_round(cand, side)
    if side == "buy" and limit_up and p > limit_up:
        p = limit_up                      # 委托价不越涨停
    return p


# ---------------------------------------------------------------- 判定
def _mk(side: str, symbol: str, status: str, reasons: list[str],
        warnings: list[str], suggested: float, price: float | None,
        ref: float | None, limit_up: float | None, limit_down: float | None,
        shares: float, cfg: FillConfig, touched: bool | None,
        depth_ok: bool | None, note: str = "", cost: float | None = None) -> FillAssessment:
    sh = _num(shares)
    amt = sh * _num(suggested)
    if side == "buy":
        f = buy_fees(amt, cfg)
        fee, delta = f["fee"], -f["cash_needed"]
        be_p = breakeven_price(suggested, sh, cfg) if sh > 0 else None
        be = (be_p / suggested - 1.0) if (be_p and suggested > 0) else None
    else:
        f = sell_fees(amt, cfg)
        fee, delta = f["fee"], f["net"]
        be_p = be = None
    return FillAssessment(
        symbol=symbol, side=side, status=status, label=STATUS_LABELS.get(status, status),
        ok=status in OK_STATUSES, reasons=list(reasons), rule_warnings=list(warnings),
        suggested_price=round(_num(suggested), 3), price=(round(price, 3) if price else price),
        reference_price=(round(ref, 3) if ref else ref),
        limit_up=limit_up, limit_down=limit_down,
        shares=sh, est_amount=round(amt, 2), est_fee=round(fee, 2),
        est_cash_delta=round(delta, 2),
        breakeven_pct=(round(be, 6) if be is not None else None),
        breakeven_price=(round(be_p, 3) if be_p else None),
        touched=touched, depth_ok=depth_ok, note=note,
    )


def _drift(price: float, ref: float | None) -> float | None:
    r = _num(ref)
    if r <= 0 or price <= 0:
        return None
    return (price / r - 1.0) * 100.0


def assess_buy(symbol: str, *, quote: Mapping | None, cfg: FillConfig, cash: float,
               shares: float | None = None, prev_close: float | None = None,
               reference: float | None = None,
               touched: tuple[float, float] | None = None,
               session: str = "open",
               opened_minutes: float | None = None) -> FillAssessment:
    """买入成交判定（严格档）。session ∈ open/pre/closed。"""
    reasons: list[str] = []
    warnings: list[str] = []
    ref = _num(reference) or _num(prev_close) or None
    price = _num((quote or {}).get("price")) if quote else 0.0
    sh = _num(shares)
    limit_up, limit_down = limit_band(prev_close, symbol)
    suggested = suggest_order_price("buy", quote, ref, limit_up, cfg)
    status = FILLABLE
    touched_flag: bool | None = None
    depth_ok: bool | None = None
    limit_locked = False          # 已封板：结论以「难成交（封板）」为准，不被漂移降级成「已错过」

    if session == "closed":
        reasons.append("非交易时段（A股 9:30-11:30 / 13:00-15:00），当前无法委托；"
                       "可先记下计划价，下一交易日按实时价确认")
        status = _worse(status, BLOCKED)
    elif session == "pre":
        reasons.append("尚未开盘（09:30 前）；集合竞价不保证成交，建议开盘后按实时价确认")
        status = _worse(status, BLOCKED)

    if not quote or price <= 0:
        reasons.append("无实时行情（多为池外股票/行情接口未覆盖），无法判定成交可行性 —— "
                       "请以券商 App 的实时价与盘口为准")
        return _mk("buy", symbol, UNKNOWN, reasons, warnings, suggested,
                   price or None, ref, limit_up, limit_down, sh, cfg, None, None,
                   note="缺实时行情")
    if is_suspended(quote):
        reasons.append("疑似停牌（今日无成交：最高=最低=0 且成交量为 0），无法买卖")
        status = _worse(status, BLOCKED)

    # 涨停封板（买不进的典型）：一字板卖一无挂单
    ask_p, ask_q = _best_level(quote.get("ask"))
    if limit_up and price >= limit_up - 0.005:
        near = "一字涨停（卖一无挂单），排队买入成交概率极低" if ask_q <= 0 \
            else "涨停价附近，封单量大时排队难成交"
        reasons.append(f"涨停封板：现价 {price:.2f} = 涨停价 {limit_up:.2f}，{near}；"
                       f"建议等开板或回落到 ¥{tick_round(limit_up * 0.98, 'buy'):.2f} 下方再看")
        status = _worse(status, HARD)
        limit_locked = True
    elif limit_up and price >= limit_up * (1 - cfg.near_limit_pct / 100):
        reasons.append(f"已接近涨停（距涨停 {(limit_up / price - 1) * 100:.2f}%），"
                       "封板时可能买不进")
        status = _worse(status, LIKELY)
    # 跌停对买入不是障碍（对手盘充足），仅提示趋势弱
    if limit_down and price <= limit_down + 0.005:
        reasons.append(f"现价已跌停（{limit_down:.2f}），买盘容易成交，但趋势极弱、注意接飞刀")

    # 价格漂移（相对参考价）
    drift = _drift(price, ref)
    tol = cfg.drift_tol_pct
    if opened_minutes is not None and opened_minutes <= cfg.open_grace_min:
        tol *= 2.0                        # 开盘跳空宽限
    if drift is not None:
        if drift > cfg.miss_tol_pct:
            reasons.append(f"已错过建议价：现价较参考价 ¥{_num(ref):.2f} 已涨 {drift:.2f}%"
                           f"（> {cfg.miss_tol_pct:.1f}%），追价风险高 —— 等回落到 "
                           f"¥{tick_round(_num(ref) * (1 + cfg.drift_tol_pct / 100), 'buy'):.2f} 附近再看")
            if not limit_locked:                 # 已封板时结论以「难成交(封板)」为准
                status = _worse(status, MISSED)
        elif drift > tol:
            reasons.append(f"现价较参考价上浮 {drift:.2f}%，需按 ¥{suggested:.2f} 改价方能成交")
            status = _worse(status, HARD)
        elif drift < -cfg.miss_tol_pct:
            reasons.append(f"现价较参考价下跌 {abs(drift):.2f}%（急跌），暂缓买入（接飞刀）")
            if not limit_locked:
                status = _worse(status, MISSED)
        elif drift < -tol:
            reasons.append(f"现价较参考价下浮 {abs(drift):.2f}%，可等企稳再按 ¥{suggested:.2f} 委托")
            status = _worse(status, LIKELY)

    # 当日已触及区间
    tr = touched or touched_range(quote)
    if tr and suggested > 0:
        low, high = tr
        edge = high * (1 + cfg.touch_tol_pct / 100)
        if suggested <= edge:
            touched_flag = True
            reasons.append(f"委托价 ¥{suggested:.2f} 落在今日区间 [{low:.2f}, {high:.2f}] 内，可成交")
        else:
            touched_flag = False
            reasons.append(f"委托价 ¥{suggested:.2f} 高于今日最高价 ¥{high:.2f}，"
                           "需价格上抬才可能成交")
            status = _worse(status, HARD)
    else:
        reasons.append("无当日区间数据，判定以现价/盘口为准")

    # 盘口深度（金额很小，一般够；仍做量级校验）
    if sh > 0 and ask_p > 0:
        if ask_q < sh * cfg.depth_ratio:
            depth_ok = False
            reasons.append(f"卖一挂单 {ask_q:.0f} 股 < 建议 {sh:.0f} 股，可能只部分成交")
            status = _worse(status, LIKELY)
        else:
            depth_ok = True

    # 规则校验（阻断级）
    lot = max(int(cfg.lot_size or 1), 1)
    if sh > 0:
        if sh < lot:
            reasons.append(f"买入须至少 {lot} 股（A股整手）")
            status = _worse(status, BLOCKED)
        elif sh % lot != 0:
            reasons.append(f"买入须为 {lot} 股整数倍（A股整手）")
            status = _worse(status, BLOCKED)
        need = buy_fees(sh * (suggested or price), cfg)["cash_needed"]
        if need > _num(cash) + 1e-6:
            reasons.append(f"资金不足：{sh:.0f} 股按 ¥{suggested:.2f} 需 ¥{need:.2f}"
                           f"（含费 ¥{buy_fees(sh * suggested, cfg)['fee']:.2f}），"
                           f"可用 ¥{_num(cash):.2f}")
            status = _worse(status, BLOCKED)
        if symbol.startswith("68") and sh < 200:
            warnings.append("科创板买入须 ≥200 股（本账户按 100 股整手记账，实盘请以券商为准）")

    # 经济性（不阻断，但影响该不该买）
    if sh > 0 and suggested > 0:
        be = (breakeven_price(suggested, sh, cfg) / suggested - 1.0) * 100
        if be > cfg.max_breakeven_pct:
            rt = buy_fees(sh * suggested, cfg)["fee"] + sell_fees(sh * suggested, cfg)["fee"]
            reasons.append(f"费用占比过高：本单往返费用约 ¥{rt:.2f}"
                           f"（佣金最低 ¥{cfg.min_commission:.2f} 起），需涨 {be:.2f}% 才回本")
        if suggested * sh < cfg.min_order_amount:
            reasons.append(f"单笔金额 ¥{suggested * sh:.0f} 低于 ¥{cfg.min_order_amount:.0f}，"
                           "费用摊薄后不划算")

    if status == FILLABLE and not reasons:
        reasons.append(f"现价 {price:.2f}，盘口有量，按 ¥{suggested:.2f} 委托可成交")
    return _mk("buy", symbol, status, reasons, warnings, suggested, price, ref,
               limit_up, limit_down, sh, cfg, touched_flag, depth_ok)


def assess_sell(symbol: str, *, quote: Mapping | None, cfg: FillConfig,
                shares: float, held: float, sellable: float,
                prev_close: float | None = None, reference: float | None = None,
                touched: tuple[float, float] | None = None, session: str = "open",
                opened_minutes: float | None = None,
                cost: float | None = None, reason: str = "") -> FillAssessment:
    """卖出成交判定（严格档）。held=持仓股数；sellable=可卖股数（T+1 后）。"""
    reasons: list[str] = []
    warnings: list[str] = []
    ref = _num(reference) or _num(prev_close) or None
    price = _num((quote or {}).get("price")) if quote else 0.0
    sh = _num(shares)
    held, sellable = _num(held), _num(sellable)
    limit_up, limit_down = limit_band(prev_close, symbol)
    suggested = suggest_order_price("sell", quote, ref, None, cfg)
    if limit_down and suggested and suggested < limit_down:
        suggested = limit_down
    status = FILLABLE
    touched_flag: bool | None = None
    depth_ok: bool | None = None
    limit_locked = False          # 已跌停封板：结论以「难成交（封板）」为准

    if session == "closed":
        reasons.append("非交易时段（A股 9:30-11:30 / 13:00-15:00），当前无法委托；"
                       "下一交易日开盘请按实时价确认（隔夜跳空风险自负）")
        status = _worse(status, BLOCKED)
    elif session == "pre":
        reasons.append("尚未开盘（09:30 前）；集合竞价不保证成交，建议开盘后按实时价确认")
        status = _worse(status, BLOCKED)

    if not quote or price <= 0:
        reasons.append("无实时行情，无法判定成交可行性 —— 请以券商 App 的实时价与盘口为准")
        return _mk("sell", symbol, UNKNOWN, reasons, warnings, suggested,
                   price or None, ref, limit_up, limit_down, sh, cfg, None, None,
                   note="缺实时行情")
    if is_suspended(quote):
        reasons.append("疑似停牌（今日无成交），无法卖出")
        status = _worse(status, BLOCKED)

    # 持仓 / T+1
    if sh > held + 1e-6:
        reasons.append(f"持仓不足：持有 {held:.0f} 股，拟卖 {sh:.0f} 股")
        status = _worse(status, BLOCKED)
    elif sh > sellable + 1e-6:
        reasons.append(f"T+1：今日买入 {max(held - sellable, 0):.0f} 股当日不可卖，"
                       f"当前可卖 {sellable:.0f} 股（拟卖 {sh:.0f} 股）")
        status = _worse(status, BLOCKED)

    # 跌停封板（卖不出的典型）
    bid_p, bid_q = _best_level(quote.get("bid"))
    if limit_down and price <= limit_down + 0.005:
        near = "一字跌停（买一无挂单），卖出难成交" if bid_q <= 0 \
            else "跌停价附近，抛压大时排队难成交"
        reasons.append(f"跌停封板：现价 {price:.2f} = 跌停价 {limit_down:.2f}，{near}；"
                       "可挂跌停价排队，或等次日再处理")
        status = _worse(status, HARD)
        limit_locked = True
    elif limit_down and price <= limit_down * (1 + cfg.near_limit_pct / 100):
        reasons.append(f"已接近跌停（距跌停 {(1 - limit_down / price) * 100:.2f}%），"
                       "封板时可能卖不出")
        status = _worse(status, LIKELY)

    # 触发价是否已错过（实盘与纸面最大的差异：你看到时已经过了）
    if ref and reason:
        drift = _drift(price, ref)
        if drift is not None and abs(drift) > cfg.drift_tol_pct:
            reasons.append(f"{reason}触发价 ¥{_num(ref):.2f}，现价 ¥{price:.2f}"
                           f"（偏离 {drift:+.2f}%）—— 触发瞬间的价格已过去，"
                           f"按 ¥{suggested:.2f} 委托或等反抽")
            if not limit_locked:              # 已跌停封板时不降级为「已错过」
                status = _worse(status, MISSED if abs(drift) > cfg.miss_tol_pct else HARD)

    # 当日已触及区间
    tr = touched or touched_range(quote)
    if tr and suggested > 0:
        low, high = tr
        if suggested >= low * (1 - cfg.touch_tol_pct / 100):
            touched_flag = True
        else:
            touched_flag = False
            reasons.append(f"委托价 ¥{suggested:.2f} 低于今日最低价 ¥{low:.2f}，"
                           "需价格下探才可能成交")
            status = _worse(status, HARD)
    else:
        reasons.append("无当日区间数据，判定以现价/盘口为准")

    # 盘口深度
    if sh > 0 and bid_p > 0:
        if bid_q < sh * cfg.depth_ratio:
            depth_ok = False
            reasons.append(f"买一挂单 {bid_q:.0f} 股 < 拟卖 {sh:.0f} 股，可能只部分成交")
            status = _worse(status, LIKELY)
        else:
            depth_ok = True

    if status == FILLABLE and not reasons:
        net = sell_fees(sh * suggested, cfg)["net"] if sh > 0 else 0.0
        reasons.append(f"现价 {price:.2f}，按 ¥{suggested:.2f}（买一价）卖出可成交，"
                       f"净收入约 ¥{net:.2f}")
    return _mk("sell", symbol, status, reasons, warnings, suggested, price, ref,
               limit_up, limit_down, sh, cfg, touched_flag, depth_ok,
               note=("含费真实盈亏需按实际成本核算" if cost is None else ""))
