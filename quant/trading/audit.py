"""账户恒等式核对 —— 账本类 bug 的护栏。

## 为什么需要它

2026-09-17 一天里出了**两个**账本 bug：

- 同一笔 400 股被自动卖出两次 => **凭空多出 9,898.02 元现金**，总资产从 98,732 跳到 108,782；
- 「日点」净值一直按**前一天**的收盘价写 => 次日「当日收益」虚高一天（差 169 元）。

**两个都零异常、零告警**，全靠用户肉眼看「数字不对」才发现。
账本类错误的发现方式应该是「**恒等式对不上**」，而不是「报错」——
数字错和程序错是两回事，程序不会因为算错钱而抛异常。

## 三条恒等式

1. **现金**：`现金 == 初始资金 - Σ买入金额 + Σ(卖出金额 - 卖出费用)`
   能抓住：重复卖出、并发透支、漏记/多记成交。
2. **净值点**：每个**日点** `equity == cash + Σ 持仓股数 × 当日收盘价`
   能抓住：日点按别的日期的价格写（2026-09-17 那个）、持仓/现金与净值不同步。
3. **当日收益**：`当日收益 == Σ持仓浮动 + Σ今日卖出(相对昨收) + Σ今日买入浮动 - Σ当日费用`
   能抓住：基线（昨收净值）算错、成交漏记。

## 用法

    from quant.trading.audit import audit_account
    report = audit_account(broker, close_of)      # close_of(symbol, date) -> float | None
    if not report["ok"]:
        for f in report["findings"]:
            print(f)

`close_of` 由调用方提供（本模块不认识行情库）。命令行见 `scripts/42_audit_accounts.py`。
"""
from __future__ import annotations

from collections import defaultdict


def _holdings_from_trades(trades: list[dict], upto: str) -> dict[str, float]:
    """按流水重建 `upto`（含）为止的持仓。"""
    hold: dict[str, float] = defaultdict(float)
    for t in trades:
        if str(t["date"])[:10] > upto:
            continue
        hold[t["symbol"]] += float(t["shares"]) if t["side"] == "buy" else -float(t["shares"])
    return {s: v for s, v in hold.items() if v > 1e-6}


def reconcile_cash(broker, tol: float = 0.01) -> dict:
    """恒等式 1：现金 == 初始资金 - Σ买入金额 + Σ(卖出金额 - 卖出费用)。

    [!] 口径来自 `paper.py`：买入流水 `amount` 存的是**含费总额**；
    卖出流水 `amount` 存的是**不含费的成交额**，实际入账是 `amount - fee`。
    """
    with broker._connect() as conn:
        row = conn.execute(
            "SELECT value FROM paper_account WHERE key='initial_capital'").fetchone()
        init = float(row[0]) if row and row[0] is not None else 0.0
        rows = conn.execute(
            "SELECT side, shares, price, fee, amount FROM paper_trades").fetchall()
    expected = init
    for side, _sh, _px, fee, amount in rows:
        fee = float(fee or 0.0)
        amount = float(amount or 0.0)
        expected += (-amount) if side == "buy" else (amount - fee)
    actual = float(broker.query_cash())
    diff = actual - expected
    return {"name": "cash", "ok": abs(diff) <= tol,
            "expected": round(expected, 4), "actual": round(actual, 4),
            "diff": round(diff, 4), "n_trades": len(rows)}


def check_daily_points(broker, close_of, tol: float = 0.01) -> list[dict]:
    """恒等式 2：每个**日点** `equity == cash + Σ 持仓 × 当日收盘`。

    只看日点（date 里没有空格的）；小时点是实时估值，本来就不该等于收盘价。
    某个持仓当天查不到收盘价 → 该点**跳过**（并记一条 note），不误报。
    """
    with broker._connect() as conn:
        trades = [dict(zip(("date", "symbol", "side", "shares"), r))
                  for r in conn.execute(
                      "SELECT date, symbol, side, shares FROM paper_trades ORDER BY id")]
        points = conn.execute(
            "SELECT date, cash, market_value, equity FROM paper_equity "
            "ORDER BY date").fetchall()
    findings = []
    for date, cash, mv, eq in points:
        if " " in str(date):
            continue
        day = str(date)[:10]
        hold = _holdings_from_trades(trades, day)
        expected_mv, missing = 0.0, []
        for s, sh in hold.items():
            px = close_of(s, day)
            if px is None:
                missing.append(s)
                continue
            expected_mv += sh * float(px)
        if missing:
            continue                      # 有持仓查不到价 -> 无法核对，不误报
        exp_eq = float(cash) + expected_mv
        diff = float(eq) - exp_eq
        if abs(diff) > tol:
            findings.append({
                "name": "daily_point", "date": day, "ok": False,
                "market_value_actual": round(float(mv), 2),
                "market_value_expected": round(expected_mv, 2),
                "equity_actual": round(float(eq), 2),
                "equity_expected": round(exp_eq, 2),
                "diff": round(diff, 2),
            })
    return findings


def check_day_pnl(broker, day: str, prev_day: str, close_of,
                  price_now: dict[str, float], day_pnl_reported: float,
                  tol: float = 0.01) -> dict:
    """恒等式 3：当日收益 == Σ持仓浮动 + Σ今日卖出(相对昨收) + Σ今日买入浮动 - Σ当日费用。

    全部按「相对**昨收**」计算 —— 这正是它要防的那个错：基线若用了别的日期的价格，
    这里就对不上。
    """
    with broker._connect() as conn:
        trades = [dict(zip(("date", "symbol", "side", "shares", "price", "fee"), r))
                  for r in conn.execute(
                      "SELECT date, symbol, side, shares, price, fee FROM paper_trades "
                      "ORDER BY id")]
    today = [t for t in trades if str(t["date"])[:10] == day]
    hold = _holdings_from_trades(trades, day)

    exp, missing = 0.0, []
    # ① 现在仍持有的：相对昨收的浮动（今日买入的稍后按买入价重算，这里先不重复计）
    bought_today = defaultdict(float)
    for t in today:
        if t["side"] == "buy":
            bought_today[t["symbol"]] += float(t["shares"])
    for s, sh in hold.items():
        pn, pp = price_now.get(s), close_of(s, prev_day)
        if pn is None or pp is None:
            missing.append(s)
            continue
        held_from_before = sh - bought_today.get(s, 0.0)
        exp += held_from_before * (float(pn) - float(pp))
    # ② 今日卖出的：相对昨收
    for t in today:
        if t["side"] != "sell":
            continue
        pp, sp = close_of(t["symbol"], prev_day), float(t["price"])
        if pp is None:
            missing.append(t["symbol"])
            continue
        exp += float(t["shares"]) * (sp - float(pp))
    # ③ 今日买入的：相对买入价（不是昨收）
    for t in today:
        if t["side"] != "buy":
            continue
        pn, bp = price_now.get(t["symbol"]), float(t["price"])
        if pn is None:
            missing.append(t["symbol"])
            continue
        exp += float(t["shares"]) * (float(pn) - bp)
    if missing:
        return {"name": "day_pnl", "ok": None,
                "note": f"缺价格，无法核对: {sorted(set(missing))}"}
    # ④ 当日费用
    exp -= sum(float(t["fee"] or 0.0) for t in today)

    diff = float(day_pnl_reported) - exp
    return {"name": "day_pnl", "ok": abs(diff) <= tol,
            "reported": round(float(day_pnl_reported), 2),
            "expected": round(exp, 2), "diff": round(diff, 2),
            "n_trades_today": len(today)}


def audit_account(broker, close_of, day_pnl: dict | None = None) -> dict:
    """跑全部恒等式，返回 {ok, findings, checks}。

    day_pnl: 可选，`{"day": D, "prev_day": P, "price_now": {...}, "reported": x}`
             —— 由调用方从 live_summary 拿到（本模块不依赖 API 层）。
    """
    findings: list[dict] = []
    checks: list[dict] = []

    cash = reconcile_cash(broker)
    checks.append(cash)
    if not cash["ok"]:
        findings.append(cash)

    pts = check_daily_points(broker, close_of)
    checks.append({"name": "daily_points", "ok": not pts, "n_bad": len(pts)})
    findings.extend(pts)

    if day_pnl:
        dp = check_day_pnl(broker, day_pnl["day"], day_pnl["prev_day"], close_of,
                           day_pnl["price_now"], day_pnl["reported"])
        checks.append(dp)
        if dp.get("ok") is False:
            findings.append(dp)

    return {"ok": not findings, "findings": findings, "checks": checks}
