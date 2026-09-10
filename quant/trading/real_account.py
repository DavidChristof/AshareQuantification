"""实盘账本（RealBroker）：人工在券商成交后「回报记账」，并按成交流水可重建。

与 `PaperBroker` 的关系：**只新增能力，不改其行为**——
    - 复用 buy/sell/query_positions/query_cash/snapshot_equity/apply_stop_rules 等全部记账逻辑；
    - 新增 `real_order_log` 表：记录「下过的单 / 建议 / 是否成交」，**未成交也留痕**
      （这正是「着重考虑成交成功与否」需要的对照数据）；
    - `record_execution()`：按人工回报的成交价与**实际手续费**记账（fee_override）；
    - `sellable_shares()`：T+1 可卖量（今日买入不可卖）；
    - `delete_trade()` / `rebuild_from_trades()`：误录修正 —— 删掉错误流水后**全量重放**，
      cash 与持仓从流水重新推导，永远自洽（也是坏数据的修复工具）。

⚠️ 本模块**不会**、也无法自动下单：没有任何券商接口，只对人工回报的成交做记账。
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from .base import TradeResult
from .paper import PaperBroker

logger = logging.getLogger(__name__)

_REAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS real_order_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT,          -- 记录时间（本地）
    date          TEXT,          -- 成交/委托日期
    symbol        TEXT,
    side          TEXT,          -- buy / sell
    shares        REAL,
    price         REAL,          -- 实际成交价（未成交时为拟委托价）
    status        TEXT,          -- filled / unfilled / void
    advice_price  REAL,          -- 当时的建议委托价
    advice_status TEXT,          -- 当时的成交判定（fillable/hard/...）
    reason        TEXT,          -- 未成交/放弃原因
    remark        TEXT
);
"""


class RealBroker(PaperBroker):
    """实盘账户记账器（现金/持仓/流水/订单留痕），人工回报成交。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_real_schema()

    def _init_real_schema(self):
        with self._connect() as conn:
            conn.executescript(_REAL_SCHEMA)

    # ---------------------------------------------------------- 成交回报
    def record_execution(self, symbol: str, side: str, shares: float, price: float,
                         date: str, fee: float | None = None, remark: str = "",
                         advice_price: float | None = None,
                         advice_status: str | None = None) -> TradeResult:
        """按人工回报的成交记账（可带券商实际手续费 fee）。"""
        fn = self.buy if side == "buy" else self.sell
        r = fn(symbol, shares, price, date, remark=remark, fee_override=fee)
        if r.success:
            self.log_order(symbol, side, shares, price, date, status="filled",
                           advice_price=advice_price, advice_status=advice_status,
                           remark=remark)
        return r

    def log_order(self, symbol: str, side: str, shares: float, price: float,
                  date: str, status: str = "unfilled", reason: str = "",
                  advice_price: float | None = None,
                  advice_status: str | None = None, remark: str = "") -> int:
        """记一条订单留痕（未成交/放弃也记）。返回日志 id。"""
        from datetime import datetime
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO real_order_log (ts,date,symbol,side,shares,price,status,"
                "advice_price,advice_status,reason,remark) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), date, symbol, side,
                 shares, price, status, advice_price, advice_status, reason, remark))
            conn.commit()
            return int(cur.lastrowid)

    # ---------------------------------------------------------- 查询
    def orders(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,ts,date,symbol,side,shares,price,status,advice_price,"
                "advice_status,reason,remark FROM real_order_log "
                "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ("id", "ts", "date", "symbol", "side", "shares", "price", "status",
                "advice_price", "advice_status", "reason", "remark")
        return [dict(zip(keys, r)) for r in rows]

    def trade_history_with_id(self, limit: int = 100) -> list[dict]:
        """成交流水（带 id，供误录删除）。基类 trade_history 不返回 id。"""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id,date,symbol,side,shares,price,fee,amount,remark "
                "FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        keys = ("id", "date", "symbol", "side", "shares", "price", "fee",
                "amount", "remark")
        return [dict(zip(keys, r)) for r in rows]

    def sellable_shares(self, symbol: str, date: str) -> float:
        """T+1：可卖股数 = 持仓 − 当日买入。"""
        with self._connect() as conn:
            pos = conn.execute(
                "SELECT shares FROM paper_positions WHERE symbol=?", (symbol,)).fetchone()
            held = float(pos[0]) if pos else 0.0
            today_buy = float(conn.execute(
                "SELECT COALESCE(SUM(shares),0) FROM paper_trades "
                "WHERE date=? AND symbol=? AND side='buy'", (date, symbol)).fetchone()[0])
        return max(held - today_buy, 0.0)

    # ---------------------------------------------------------- 更正
    def void_order(self, order_id: int) -> bool:
        """把一条订单留痕标记为作废（不改动资金/持仓）。"""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE real_order_log SET status='void' WHERE id=?", (order_id,))
            conn.commit()
            return cur.rowcount > 0

    def delete_trade(self, trade_id: int) -> bool:
        """删除一条错误流水（随后应调用 rebuild_from_trades 重建）。"""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM paper_trades WHERE id=?", (trade_id,))
            conn.commit()
            return cur.rowcount > 0

    def rebuild_from_trades(self) -> dict:
        """按现存成交流水**全量重放**，重建 cash 与持仓（自洽、可作修复工具）。

        口径：买入 `amount` 已含费用 → cash -= amount；卖出 `amount` 为成交额 →
        cash += amount - fee。持仓成本按移动加权平均重算。
        """
        with self._connect() as conn:
            init_row = conn.execute(
                "SELECT value FROM paper_account WHERE key='initial_capital'").fetchone()
            cash = float(init_row[0]) if init_row and init_row[0] is not None else 0.0
            rows = conn.execute(
                "SELECT id,date,symbol,side,shares,price,fee,amount FROM paper_trades "
                "ORDER BY id").fetchall()
            old_max = {r[0]: float(r[1] or 0) for r in conn.execute(
                "SELECT symbol, max_price FROM paper_positions").fetchall()}

        pos: dict[str, list] = {}
        for _id, _d, sym, side, sh, px, fee, amt in rows:
            sh, px, fee, amt = float(sh), float(px), float(fee or 0), float(amt)
            if side == "buy":
                cash -= amt
                cur = pos.setdefault(sym, [0.0, 0.0])
                new_sh = cur[0] + sh
                cur[1] = (cur[0] * cur[1] + sh * px) / new_sh if new_sh else 0.0
                cur[0] = new_sh
            else:
                cash += amt - fee
                cur = pos.setdefault(sym, [0.0, 0.0])
                cur[0] -= sh
                if cur[0] <= 1e-9:
                    pos.pop(sym, None)

        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute("DELETE FROM paper_positions")
            conn.execute("UPDATE paper_account SET value=? WHERE key='cash'", (cash,))
            for sym, (sh, cost) in pos.items():
                if sh <= 0:
                    continue
                conn.execute(
                    "INSERT INTO paper_positions (symbol,shares,avg_cost,max_price) "
                    "VALUES (?,?,?,?)", (sym, sh, cost, old_max.get(sym, 0.0) or cost))
            conn.commit()
        logger.info("实盘账本已按流水重建：cash=%.2f，持仓 %d 只", cash, len(pos))
        return {"cash": round(cash, 2),
                "positions": {s: round(v[0], 4) for s, v in pos.items()}}
