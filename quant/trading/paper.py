"""纸面交易（Paper Trading）实现：用真实行情 + 模拟撮合，跟踪「如果实盘会怎样」。

数据持久化到 SQLite，表结构：
    paper_account   (key, value)            账户资金
    paper_positions (symbol, shares, avg_cost)  持仓
    paper_trades    (date, symbol, side, shares, price, fee, amount)  成交记录
    paper_equity    (date, cash, market_value, equity)  每日净值快照
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date
from pathlib import Path

from .base import Broker, Position, TradeResult

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS paper_account (
    key   TEXT PRIMARY KEY,
    value REAL
);
CREATE TABLE IF NOT EXISTS paper_positions (
    symbol   TEXT PRIMARY KEY,
    shares   REAL,
    avg_cost REAL,
    max_price REAL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS paper_trades (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    date   TEXT,
    symbol TEXT,
    side   TEXT,
    shares REAL,
    price  REAL,
    fee    REAL,
    amount REAL,
    remark TEXT
);
CREATE TABLE IF NOT EXISTS paper_equity (
    date         TEXT PRIMARY KEY,
    cash         REAL,
    market_value REAL,
    equity       REAL
);
"""


class PaperBroker(Broker):
    """模拟券商：状态全部持久化到 SQLite，重启不丢失。"""

    def __init__(self, db_path: str | Path, initial_capital: float = 100_000.0,
                 commission: float = 0.0003, slippage: float = 0.0002,
                 stamp_tax: float = 0.0005, lot_size: int = 1):
        self.db_path = Path(db_path)
        self.commission = commission
        self.slippage = slippage
        self.stamp_tax = stamp_tax      # 印花税（卖出单边，A股 0.05%）
        self.lot_size = lot_size        # 整手限制：>1 时买入必须是其整数倍（A股 100 股）
        self._initial_capital = initial_capital
        self._init_schema()
        self._ensure_initialized(initial_capital)

    # ---------- 内部工具 ----------
    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_schema(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            # 兼容旧库：给已有表补列（忽略“列已存在”错误）
            for alter in (
                "ALTER TABLE paper_positions ADD COLUMN max_price REAL DEFAULT 0",
                "ALTER TABLE paper_trades ADD COLUMN remark TEXT",
            ):
                try:
                    conn.execute(alter)
                except sqlite3.OperationalError:
                    pass

    def _ensure_initialized(self, initial_capital: float):
        """首次使用时注入初始资金。"""
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM paper_account WHERE key='cash'").fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO paper_account (key, value) VALUES ('cash', ?)", (initial_capital,))
                conn.execute(
                    "INSERT INTO paper_account (key, value) VALUES ('initial_capital', ?)",
                    (initial_capital,))
                logger.info("纸面账户初始化完成，初始资金 %.2f", initial_capital)

    # ---------- Broker 接口实现 ----------
    def query_cash(self) -> float:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM paper_account WHERE key='cash'").fetchone()
        return float(row[0]) if row else 0.0

    def query_positions(self) -> list[Position]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT symbol, shares, avg_cost FROM paper_positions WHERE shares > 0"
            ).fetchall()
        return [Position(symbol=r[0], shares=float(r[1]), avg_cost=float(r[2])) for r in rows]

    def buy(self, symbol: str, shares: float, price: float, date: str,
            remark: str = "") -> TradeResult:
        if shares <= 0:
            return TradeResult(symbol, "buy", 0, price, 0, 0, False, "买入数量必须为正")
        if self.lot_size > 1 and shares % self.lot_size != 0:
            return TradeResult(symbol, "buy", 0, price, 0, 0, False,
                               f"买入必须为 {self.lot_size} 股整数倍（A股整手）")
        buy_price = price * (1 + self.slippage)      # 滑点抬高买价
        amount = shares * buy_price
        fee = amount * self.commission
        total_cost = amount + fee

        cash = self.query_cash()
        if total_cost > cash + 1e-6:
            return TradeResult(symbol, "buy", shares, buy_price, fee, total_cost,
                               False, f"资金不足: 需 {total_cost:.2f} 可用 {cash:.2f}")

        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute("UPDATE paper_account SET value = value - ? WHERE key='cash'", (total_cost,))
            # 更新持仓（移动加权平均成本）
            pos = conn.execute(
                "SELECT shares, avg_cost FROM paper_positions WHERE symbol=?", (symbol,)).fetchone()
            if pos:
                old_shares, old_cost = float(pos[0]), float(pos[1])
                new_shares = old_shares + shares
                new_cost = (old_shares * old_cost + shares * buy_price) / new_shares
                conn.execute(
                    "UPDATE paper_positions SET shares=?, avg_cost=? WHERE symbol=?",
                    (new_shares, new_cost, symbol))
            else:
                conn.execute(
                    "INSERT INTO paper_positions (symbol, shares, avg_cost, max_price) "
                    "VALUES (?,?,?,?)",
                    (symbol, shares, buy_price, buy_price))
            conn.execute(
                "INSERT INTO paper_trades (date, symbol, side, shares, price, fee, amount, remark) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (date, symbol, "buy", shares, buy_price, fee, total_cost, remark))
            conn.commit()

        logger.info("[%s] 买入 %s %.0f 股 @ %.2f (费 %.2f)", date, symbol, shares, buy_price, fee)
        return TradeResult(symbol, "buy", shares, buy_price, fee, total_cost)

    def sell(self, symbol: str, shares: float, price: float, date: str,
             remark: str = "") -> TradeResult:
        if shares <= 0:
            return TradeResult(symbol, "sell", 0, price, 0, 0, False, "卖出数量必须为正")
        with self._connect() as conn:
            pos = conn.execute(
                "SELECT shares, avg_cost FROM paper_positions WHERE symbol=?", (symbol,)).fetchone()
            if pos is None or float(pos[0]) < shares - 1e-6:
                return TradeResult(symbol, "sell", shares, price, 0, 0, False, "持仓不足")
            # T+1 规则：今日买入的份额当日不可卖出
            today_buy = float(conn.execute(
                "SELECT COALESCE(SUM(shares),0) FROM paper_trades "
                "WHERE date=? AND symbol=? AND side='buy'",
                (date, symbol)).fetchone()[0])
            if shares > (float(pos[0]) - today_buy) + 1e-6:
                return TradeResult(symbol, "sell", shares, price, 0, 0, False,
                                   f"T+1：今日已买入 {today_buy:.0f} 股，当日不能卖出")
        sell_price = price * (1 - self.slippage)     # 滑点压低卖价
        proceeds = shares * sell_price
        fee = proceeds * (self.commission + self.stamp_tax)   # 佣金 + 印花税（卖出单边）
        net = proceeds - fee

        with self._connect() as conn:
            pos = conn.execute(
                "SELECT shares, avg_cost FROM paper_positions WHERE symbol=?", (symbol,)).fetchone()
            if pos is None or float(pos[0]) < shares - 1e-6:
                return TradeResult(symbol, "sell", shares, sell_price, fee, proceeds,
                                   False, "持仓不足")
            conn.execute("BEGIN")
            conn.execute("UPDATE paper_account SET value = value + ? WHERE key='cash'", (net,))
            remain = float(pos[0]) - shares
            if remain < 1e-6:
                conn.execute("DELETE FROM paper_positions WHERE symbol=?", (symbol,))
            else:
                conn.execute("UPDATE paper_positions SET shares=? WHERE symbol=?", (remain, symbol))
            conn.execute(
                "INSERT INTO paper_trades (date, symbol, side, shares, price, fee, amount, remark) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (date, symbol, "sell", shares, sell_price, fee, proceeds, remark))
            conn.commit()

        logger.info("[%s] 卖出 %s %.0f 股 @ %.2f (费 %.2f) %s",
                    date, symbol, shares, sell_price, fee, remark)
        return TradeResult(symbol, "sell", shares, sell_price, fee, proceeds)

    # ---------- 止盈止损 ----------
    def _update_max_price(self, symbol: str, price: float):
        """更新持仓期间最高价（用于移动止损）。"""
        with self._connect() as conn:
            conn.execute(
                "UPDATE paper_positions SET max_price = MAX(COALESCE(max_price,0), ?) "
                "WHERE symbol=?", (price, symbol))

    def _get_max_price(self, symbol: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT max_price FROM paper_positions WHERE symbol=?", (symbol,)).fetchone()
        return float(row[0]) if row and row[0] else 0.0

    def apply_stop_rules(self, date: str, prices: dict[str, float],
                         stop_loss_pct: float = 0.08, take_profit_pct: float = 0.15,
                         trailing_pct: float | None = None,
                         vol: dict | None = None, vol_cfg: dict | None = None
                         ) -> list[dict]:
        """止盈止损检查：持仓触发条件则自动卖出。

        顺序：止损 > 止盈 > 移动止损（同时满足时止损优先）。

        当传入 ``vol``（每只股票的 ATR 信息，来自 build_vol_map）和 ``vol_cfg``
        （动态参数，来自 vol_cfg_from_risk）时，改用「按波动率动态」百分比：
        高波动股票止损/止盈线更宽，低波动股票更窄；否则用固定百分比。

        Returns: 触发的卖出列表 [{symbol, reason, price}]。
        """
        from ..risk.volatility import dynamic_pcts

        vol_cfg = vol_cfg or {}
        triggered = []
        for pos in self.query_positions():
            price = prices.get(pos.symbol)
            if not price or price <= 0:
                continue
            self._update_max_price(pos.symbol, price)      # 记录持仓最高价
            cost = pos.avg_cost

            # 默认用固定百分比
            sl, tp = stop_loss_pct, take_profit_pct
            trail = trailing_pct
            # 动态波动率模式：按该股 ATR 计算个性化百分比
            v = (vol or {}).get(pos.symbol)
            if v and v.get("atr", 0) > 0 and cost > 0:
                dyn = dynamic_pcts(
                    v, cost, high=self._get_max_price(pos.symbol),
                    stop_mult=vol_cfg.get("atr_stop_mult", 2.5),
                    take_mult=vol_cfg.get("atr_take_mult", 3.5),
                    trail_mult=vol_cfg.get("atr_trailing_mult", 2.5),
                    min_pct=vol_cfg.get("vol_min_pct", 0.03),
                    max_pct=vol_cfg.get("vol_max_pct", 0.15),
                    take_min_pct=vol_cfg.get("take_min_pct", 0.05),
                    take_max_pct=vol_cfg.get("take_max_pct", 0.30),
                )
                if dyn["stop_pct"]:
                    sl, tp = dyn["stop_pct"], dyn["take_pct"]
                    trail = dyn["trail_pct"] if vol_cfg.get("trailing_enabled", True) else None

            reason = None
            if price <= cost * (1 - sl):
                reason = f"止损：现价{price:.2f}≤成本{cost:.2f}×{1 - sl:.1%}"
            elif price >= cost * (1 + tp):
                reason = f"止盈：现价{price:.2f}≥成本{cost:.2f}×{1 + tp:.1%}"
            elif trail:
                high = self._get_max_price(pos.symbol)
                if high > cost and price <= high * (1 - trail):
                    reason = f"移动止损：从高点{high:.2f}回撤{trail:.1%}"
            if reason:
                r = self.sell(pos.symbol, pos.shares, price, date, remark=reason)
                if r.success:
                    triggered.append({"symbol": pos.symbol, "reason": reason,
                                      "price": round(price, 2)})
        return triggered

    # ---------- 扩展：净值快照与历史 ----------
    def reset(self, initial_capital: float | None = None) -> None:
        """重置账户：清空持仓/成交/净值，资金恢复初始资金。

        供开市前清理试验数据使用（脚本 scripts/09_reset_accounts.py）。
        """
        init = initial_capital if initial_capital is not None else self._initial_capital
        with self._connect() as conn:
            conn.execute("DELETE FROM paper_positions")
            conn.execute("DELETE FROM paper_trades")
            conn.execute("DELETE FROM paper_equity")
            conn.execute("UPDATE paper_account SET value=? WHERE key='cash'", (init,))
            conn.execute(
                "UPDATE paper_account SET value=? WHERE key='initial_capital'", (init,))
        logger.info("纸面账户已重置，初始资金 %.2f", init)

    def snapshot_equity(self, date: str, latest_prices: dict[str, float]) -> float:
        """按最新收盘价计算总资产并写入净值快照，返回 equity。"""
        cash = self.query_cash()
        market_value = 0.0
        for pos in self.query_positions():
            price = latest_prices.get(pos.symbol)
            if price:
                market_value += pos.shares * price
        equity = cash + market_value
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO paper_equity (date, cash, market_value, equity) "
                "VALUES (?,?,?,?)",
                (date, cash, market_value, equity))
        return equity

    def equity_history(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, cash, market_value, equity FROM paper_equity ORDER BY date"
            ).fetchall()
        return [{"date": r[0], "cash": r[1], "market_value": r[2], "equity": r[3]} for r in rows]

    def trade_history(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT date, symbol, side, shares, price, fee, amount, remark "
                "FROM paper_trades ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"date": r[0], "symbol": r[1], "side": r[2], "shares": r[3],
                 "price": r[4], "fee": r[5], "amount": r[6], "remark": r[7]} for r in rows]

    def account_summary(self) -> dict:
        cash = self.query_cash()
        with self._connect() as conn:
            init = conn.execute(
                "SELECT value FROM paper_account WHERE key='initial_capital'").fetchone()
            latest = conn.execute(
                "SELECT equity FROM paper_equity ORDER BY date DESC LIMIT 1").fetchone()
        initial = float(init[0]) if init else cash
        equity = float(latest[0]) if latest else cash
        return {
            "cash": cash,
            "equity": equity,
            "initial_capital": initial,
            "total_return": equity / initial - 1 if initial else 0.0,
        }

    def live_summary(self, latest_prices: dict[str, float]) -> dict:
        """按最新价格（盘中实时价）估算总资产，不回写历史净值快照。

        Args:
            latest_prices: {symbol: 现价}，缺失的持仓回退到成本价。
        """
        cash = self.query_cash()
        mv = 0.0
        for pos in self.query_positions():
            p = latest_prices.get(pos.symbol) or pos.avg_cost
            if p and p > 0:
                mv += pos.shares * p
        equity = cash + mv
        with self._connect() as conn:
            init = conn.execute(
                "SELECT value FROM paper_account WHERE key='initial_capital'").fetchone()
        initial = float(init[0]) if init else cash
        # 当日收益口径：相对「上一交易日收盘」而非本金（净值历史里 date < 今天的最后一个点）
        today = date.today().isoformat()
        prev_close = None
        for _r in self.equity_history():
            if str(_r["date"])[:10] < today:
                prev_close = _r["equity"]
        base = float(prev_close) if prev_close is not None else float(initial)
        day_pnl = equity - base
        return {
            "cash": round(cash, 2),
            "market_value": round(mv, 2),
            "equity": round(equity, 2),
            "initial_capital": initial,
            "total_return": equity / initial - 1 if initial else 0.0,
            "prev_close_equity": round(base, 2),
            "day_pnl": round(day_pnl, 2),
            "day_return": round(day_pnl / base, 4) if base else 0.0,
        }
