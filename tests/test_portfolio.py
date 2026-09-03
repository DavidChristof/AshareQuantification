"""组合交易策略回测 单元测试：topN 选股 / 定期调仓 / 下跌空仓 / 费用。

运行：python -m pytest tests/test_portfolio.py -v  或  python tests/test_portfolio.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quant.backtest.portfolio import PortfolioBacktest   # noqa: E402


def _idx(days):
    return pd.date_range("2026-01-01", periods=days, freq="B")


def test_select_top_n_only():
    """只买入上涨概率最高的 top N。"""
    idx = _idx(10)
    close = pd.DataFrame({s: 100.0 for s in "ABC"}, index=idx)
    prob = pd.DataFrame({"A": 0.9, "B": 0.8, "C": 0.7}, index=idx)
    eng = PortfolioBacktest(max_positions=2, initial_capital=100_000.0)
    res = eng.run(close, prob, rebalance_every=10)
    assert res["n_positions"].iloc[-1] == 2
    buys = {t["symbol"] for t in res.attrs["trades"] if t["side"] == "buy"}
    assert buys == {"A", "B"}


def test_rebalance_switch_stocks():
    """概率排名变化 → 定期调仓换股（卖 A 买 C）。"""
    days = 20
    idx = _idx(days)
    close = pd.DataFrame({s: 100.0 for s in "ABC"}, index=idx)
    prob = pd.DataFrame(index=idx, columns=list("ABC"), dtype=float)
    prob.iloc[:10] = [0.9, 0.8, 0.7]   # 前半：A 最高
    prob.iloc[10:] = [0.7, 0.8, 0.9]   # 后半：C 最高
    eng = PortfolioBacktest(max_positions=2, initial_capital=100_000.0)
    res = eng.run(close, prob, rebalance_every=10)
    trades = res.attrs["trades"]
    sells = {(t["symbol"], t["side"]) for t in trades}
    assert ("A", "sell") in sells          # A 掉出前 2 → 卖出
    assert ("C", "buy") in sells           # C 进前 2 → 买入
    assert res["n_positions"].iloc[-1] == 2


def test_flat_on_downtrend():
    """下跌市整体空仓（组合级择时）。"""
    idx = _idx(15)
    close = pd.DataFrame({s: 100.0 for s in "ABC"}, index=idx)
    prob = pd.DataFrame({"A": 0.9, "B": 0.8, "C": 0.7}, index=idx)
    regime = pd.Series("downtrend", index=idx)
    eng = PortfolioBacktest(max_positions=2, initial_capital=100_000.0)
    res = eng.run(close, prob, rebalance_every=5,
                  regime_series=regime, flat_on_downtrend=True)
    assert (res["n_positions"] == 0).all()
    assert abs(res["equity"].iloc[-1] - 100_000.0) < 1e-6   # 空仓不亏不赚


def test_equity_after_price_move():
    """组合净值随持仓市值变化。"""
    idx = _idx(10)
    # A 从 100 涨到 120，B 从 100 跌到 80
    close = pd.DataFrame({"A": np.linspace(100, 120, 10),
                          "B": np.linspace(100, 80, 10),
                          "C": np.full(10, 100.0)}, index=idx)
    prob = pd.DataFrame({"A": 0.9, "B": 0.8, "C": 0.7}, index=idx)
    eng = PortfolioBacktest(max_positions=2, initial_capital=100_000.0)
    res = eng.run(close, prob, rebalance_every=10)
    final = res["equity"].iloc[-1]
    # 持有 A(涨)、B(跌) 各约半仓，最终净值应在初始附近且 > 纯 B 跌幅情形
    assert 80_000 < final < 130_000
    # 首日买入后出现交易费用 → 净值略低于按市值直接算的
    assert res["equity"].iloc[0] < 100_000.0


if __name__ == "__main__":
    tests = [test_select_top_n_only, test_rebalance_switch_stocks,
             test_flat_on_downtrend, test_equity_after_price_move]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
