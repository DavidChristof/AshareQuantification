"""第 10 步：组合交易策略回测——多股票按概率排名动态调仓（等权）。

对比：
    1. 等权全持（组合基准：首日等权买入持有）
    2. 组合策略 topN 定期调仓
    3. 组合策略 + 下跌市空仓（组合级择时）

用法：
    python scripts/10_backtest_portfolio.py
    python scripts/10_backtest_portfolio.py --top 5 --every 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                        # noqa: E402

from quant.backtest.metrics import (                       # noqa: E402
    compare, compare_risk_adjusted, plot_equity,
)
from quant.backtest.portfolio import PortfolioBacktest     # noqa: E402
from quant.config import load_config                       # noqa: E402
from quant.data.loader import load_all                     # noqa: E402
from quant.models.predict import ModelPredictor            # noqa: E402
from quant.timing.roll import roll_regime                  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=5, help="组合持仓只数（top N）")
    parser.add_argument("--every", type=int, default=5, help="每 N 个交易日调仓")
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = cfg["backtest"]
    data = load_all(cfg)

    checkpoint = cfg.resolve("results") / f"{cfg['model']['type']}_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型不存在: {checkpoint}，请先运行 scripts/03_train.py")
    predictor = ModelPredictor(checkpoint)

    # ---- 面板：收盘价 + 上涨概率 ----
    closes, probs = {}, {}
    for symbol, bars in data.items():
        if len(bars) < predictor.window + 30:
            logger.warning("%s 数据过少，跳过", symbol)
            continue
        idx = pd.to_datetime(bars["date"])
        closes[symbol] = pd.Series(bars["close"].values, index=idx)
        p = predictor.predict_probability(bars)
        p.index = pd.to_datetime(p.index)
        probs[symbol] = p

    close_panel = pd.DataFrame(closes).sort_index()
    prob_panel = pd.DataFrame(probs).reindex(close_panel.index).sort_index()
    logger.info("面板 %d 天 × %d 只，概率有效占比 %.1f%%",
                len(close_panel), close_panel.shape[1],
                prob_panel.notna().mean().mean() * 100)

    # ---- 市场代理指数 + 逐日市场状态 ----
    proxy = close_panel.mean(axis=1)
    regime = roll_regime(proxy)
    logger.info("市场状态分布: %s", regime.value_counts().to_dict())

    engine = PortfolioBacktest(
        initial_capital=bt_cfg["initial_capital"],
        commission=bt_cfg["commission"],
        slippage=bt_cfg["slippage"],
        stamp_tax=bt_cfg.get("stamp_tax", 0.0005),
        max_positions=args.top,
        position_pct=bt_cfg["position_pct"],
    )

    # 1. 等权全持（基准）
    hold = _equal_weight_hold(close_panel, engine)
    # 2. 组合策略定期调仓
    pf = engine.run(close_panel, prob_panel, rebalance_every=args.every,
                    regime_series=None, flat_on_downtrend=False)
    # 3. 组合策略 + 下跌市空仓
    pf_regime = engine.run(close_panel, prob_panel, rebalance_every=args.every,
                           regime_series=regime, flat_on_downtrend=True)

    summary = compare(
        ("组合top%d 调仓" % args.top, pf["equity"]),
        ("组合+下跌空仓", pf_regime["equity"]),
        ("等权全持", hold["equity"]),
        ("代理指数", proxy / proxy.iloc[0] * bt_cfg["initial_capital"]),
    )
    print("\n==================== 组合回测结果 ====================")
    print(summary.round(4).to_string())

    # ---- ③ 风险调整指标（以代理指数为基准：β/Alpha/IR/Treynor/Sortino/捕获率） ----
    benchmark = proxy / proxy.iloc[0] * bt_cfg["initial_capital"]
    risk_adj = compare_risk_adjusted(
        ("组合top%d 调仓" % args.top, pf["equity"]),
        ("组合+下跌空仓", pf_regime["equity"]),
        ("等权全持", hold["equity"]),
        ("代理指数", benchmark),
        benchmark=benchmark,
        risk_free=bt_cfg.get("risk_free", 0.02),
    )
    print("\n------------ 风险调整指标（基准=代理指数） ------------")
    print(risk_adj.round(4).to_string())

    n_trades = len(pf.attrs.get("trades", []))
    print(f"\n组合策略调仓总次数: {n_trades} 笔（{args.every} 天调仓 × {args.top} 只）")

    results_dir = cfg.resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(results_dir / "backtest_portfolio_summary.csv", encoding="utf-8-sig")
    risk_adj.to_csv(results_dir / "backtest_portfolio_risk_adjusted.csv",
                    encoding="utf-8-sig")
    plot_equity(
        {"组合top%d 调仓" % args.top: pf["equity"],
         "组合+下跌空仓": pf_regime["equity"],
         "等权全持": hold["equity"],
         "代理指数": proxy / proxy.iloc[0] * bt_cfg["initial_capital"]},
        title=f"组合策略回测（top{args.top} · {args.every}天调仓，归一化）",
        save_path=str(results_dir / "backtest_portfolio_equity.png"))
    logger.info("净值图已保存到 %s", results_dir / "backtest_portfolio_equity.png")


def _equal_weight_hold(close_panel: pd.DataFrame, engine: PortfolioBacktest
                       ) -> pd.DataFrame:
    """基准：首日对全部股票等权买入，持有到结束。"""
    dates = close_panel.index
    day0 = close_panel.iloc[0]
    valid = day0.dropna()
    if valid.empty:
        raise RuntimeError("首日无有效价格")
    per = engine.initial_capital * engine.position_pct / len(valid)
    shares = {}
    cash = float(engine.initial_capital)
    for sym, price0 in valid.items():
        buy_price = float(price0) * (1 + engine.slippage)
        sh = per / buy_price
        fee = sh * buy_price * engine.commission
        if cash >= sh * buy_price + fee:
            cash -= sh * buy_price + fee
            shares[sym] = sh

    rows = []
    for date in dates:
        mv = sum(sh * float(close_panel.loc[date, s])
                 for s, sh in shares.items() if pd.notna(close_panel.loc[date, s]))
        rows.append({"date": date, "cash": cash, "holdings": mv,
                     "equity": cash + mv, "n_positions": len(shares)})
    return pd.DataFrame(rows).set_index("date")


if __name__ == "__main__":
    main()
