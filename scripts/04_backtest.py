"""第 4 步：回测——用训练好的模型对每只股票做信号回测。

用法：
    python scripts/04_backtest.py                          # 用默认 lstm_model.pt
    python scripts/04_backtest.py --model transformer      # 用 transformer 模型
    python scripts/04_backtest.py --symbol 600519          # 只回测单只股票
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                        # noqa: E402
import numpy as np                                         # noqa: E402

from quant.backtest.engine import BacktestEngine           # noqa: E402
from quant.backtest.metrics import (compare, compare_risk_adjusted,
                                    plot_equity)          # noqa: E402
from quant.config import load_config                       # noqa: E402
from quant.data.loader import load_all                     # noqa: E402
from quant.features.technical import compute_technical     # noqa: E402
from quant.models.predict import ModelPredictor            # noqa: E402
from quant.risk.volatility import vol_cfg_from_risk        # noqa: E402
from quant.timing.roll import roll_regime, roll_timing_signal  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "transformer"])
    parser.add_argument("--symbol", default=None, help="只回测指定股票")
    args = parser.parse_args()

    cfg = load_config()
    bt_cfg = cfg["backtest"]
    data = load_all(cfg)

    # 加载训练好的模型
    checkpoint = cfg.resolve("results") / f"{args.model}_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型不存在: {checkpoint}，请先运行 scripts/03_train.py")
    predictor = ModelPredictor(checkpoint)
    logger.info("已加载模型 %s (window=%d, horizon=%d)",
                args.model, predictor.window, predictor.horizon)

    # 回测引擎
    engine = BacktestEngine(
        initial_capital=bt_cfg["initial_capital"],
        commission=bt_cfg["commission"],
        slippage=bt_cfg["slippage"],
        position_pct=bt_cfg["position_pct"],
        stamp_tax=bt_cfg.get("stamp_tax", 0.0005),
    )

    # 动态止盈止损参数（ATR 自适应）
    risk = cfg.get("risk", {})
    risk_cfg = vol_cfg_from_risk(risk) if risk.get("dynamic_volatility", True) else None

    # 市场代理指数（股票池等权收盘）→ 逐日市场状态（择时用）
    closes = {s: bars.set_index(pd.to_datetime(bars["date"]))["close"]
              for s, bars in data.items() if len(bars) >= 60}
    market_close = pd.DataFrame(closes).mean(axis=1).sort_index()
    regime_series = roll_regime(market_close)
    logger.info("市场状态序列 %d 天（代理指数 %d 只）", len(regime_series), len(closes))

    results_dir = cfg.resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    symbols = [args.symbol] if args.symbol else list(data)
    # 各策略净值收集（跨股票归一化后求平均画图）
    norm = {k: [] for k in ("baseline", "stop", "timing", "buy&hold")}
    summary_rows = []

    for symbol in symbols:
        df = data[symbol]
        if len(df) < predictor.window + 10:
            logger.warning("%s 数据过少，跳过", symbol)
            continue

        # 1. 基础信号表（概率信号）
        signal_table = predictor.make_signal(df, threshold=bt_cfg["threshold"])
        signal_table.index = pd.to_datetime(signal_table.index)
        # 每日 ATR（index 对齐 signal_table）
        atr = compute_technical(df.copy(), ["atr"])["atr"]
        atr.index = pd.to_datetime(df["date"])
        logger.info("%s: 信号 %d 天，做多占比 %.1f%%",
                    symbol, len(signal_table), signal_table["signal"].mean() * 100)

        # 2. 策略对比
        eq_base = engine.run(signal_table)["equity"]                        # 纯概率
        eq_stop = engine.run(signal_table, risk_cfg=risk_cfg, atr=atr)["equity"]  # +动态止损
        # 择时：买入需 概率看多 且 择时不看空
        if risk_cfg:
            _, actions = roll_timing_signal(
                df.set_index(pd.to_datetime(df["date"])),
                signal_table["prob_up"], regime_series.reindex(signal_table.index),
                threshold=bt_cfg["threshold"])
            actions = actions.reindex(signal_table.index)   # 对齐到信号表日期
            sig_t = signal_table.copy()
            sig_t["signal"] = np.where((sig_t["signal"] == 1) & (actions != "sell"), 1, 0)
            eq_timing = engine.run(sig_t, risk_cfg=risk_cfg, atr=atr)["equity"]
        else:
            eq_timing = eq_stop
        eq_bh = engine.buy_and_hold(signal_table)["equity"]

        summary = compare(
            ("baseline", eq_base), ("stop", eq_stop),
            ("timing", eq_timing), ("buy&hold", eq_bh))
        summary.insert(0, "symbol", symbol)
        summary_rows.append(summary)

        for k, eq in (("baseline", eq_base), ("stop", eq_stop),
                      ("timing", eq_timing), ("buy&hold", eq_bh)):
            if len(eq) > 1:
                norm[k].append(eq / eq.iloc[0])

    # 3. 汇总：跨股票平均绩效
    if not summary_rows:
        logger.error("没有可回测的股票")
        return
    final = pd.concat(summary_rows)
    avg = final.groupby(level="label")[["total_return", "annual_return", "sharpe",
                                         "max_drawdown", "win_rate"]].mean().round(4)
    print("\n============== 回测结果汇总（跨股票平均） ==============")
    print(avg.to_string())

    print("\n------------------ 明细（每只股票 total_return） ------------------")
    detail = final.reset_index().pivot_table(index="symbol", columns="label",
                                             values="total_return").round(4)
    print(detail.to_string())

    final.to_csv(results_dir / "backtest_summary.csv", encoding="utf-8-sig")
    logger.info("明细已保存到 %s", results_dir / "backtest_summary.csv")

    # 4. 净值曲线（各策略归一化平均）
    avg_curves = {k: pd.concat(v).groupby(level=0).mean()
                  for k, v in norm.items() if v}
    plot_equity(avg_curves, title="3 策略对比（跨股票等权平均，归一化）",
                save_path=str(results_dir / "backtest_equity.png"))
    logger.info("净值曲线图已保存到 %s", results_dir / "backtest_equity.png")

    # 5. ③ 风险调整指标（基准=代理指数）：β/Alpha/IR/Treynor/Sortino/捕获率
    benchmark = market_close / market_close.iloc[0] * bt_cfg["initial_capital"]
    risk_adj = compare_risk_adjusted(
        *[(label, eq) for label, eq in avg_curves.items()],
        ("代理指数", benchmark),
        benchmark=benchmark,
        risk_free=bt_cfg.get("risk_free", 0.02),
    )
    print("\n------------ 风险调整指标（基准=代理指数，跨股票平均净值） ------------")
    print(risk_adj.round(4).to_string())
    risk_adj.to_csv(results_dir / "backtest_risk_adjusted.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
