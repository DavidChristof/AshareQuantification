"""第 5 步：每日更新——拉最新数据 → 重新预测 → 执行调仓 → 更新纸面账户。

这个脚本模拟「每天收盘后跑一次」的例行任务，可配合计划任务每天定时执行。

用法：
    python scripts/05_update.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import load_config                        # noqa: E402
from quant.data.fetcher import fetch_universe               # noqa: E402
from quant.data.loader import load_all                      # noqa: E402
from quant.data.storage import MarketDB                     # noqa: E402
from quant.models.predict import ModelPredictor             # noqa: E402
from quant.trading.engine import TradingEngine              # noqa: E402
from quant.trading.paper import PaperBroker                 # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]
    bt_cfg = cfg["backtest"]

    # ---- 1. 拉取最新行情并入库 ----
    logger.info("拉取最新数据 ...")
    df = fetch_universe(data_cfg["universe"], data_cfg["start_date"], data_cfg["end_date"])
    db = MarketDB(cfg.resolve(data_cfg["db_path"]))
    db.save_bars(df)

    # ---- 2. 加载模型并生成最新预测信号 ----
    checkpoint = cfg.resolve("results") / f"{cfg['model']['type']}_model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型不存在: {checkpoint}，请先运行 scripts/03_train.py")
    predictor = ModelPredictor(checkpoint)
    data = load_all(cfg)

    signals, prices = {}, {}
    for symbol, bars in data.items():
        signal = predictor.make_signal(bars, threshold=bt_cfg["threshold"])
        if signal.empty:
            continue
        latest = signal.iloc[-1]
        signals[symbol] = float(latest["prob_up"])
        prices[symbol] = float(latest["close"])
        logger.info("%s: 最新收盘 %.2f, 上涨概率 %.3f, 信号 %d",
                    symbol, latest["close"], latest["prob_up"], latest["signal"])

    # ---- 3. 纸面交易：调仓 + 净值快照 ----
    latest_date = str(max(
        (data[s].iloc[-1]["date"].date() for s in data), default="2020-01-01"))
    broker = PaperBroker(
        cfg.resolve("paper/paper_account.db"),
        initial_capital=bt_cfg["initial_capital"],
        commission=bt_cfg["commission"],
        slippage=bt_cfg["slippage"],
    )
    tr_cfg = cfg.get("trading", {})
    engine = TradingEngine(
        broker,
        threshold=bt_cfg["threshold"],
        position_pct=bt_cfg["position_pct"],
        max_positions=3,     # 小额账户最多同时持 3 只
        clear_threshold=tr_cfg.get("clear_threshold"),   # 双阈值多日持有（null=旧掉榜即清）
        hold_trim_pct=tr_cfg.get("hold_trim_pct", 0.5),
    )
    summary = engine.rebalance(latest_date, signals, prices)

    # ---- 4. 输出账户摘要 ----
    account = broker.account_summary()
    print("\n==================== 纸面账户日更 ====================")
    print(f"交易日: {summary['date']}")
    print(f"目标持仓: {summary['target']}")
    for a in summary["actions"]:
        print(f"  - {a}")
    print(f"账户权益: {account['equity']:.2f} | 现金: {account['cash']:.2f} | "
          f"累计收益: {account['total_return'] * 100:.2f}%")
    print("======================================================")


if __name__ == "__main__":
    main()
