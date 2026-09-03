"""第 2 步：特征工程演示。

目的：让你直观看到「原始 K 线」如何变成「模型输入」。
- 计算技术指标
- 滚动标准化
- 生成时序窗口 (X, y)
- 画出特征与标签的关系

用法：
    python scripts/02_build_features.py --symbol 600519
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                     # noqa: E402

from quant.config import load_config                    # noqa: E402
from quant.data.loader import load_all                  # noqa: E402
from quant.features.pipeline import FeaturePipeline     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="指定股票，默认用第一只")
    args = parser.parse_args()

    cfg = load_config()
    feat_cfg = cfg["features"]
    data = load_all(cfg)

    symbol = args.symbol or next(iter(data))
    df = data[symbol]
    logger.info("使用股票 %s，共 %d 个交易日", symbol, len(df))

    # 1. 构建特征
    pipeline = FeaturePipeline(window=feat_cfg["window"], horizon=feat_cfg["horizon"])
    feat = pipeline.build_features(df)
    logger.info("特征表: %d 行 x %d 列", *feat.shape)

    # 2. 生成窗口
    X, y, dates = pipeline.make_windows(df)
    logger.info("窗口样本: X=%s, y=%s", X.shape, y.shape)
    logger.info("正样本（未来上涨）比例: %.1f%%", y.mean() * 100)

    # 3. 打印最近 3 个样本看看长什么样
    print("\n=== 最近 3 个样本 ===")
    for i in range(1, 4):
        idx = len(dates) - i
        print(f"日期 {pd.Timestamp(dates[idx]).date()} | 未来{feat_cfg['horizon']}天方向: "
              f"{'上涨' if y[idx] == 1 else '下跌'}")

    # 4. 特征相关性（可选：保存到 results）
    results_dir = cfg.resolve("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    feat.tail(200).drop(columns=["date"]).corr().to_csv(
        results_dir / "feature_corr.csv", encoding="utf-8-sig")
    logger.info("特征相关性矩阵已保存到 %s", results_dir / "feature_corr.csv")

    # 5. 可视化特征随时间变化
    try:
        from quant.backtest import metrics
        metrics.plot_equity(
            {"close": df.set_index("date")["close"]},
            title=f"{symbol} 收盘价走势",
            save_path=str(results_dir / f"{symbol}_price.png"),
        )
        logger.info("价格走势图已保存")
    except Exception as exc:  # noqa: BLE001
        logger.warning("绘图失败（不影响流程）: %s", exc)


if __name__ == "__main__":
    main()
