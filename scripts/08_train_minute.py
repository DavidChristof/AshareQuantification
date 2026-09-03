"""第 8 步：训练分钟级 LSTM 模型（阶段四：盘中分钟信号）。

与日线模型的核心区别：
    - 数据：5 分钟 K 线（MinuteStore 累积）
    - 窗口：过去 30 根 5 分钟 K 线（≈2.5 小时）→ 预测未来 5 根（25 分钟）涨跌
    - 按「交易日分组」切窗：避免把隔夜跳空混进窗口（分钟建模的关键细节）

用法：
    python scripts/08_train_minute.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                     # noqa: E402

from quant.config import load_config                                   # noqa: E402
from quant.features.pipeline import FeaturePipeline                    # noqa: E402
from quant.models.trainer import save_checkpoint, set_seed, train_model  # noqa: E402
from quant.realtime.minute_store import MinuteStore                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

MINUTE_SCALE = 5      # 5 分钟 K 线
WINDOW = 30           # 窗口：过去 30 根（2.5 小时）
HORIZON = 5           # 目标：未来 5 根（25 分钟）


def build_minute_windows(symbol: str, df, window: int, horizon: int):
    """按交易日分组切窗，过滤跨日与 NaN 标签。返回 (X, y)。"""
    pipeline = FeaturePipeline(window=window, horizon=horizon)
    X_list, y_list = [], []
    for _, day_df in df.groupby(df["datetime"].dt.date):
        if len(day_df) < window + 2:      # 一天数据太短，跳过
            continue
        day_df = day_df.copy()
        day_df["date"] = day_df["datetime"]   # pipeline 用 date 列
        X, y, _ = pipeline.make_windows(day_df)
        mask = ~np.isnan(y)                   # 过滤日末无未来标签的样本
        X, y = X[mask], y[mask]
        if len(y) > 0:
            X_list.append(X)
            y_list.append(y)
    if not X_list:
        return np.zeros((0, window, 1)), np.zeros((0,))
    return np.concatenate(X_list), np.concatenate(y_list)


def main():
    cfg = load_config()
    model_cfg = cfg["model"]
    set_seed(model_cfg.get("seed", 42))

    # 1. 加载当前股票池的分钟数据
    store = MinuteStore(cfg.resolve(cfg["minute"]["db_path"]))
    universe = cfg["data"]["universe"]
    loaded = 0
    X_all, y_all = [], []
    for symbol in universe:
        df = store.load_symbol(symbol, scale=MINUTE_SCALE, days=60)
        if len(df) < WINDOW + 2:
            continue
        X, y = build_minute_windows(symbol, df, WINDOW, HORIZON)
        if len(y) > 0:
            X_all.append(X)
            y_all.append(y)
            loaded += 1
            logger.info("%s: %d 个分钟样本", symbol, len(y))
    if not X_all:
        raise RuntimeError("分钟数据不足，请先让系统运行积累数据（scripts 服务常驻）")

    X, y = np.concatenate(X_all), np.concatenate(y_all)
    logger.info("训练数据: X=%s, y=%s, 正样本比例 %.1f%%（%d 只股票）",
                X.shape, y.shape, y.mean() * 100, loaded)

    # 2. 训练
    result = train_model(X, y, model_cfg, val_ratio=0.2,
                         device=model_cfg.get("device", "auto"))
    logger.info("最佳验证集 loss: %.4f | 准确率: %.3f",
                result["best_val_loss"], result["history"]["val_acc"][-1])

    # 3. 保存分钟模型（feature_columns 与日线一致，窗口不同）
    out_path = cfg.resolve("results") / "minute_model.pt"
    pipeline = FeaturePipeline(window=WINDOW, horizon=HORIZON)
    save_checkpoint(result["model"], out_path, model_cfg,
                    feature_columns=pipeline.feature_columns,
                    window=WINDOW, horizon=HORIZON)
    logger.info("分钟模型已保存到 %s", out_path)


if __name__ == "__main__":
    main()
