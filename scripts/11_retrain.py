"""第 11 步：一键重训练——拉最新日线 → 重训日线模型 → 提示重启服务。

为什么需要：模型权重在训练时固定，不会随新数据自动更新。
数据不断累积后，市场风格漂移（concept drift），旧模型会逐渐失效。
定期把最新数据纳入训练重新 fit，能让模型跟上当前市场。

重训前自动备份旧模型（results/lstm_model.pt.bak），若新模型变差可回退。

用法：
    python scripts/11_retrain.py              # 先拉最新数据，再重训（建议每周/每月跑一次）
    python scripts/11_retrain.py --no-fetch   # 只重训，用库中已有数据
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from quant.config import load_config                        # noqa: E402
from quant.data.loader import load_all                      # noqa: E402
from quant.data.fetcher import fetch_universe               # noqa: E402
from quant.data.storage import MarketDB                     # noqa: E402
from quant.features.pipeline import FeaturePipeline         # noqa: E402
from quant.models.trainer import (                          # noqa: E402
    save_checkpoint, set_seed, train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def build_all_windows(cfg) -> tuple[np.ndarray, np.ndarray]:
    """把股票池所有股票的窗口拼接起来（同 03_train）。"""
    feat_cfg = cfg["features"]
    data = load_all(cfg)
    pipeline = FeaturePipeline(window=feat_cfg["window"], horizon=feat_cfg["horizon"])
    X_all, y_all = [], []
    for symbol, df in data.items():
        X, y, _ = pipeline.make_windows(df)
        if len(y) > 0:
            X_all.append(X)
            y_all.append(y)
    return np.concatenate(X_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fetch", action="store_true",
                        help="不拉取最新数据，直接用库中已有数据重训")
    parser.add_argument("--model", choices=["lstm", "transformer"], default=None)
    args = parser.parse_args()

    cfg = load_config()
    model_cfg = cfg["model"]
    if args.model:
        model_cfg = {**model_cfg, "type": args.model}

    # 1. 拉最新数据（可选）
    if not args.no_fetch:
        data_cfg = cfg["data"]
        logger.info("拉取最新日线（end_date=%s）...", data_cfg["end_date"])
        df = fetch_universe(data_cfg["universe"], data_cfg["start_date"], data_cfg["end_date"])
        n = MarketDB(cfg.resolve(data_cfg["db_path"])).save_bars(df)
        logger.info("已更新 %d 行", n)

    # 2. 备份旧模型
    out_path = cfg.resolve("results") / f"{model_cfg['type']}_model.pt"
    bak_path = out_path.with_suffix(".pt.bak")
    if out_path.exists():
        shutil.copy2(out_path, bak_path)
        logger.info("旧模型已备份到 %s", bak_path)

    # 3. 构建窗口 + 训练
    set_seed(model_cfg.get("seed", 42))
    X, y = build_all_windows(cfg)
    logger.info("训练数据: X=%s, 正样本 %.1f%%", X.shape, y.mean() * 100)
    result = train_model(X, y, model_cfg, val_ratio=0.2,
                         device=model_cfg.get("device", "auto"))
    final_acc = result["history"]["val_acc"][-1]
    logger.info("重训完成，最终验证集准确率 %.3f（loss %.4f）",
                final_acc, result["best_val_loss"])

    # 4. 保存新模型（覆盖）
    pipeline = FeaturePipeline(window=cfg["features"]["window"],
                               horizon=cfg["features"]["horizon"])
    save_checkpoint(
        result["model"], out_path, model_cfg,
        feature_columns=pipeline.feature_columns,
        window=cfg["features"]["window"],
        horizon=cfg["features"]["horizon"],
        best_threshold=result["best_threshold"],
    )
    logger.info("新模型已保存到 %s", out_path)

    # 5. 保存训练历史 + 提示重启
    import json
    (cfg.resolve("results") / f"{model_cfg['type']}_history.json").write_text(
        json.dumps(result["history"]), encoding="utf-8")

    print("\n==================== 重训完成 ====================")
    print(f"  新模型: {out_path.name}（验证集准确率 {final_acc:.3f}）")
    print(f"  旧模型备份: {bak_path.name}")
    print("\n  >>> 请重启 API 服务，让新模型生效：")
    print("      D:/Python/Python3_12/python.exe -m uvicorn api.main:app --port 8001")
    print("      （或用 启动量化系统.bat 重启）")
    print("  >>> 若新模型效果不佳，可把 .bak 复制回原文件名回退。")


if __name__ == "__main__":
    main()
