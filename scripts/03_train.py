"""第 3 步：训练深度学习模型。

流程：
    加载所有股票数据 → 每只构建时序窗口 → 拼接 → 按时间切训练/验证 → 训练 → 保存模型

用法：
    python scripts/03_train.py                     # 默认 lstm
    python scripts/03_train.py --model transformer # 切换 Transformer
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from quant.config import load_config                        # noqa: E402
from quant.data.loader import load_all                      # noqa: E402
from quant.features.pipeline import FeaturePipeline         # noqa: E402
from quant.models.trainer import (                          # noqa: E402
    save_checkpoint, set_seed, train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def build_all_windows(cfg) -> tuple[np.ndarray, np.ndarray]:
    """把股票池所有股票的窗口拼接起来，得到一个通用模型。"""
    feat_cfg = cfg["features"]
    data = load_all(cfg)
    pipeline = FeaturePipeline(window=feat_cfg["window"], horizon=feat_cfg["horizon"])

    X_all, y_all = [], []
    for symbol, df in data.items():
        X, y, _ = pipeline.make_windows(df)
        if len(y) > 0:
            X_all.append(X)
            y_all.append(y)
            logger.info("%s: %d 个样本", symbol, len(y))

    return np.concatenate(X_all), np.concatenate(y_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "transformer"], default=None,
                        help="模型类型（默认读配置）")
    args = parser.parse_args()

    cfg = load_config()
    model_cfg = cfg["model"]
    if args.model:
        model_cfg = {**model_cfg, "type": args.model}

    set_seed(model_cfg.get("seed", 42))

    # 1. 构建全部窗口
    X, y = build_all_windows(cfg)
    logger.info("训练数据: X=%s, y=%s, 正样本比例 %.1f%%", X.shape, y.shape, y.mean() * 100)

    # 2. 训练
    result = train_model(
        X, y, model_cfg,
        val_ratio=0.2,
        device=model_cfg.get("device", "auto"),
    )

    # 3. 报告最佳验证集效果（完整分类指标）
    logger.info("最佳验证集 loss: %.4f", result["best_val_loss"])
    rep = result["val_report"]
    print("\n================ 验证集分类指标 ================")
    print(f"  准确率    : {rep['accuracy']}")
    print(f"  精确率(P) : {rep['precision']}   (预测上涨中真正上涨的比例)")
    print(f"  召回率(R) : {rep['recall']}   (真正上涨中被抓住的比例)")
    print(f"  F1        : {rep['f1']}   (P、R 的调和平均)")
    print(f"  最优阈值  : {rep['threshold']}   (验证集 F1 最优)")
    print(f"  混淆矩阵  : TP={rep['confusion']['tp']} FP={rep['confusion']['fp']} "
          f"TN={rep['confusion']['tn']} FN={rep['confusion']['fn']}")

    # 4. 保存模型
    out_path = cfg.resolve("results") / f"{model_cfg['type']}_model.pt"
    pipeline = FeaturePipeline(window=cfg["features"]["window"],
                               horizon=cfg["features"]["horizon"])
    save_checkpoint(
        result["model"], out_path, model_cfg,
        feature_columns=pipeline.feature_columns,
        window=cfg["features"]["window"],
        horizon=cfg["features"]["horizon"],
        best_threshold=result["best_threshold"],
    )

    # 5. 保存训练历史（便于后续分析过拟合）
    import json
    hist_path = cfg.resolve("results") / f"{model_cfg['type']}_history.json"
    hist_path.write_text(json.dumps(result["history"]), encoding="utf-8")
    logger.info("训练历史已保存到 %s", hist_path)


if __name__ == "__main__":
    main()
