"""第 15 步：时序 K 折交叉验证——评估模型泛化稳定性。

对应教材「K 折交叉验证」：随机 K 折会泄漏未来信息，
时序数据必须按时间顺序切分。每折独立训练，汇总均值±标准差。

用法：
    python scripts/15_cross_validate.py            # 5 折（默认）
    python scripts/15_cross_validate.py --splits 5 --model lstm
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
from quant.models.cross_validate import (                   # noqa: E402
    _format_report, cross_validate,
)
from quant.models.trainer import set_seed                   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def build_all_windows(cfg) -> tuple[np.ndarray, np.ndarray]:
    """把股票池所有股票的窗口拼接起来。"""
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
    parser.add_argument("--splits", type=int, default=5, help="交叉验证折数")
    parser.add_argument("--model", choices=["lstm", "transformer"], default=None,
                        help="模型类型（默认读配置）")
    args = parser.parse_args()

    cfg = load_config()
    model_cfg = cfg["model"]
    if args.model:
        model_cfg = {**model_cfg, "type": args.model}

    X, y = build_all_windows(cfg)
    logger.info("数据: X=%s, y=%s, 正样本 %.1f%%", X.shape, y.shape, y.mean() * 100)

    result = cross_validate(
        X, y, model_cfg,
        n_splits=args.splits,
        device=model_cfg.get("device", "auto"),
        seed=model_cfg.get("seed", 42),
    )

    print("\n==================== 时序 K 折交叉验证 ====================")
    print(_format_report(result))

    out = cfg.resolve("results")
    out.mkdir(parents=True, exist_ok=True)
    result["folds"].to_csv(out / "cross_validate_folds.csv",
                           index=False, encoding="utf-8-sig")
    logger.info("各折明细已保存到 %s", out / "cross_validate_folds.csv")


if __name__ == "__main__":
    main()
