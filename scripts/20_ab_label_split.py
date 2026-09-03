"""A/B 实验：同特征、同日期切分下，「绝对标签」vs「相对标签」的真实外推 RankIC。

两种用途：
    1) 验证旧 LSTM 的高 RankIC 是不是「行序乱切」样本内假象（用默认 5 日）
    2) 扫描预测期 horizon（5/10/20 日）：更长周期是否有可外推信号

用法：python scripts/20_ab_label_split.py                 # 5 日（A/B 复现）
      python scripts/20_ab_label_split.py --horizon 10    # 10 日预测期
      python scripts/20_ab_label_split.py --horizon 20    # 20 日预测期
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import torch

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                         # noqa: E402
from quant.data.loader import load_all                       # noqa: E402
from quant.models.cross_dataset import make_samples, split_by_date  # noqa: E402
from quant.models.cross_model import _cross_metrics, _seq_probs  # noqa: E402
from quant.models.dataset import TimeSeriesDataset           # noqa: E402
from quant.models.trainer import _fit_model, set_seed        # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--horizon", type=int, default=5, help="预测期（交易日），默认 5")
args = parser.parse_args()

cfg = load_config()
model_cfg = {**cfg["model"], "type": "lstm", "epochs": 12}
feat_cfg = cfg["features"]
window = int(feat_cfg["window"])
horizon = args.horizon
device = "cpu"

data = load_all(cfg)
set_seed(42)

print(f"===== horizon={horizon}日 | window={window} | 日期切分(后20%时间外) =====")
print(f"{'标签':<10} {'样本':>7} {'RankIC':>8} {'ICIR':>7} {'Top-Bottom':>11}")
for label_mode in ("absolute", "relative"):
    X, y, rel, dates, syms, fc = make_samples(
        data, window=window, horizon=horizon, label_mode=label_mode)
    tr, va = split_by_date(dates, 0.8)
    res = _fit_model(TimeSeriesDataset(X[tr], y[tr]),
                     TimeSeriesDataset(X[va], y[va]), model_cfg, device)
    prob = _seq_probs(res["model"], X[va], device)
    cm = _cross_metrics(dates[va], prob, rel[va])
    n_val = int(len(va))
    print(f"{label_mode:<10} {n_val:>7} {cm['rankic_mean']:>8.4f} "
          f"{cm['icir']:>7.3f} {cm['top_bottom']:>11.4f}")

# 提示：RankIC 判定（教材 IC 口径）
print("\n提示: |RankIC|>0.05 才算有可用的微弱信号；ICIR>0.5 才算稳定")
