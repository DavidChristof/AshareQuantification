"""审计：旧 LSTM 在验证段的 RankIC=0.318 是否真实？还是评估口径/泄漏假象。

对照 18_train_ensemble 里旧模型 RankIC 异常偏高（0.318，远超合理 <0.05），
需要判断：
  A. 旧 prob 是否真的能预测未来收益（若真，实盘不该亏 → 矛盾）
  B. 还是 prob 只是近期动量/收益的镜像，被误算成"未来预测"
  C. 还是索引错位（概率标在错误日期）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from quant.config import load_config
from quant.data.loader import load_all
from quant.models.cross_dataset import fwd_return_panel
from quant.models.cross_model import load_ensemble
from quant.models.predict import ModelPredictor

cfg = load_config()
data = load_all(cfg)

horizon = 5
ret_fwd = fwd_return_panel(data, horizon)          # 未来 h 收益面板
close = pd.DataFrame({s: b.set_index("date")["close"] for s, b in data.items()}).sort_index()
ret_past1 = close.pct_change(1)                    # 昨日收益（t-1 → t）
ret_past5 = close.pct_change(5)                    # 过去 5 日收益

# --- v2 验证日期 ---
from quant.models.cross_dataset import make_samples
X, y, rel, dates, syms, fc = make_samples(data, window=30, horizon=horizon)
_, vidx = __import__("quant.models.cross_dataset", fromlist=["split_by_date"]).split_by_date(dates, 0.8)
val_dates = np.unique(dates[vidx])
print(f"验证段 {len(val_dates)} 天: {np.datetime_as_string(val_dates[0])[:10]} ~ {np.datetime_as_string(val_dates[-1])[:10]}")


def _panel_corr(prob_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                label: str):
    """每日截面 spearman 序列，返回均值/ICIR/天数。"""
    ics = []
    for d in val_dates:
        d = pd.Timestamp(d)
        if d not in prob_panel.index or d not in ret_panel.index:
            continue
        p = prob_panel.loc[d].dropna()
        r = ret_panel.loc[d].reindex(p.index).dropna()
        both = p.index.intersection(r.index)
        if len(both) < 6:
            continue
        rho, _ = spearmanr(p[both], r[both])
        if np.isfinite(rho):
            ics.append(rho)
    a = np.asarray(ics)
    m, s = a.mean(), a.std()
    print(f"  {label:<34s} RankIC={m:.4f} ICIR={m/s if s>0 else 0:.3f}  n={len(ics)}")
    return m


# --- 旧模型 panel ---
old = ModelPredictor(cfg.resolve("results") / "lstm_model.pt")
old_panel = {}
for s, bars in data.items():
    sig = old.make_signal(bars)
    if len(sig):
        old_panel[s] = sig["prob_up"]
old_panel = pd.DataFrame(old_panel).sort_index()
old_panel.index = pd.to_datetime(old_panel.index)

# v2 ensemble panel
ens = load_ensemble(cfg.resolve("results") / "model_v2")
sig_all = ens.make_signals_all(data)
ens_panel = pd.DataFrame({s: df["prob_up"] for s, df in sig_all.items()}).sort_index()
ens_panel.index = pd.to_datetime(ens_panel.index)

print("\n=== 旧 LSTM (绝对概率) ===")
_panel_corr(old_panel, ret_fwd, "prob vs 未来5日收益(审计用口径)")
_panel_corr(old_panel, ret_past1, "prob vs 昨日收益(镜像检查)")
_panel_corr(old_panel, ret_past5, "prob vs 过去5日收益(镜像检查)")
# 错位检查：prob[t] vs fwd_ret[t+1]（若 prob 本应早一天）
_panel_corr(old_panel.shift(1), ret_fwd, "prob[t-1] vs 未来5日(错位-1)")
_panel_corr(old_panel.shift(-1), ret_fwd, "prob[t+1] vs 未来5日(错位+1)")

print("\n=== v2 集成 (相对概率) ===")
_panel_corr(ens_panel, ret_fwd, "prob vs 未来5日收益")
_panel_corr(ens_panel, ret_past1, "prob vs 昨日收益(镜像)")

# 概率数值分布
print("\nprob 数值范围:")
print("  旧 LSTM:", round(float(old_panel.min().min()), 4), "~", round(float(old_panel.max().max()), 4),
      "中位", round(float(old_panel.median().median()), 4))
print("  v2 集成:", round(float(ens_panel.min().min()), 4), "~", round(float(ens_panel.max().max()), 4),
      "中位", round(float(ens_panel.median().median()), 4))
