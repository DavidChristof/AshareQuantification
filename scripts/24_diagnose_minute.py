"""诊断分钟模型：数据量/行情覆盖 + 现模型概率分布 + 标签平衡。

目的：定位分钟模型为何输出极端概率（常 0.002~0.003 恒判 sell）——是
    A. 训练数据只有一小段单边行情（样本偏）？
    B. 模型过拟合/饱和（sigmoid 输出被推极端）？
    C. 标签本身不平衡？
据此决定「重训（按交易日切分）」还是「概率校准（Platt）」。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                             # noqa: E402
import pandas as pd                                            # noqa: E402

from quant.config import load_config                           # noqa: E402
from quant.models.predict import ModelPredictor                # noqa: E402
from quant.realtime.minute_store import MinuteStore            # noqa: E402

cfg = load_config()
mc = cfg.get("minute", {})
SCALE = int(mc.get("scale", 5))
store = MinuteStore(cfg.resolve(mc.get("db_path", "data/minute.db")))
universe = cfg["data"]["universe"]

print("=== 1) 分钟数据量 / 覆盖 ===")
total_bars = 0
day_ranges = []
for s in universe[:40]:
    try:
        df = store.load_symbol(s, scale=SCALE, days=400)
    except Exception:
        continue
    if df is None or df.empty:
        continue
    total_bars += len(df)
    days = sorted(df["datetime"].dt.date.unique())
    if days:
        day_ranges.append((len(df), days[0], days[-1], len(days)))
bars = sorted(day_ranges, key=lambda x: -x[0])
if bars:
    print(f"40 只合计 bars={total_bars}")
    for n, d0, d1, nd in bars[:3]:
        print(f"  单只最多 bars={n} 日期 {d0} ~ {d1}（{nd} 个交易日）")
    all_d0 = min(b[1] for b in bars); all_d1 = max(b[2] for b in bars)
    print(f"  覆盖日期 {all_d0} ~ {all_d1}")

print("\n=== 2) 标签平衡：按交易日『未来 25 分钟上涨』比例 ===")
ratio_by_day = {}
for s in universe[:40]:
    try:
        df = store.load_symbol(s, scale=SCALE, days=40)
    except Exception:
        continue
    if df is None or df.empty or len(df) < 6:
        continue
    c = df["close"].reset_index(drop=True)
    fwd = c.shift(-5) / c - 1
    df = df.reset_index(drop=True)
    df["fwd_pos"] = fwd > 0
    for d, g in df.groupby(df["datetime"].dt.date):
        g = g.dropna(subset=["fwd_pos"])
        if len(g) >= 20:
            ratio_by_day.setdefault(d, []).append(g["fwd_pos"].mean())
days_sorted = sorted(ratio_by_day)
if days_sorted:
    for d in days_sorted[-12:]:
        r = np.mean(ratio_by_day[d])
        print(f"  {d}: 上涨比例 {r*100:.1f}%（{len(ratio_by_day[d])} 只均值）")

print("\n=== 3) 现模型概率分布（最近 2 天，逐窗口 sigmoid） ===")
mp = ModelPredictor(cfg.resolve("results") / "minute_model.pt")
print(f"  meta: window={mp.window} horizon={mp.horizon} best_threshold={mp.best_threshold}")
probs = []
for s in universe[:12]:
    try:
        df = store.load_symbol(s, scale=SCALE, days=2)
    except Exception:
        continue
    if df is None or df.empty or len(df) < mp.window:
        continue
    try:
        p = mp.predict_probability(df.rename(columns={"datetime": "date"}))
        p = p.dropna()
        probs += list(p.tail(60).values)
    except Exception as e:
        print(f"  {s} 预测失败: {e}")
if probs:
    a = np.asarray(probs)
    print(f"  样本数 {len(a)} | min={a.min():.4f} max={a.max():.4f} mean={a.mean():.4f}")
    for q in (5, 25, 50, 75, 95):
        print(f"    {q}% 分位 = {np.percentile(a, q):.4f}")
    print(f"  sell(<0.45)占比 {100*(a<0.45).mean():.1f}% | buy(>0.55)占比 {100*(a>0.55).mean():.1f}%")
