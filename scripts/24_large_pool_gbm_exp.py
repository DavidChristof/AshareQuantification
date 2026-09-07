"""第 24 步（阶段二·A/B 实验）：40 池 vs 大池子样本 的 v2-GBM 时点外 RankIC 对比。

背景/目的：判断「把 v2 短期模型从 40 池扩到大横截面训练」是否值得（阶段一因子 IC 已大幅上升，
但那是单因子 IC，需验证**训练后的模型**在时点外能否真正排序）。

做法：同一套 v2 特征/标签管线，只换训练截面，GBM 成员（本机内存/时间可行），
在**各自验证段（按日期切最后 20%）**算 RankIC —— 注意 RankIC 对的是**真实未来收益**，
跨池可比（标签只用于训练目标，不影响评估口径）。

先小样本(K)校准跑通 + 外推全量(559)的内存/时间，再决定是否全量。

用法：
    python scripts/24_large_pool_gbm_exp.py --pool base              # 40 池基线
    python scripts/24_large_pool_gbm_exp.py --pool large --k 50     # 大池随机抽 50
    python scripts/24_large_pool_gbm_exp.py --pool mix --k 50       # 现池40 + 大池抽50
    python scripts/24_large_pool_gbm_exp.py --full                  # 全量 599（需内存富余）
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd                                                    # noqa: E402

from quant.config import load_config                                   # noqa: E402
from quant.data.loader import load_all                                 # noqa: E402
from quant.models.cross_model import (_cross_metrics, _make_gbm, tabular_features,
                                      train_ensemble)
from quant.models.cross_dataset import make_samples, split_by_date

LARGE_DB = Path("data/large_pool.db")
REPORT = Path("results/large_pool_gbm_exp.json")

parser = argparse.ArgumentParser()
parser.add_argument("--pool", default="base", choices=["base", "large", "mix", "full"])
parser.add_argument("--k", type=int, default=50, help="大池随机抽取数量（large/mix 用）")
parser.add_argument("--full", action="store_true", help="large 用全量 559（忽略 --k）")
parser.add_argument("--recent", type=int, default=0,
                    help="每只只用最近 N 个交易日（0=全部）。控内存：全 599 时建议 350~700")
parser.add_argument("--trees", type=int, default=300, help="GBM 树数（quick 校准）")
parser.add_argument("--ensemble", action="store_true",
                    help="跑完整 v2 集成(lstm+transformer+gbm, epochs=6)——需 torch CUDA 可用即自动上 GPU")
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

cfg = load_config()
mv2 = cfg.get("model_v2", {})
feat = cfg["features"]
window, horizon = int(feat["window"]), int(feat["horizon"])


def _trim_recent(d: dict, r: int) -> dict:
    """每只只保留最近 r 个交易日（r<=0 表示全部），控 make_samples 内存。"""
    if r <= 0:
        return d
    return {c: df.tail(r) for c, df in d.items()}


def load_large_pool(sample_k: int | None, rng: random.Random,
                    add_base: bool = False, recent: int = 0) -> dict:
    """读 data/large_pool.db（559 只全历史），可选抽 sample_k；add_base 并入现池40。"""
    codes = [r[0] for r in sqlite3.connect(str(LARGE_DB)).execute(
        "SELECT DISTINCT symbol FROM large_daily").fetchall()]
    if sample_k is not None and sample_k < len(codes):
        codes = rng.sample(sorted(codes), sample_k)
    out: dict = {}
    con = sqlite3.connect(str(LARGE_DB))
    for c in codes:
        rows = con.execute(
            "SELECT date, open, high, low, close, volume, amount FROM large_daily "
            "WHERE symbol=? ORDER BY date", (c,)).fetchall()
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                         "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        out[c] = df
    con.close()
    if add_base:
        out.update(_trim_recent(load_all(cfg), recent))   # 现池40（market.db）
    return _trim_recent(out, recent)


def run_pool(data: dict, tag: str) -> dict:
    t0 = time.time()
    X, y, rel_ret, dates, symbols, fcols = make_samples(
        data, window=window, horizon=horizon,
        use_cross=bool(mv2.get("cross_features", True)),
        use_alpha=bool(mv2.get("alpha_features", True)),
        label_mode=mv2.get("label_mode", "relative"),
    )
    t_build = time.time() - t0
    tidx, vidx = split_by_date(dates, 0.8)
    print(f"[{tag}] make_samples 完成: N={len(X)} (训练 {len(tidx)}/验证 {len(vidx)})"
          f" 形状={X.shape} 耗时 {t_build:.0f}s X内存≈{X.nbytes/1e6:.0f}MB", flush=True)

    t0 = time.time()
    Xtr2 = tabular_features(X[tidx])
    Xva2 = tabular_features(X[vidx])
    gbm_cfg = dict(mv2.get("gbm", {}))
    gbm_cfg["n_estimators"] = args.trees
    clf = _make_gbm(gbm_cfg)
    clf.fit(Xtr2, y[tidx])
    prob = clf.predict_proba(Xva2)[:, 1]
    t_fit = time.time() - t0
    print(f"[{tag}] GBM 拟合耗时 {t_fit:.0f}s（{args.trees} 树）", flush=True)

    cm = _cross_metrics(dates[vidx], prob, rel_ret[vidx])
    print(f"[{tag}] 时点外(验证段) RankIC均值={cm['rankic_mean']:.4f}  ICIR={cm['icir']:.3f} "
          f"Top-Bottom={cm['top_bottom']:.4f}  验证日数={cm['n_days']}", flush=True)
    del X, Xtr2, Xva2, prob, clf
    gc.collect()
    return {"pool": tag, "n_symbols": len(data), "n_train": int(len(tidx)),
            "n_val": int(len(vidx)), "build_sec": round(t_build, 1), "fit_sec": round(t_fit, 1),
            "rankic_mean": cm["rankic_mean"], "icir": cm["icir"],
            "top_bottom": cm["top_bottom"], "n_days": int(cm["n_days"])}


def run_ensemble(data: dict, tag: str) -> dict:
    """完整 v2 集成（lstm+transformer+gbm）。torch CUDA 可用即自动上 GPU。"""
    dcfg = {
        "features": dict(cfg["features"]),
        "model": {**dict(cfg["model"]), "device": "auto"},
        "model_v2": {**mv2,
                     "members": list(mv2.get("members", ["lstm", "transformer", "gbm"])),
                     "epochs": int(args.epochs),
                     "gbm": {**dict(mv2.get("gbm", {})), "n_estimators": int(args.trees)}},
    }
    t0 = time.time()
    res = train_ensemble(data, dcfg)
    cm = res["cross_metrics"]
    print(f"[{tag}] 完整集成完成，耗时 {time.time() - t0:.0f}s（成员={res['members']}）", flush=True)
    e = cm["ensemble"]
    print(f"[{tag}] 集成 时点外RankIC={e['rankic_mean']:.4f} ICIR={e['icir']:.3f} "
          f"n_days={e['n_days']}", flush=True)
    for m, mm in cm["members"].items():
        print(f"[{tag}]  成员 {m:<11} RankIC={mm['rankic_mean']:.4f} ICIR={mm['icir']:.3f}", flush=True)
    out = {"pool": tag, "n_symbols": len(data), "mode": "ensemble",
           "members": res["members"], "epochs": args.epochs,
           "rankic_mean": e["rankic_mean"], "icir": e["icir"],
           "top_bottom": e["top_bottom"], "n_days": int(e["n_days"]),
           "n_train": int(len(res["tidx"])), "n_val": int(len(res["vidx"])),
           "members_detail": {m: {"rankic": mm["rankic_mean"], "icir": mm["icir"]}
                              for m, mm in cm["members"].items()}}
    gc.collect()
    return out


def main():
    rng = random.Random(args.seed)
    results: dict = {"generated_at": datetime.now().isoformat(timespec="seconds"),
                     "seed": args.seed, "trees": args.trees,
                     "mode": "ensemble" if args.ensemble else "gbm-only"}
    runner = run_ensemble if args.ensemble else run_pool
    if args.pool == "base":
        data = _trim_recent(load_all(cfg), args.recent)
        results["runs"] = [runner(data, f"base40(recent{args.recent or 'all'})")]
    elif args.pool == "full":
        data = load_large_pool(None, rng, add_base=True, recent=args.recent)
        results["runs"] = [runner(data, f"full{len(data)}(40+大池,recent{args.recent or 'all'})")]
    else:
        k = None if args.full else args.k
        add_base = args.pool == "mix"
        data = load_large_pool(k, rng, add_base=add_base, recent=args.recent)
        results["runs"] = [runner(data,
                                  f"large{len(data)}" + ("(mix+base40)" if add_base else ""))]
    # 内存/时间外推（全量 599）
    large_total = sqlite3.connect(str(LARGE_DB)).execute(
        "SELECT COUNT(DISTINCT symbol) FROM large_daily").fetchone()[0]
    r = results["runs"][0]
    per = (r["n_train"] + r["n_val"]) / max(r["n_symbols"], 1)
    est = per * (40 + large_total)
    results["extrapolate_full599"] = {
        "est_n_samples": int(est),
        "note": "全量 599≈15×40池样本；make_samples 峰值内存≈N×30×51×8B，全量约 11GB，"
                "本机 15.6GB 且当前仅 ~2GB 空闲 → 需先释放内存或降特征/分片才能全量。",
    }
    REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[报告] results/large_pool_gbm_exp.json")


if __name__ == "__main__":
    main()
