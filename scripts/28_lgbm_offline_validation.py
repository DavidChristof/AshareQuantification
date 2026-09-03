"""第 28 步：600 只大池横截面模型的 LightGBM 离线验证（全量训练前的决策闸门）。

背景：用户问「600 只全市场横截面模型值不值得做」。完整 v2 集成
(LSTM+Transformer+GBM) 在 600 只上全量训练预估 5~7h CPU / ~15GB 内存——
若训完验证段仍无信号就白烧。本脚本用「仅 LightGBM」在**完全相同的数据装配**
(51 特征 + relative 标签 + 严格按日切分) 上先廉价验证，作为决策闸门：

对照实验：
    池 40（现生产范围：data.universe，base 池）
    vs
    池 600（base 40 ∪ 21/22 号下载的中小盘日线缓存 large_scan_cache.pkl）

判定：
    - 600 验证段 RankIC / Top-Bottom 显著 > 40 且 > 0 → 值得投入全量训练
    - 600 ≈ 40 ≈ 0                               → 扩池无益，省下 5~7h 不训

用法（compare 会同时跑 40 与 600，完整对照）：
    python scripts/28_lgbm_offline_validation.py --horizon 5  --pool compare
    python scripts/28_lgbm_offline_validation.py --horizon 20 --pool compare
    python scripts/28_lgbm_offline_validation.py --quick            # 冒烟（小池+少树）
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from quant.config import load_config                                  # noqa: E402
from quant.data.loader import load_all                                # noqa: E402
from quant.models.cross_dataset import (                              # noqa: E402
    build_enhanced_features, fwd_return_panel, relative_label_panel,
    split_by_date,
)
from quant.models.cross_model import _cross_metrics, _make_gbm        # noqa: E402
from quant.models.trainer import set_seed                             # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

CACHE = Path("results/large_scan_cache.pkl")
OUT_JSON = Path("results/lgbm_offline_validation.json")
WINDOW = 30
MIN_BARS = 90          # 少于该根数的标的剔除（滚动 zscore / 特征不稳定）
FEAT_BLOCKS = ["last", "mean", "std", "slope"]   # tabular_features 的 4 个聚合块


def load_pool(pool: str, end_date: str, max_symbols: int | None):
    """按池装配 data dict：{code: df}，统一截断到 end_date、过滤短序列。

    pool='40' → 现池(base) | '600' → base ∪ 中小盘缓存
    """
    cfg_end = pd.Timestamp(end_date)
    base = load_all(load_config())
    data = dict(base)
    if pool == "600":
        if not CACHE.exists():
            logger.warning("%s 不存在，600 池退化为 base 池", CACHE)
        else:
            with open(CACHE, "rb") as f:
                cache = pickle.load(f).get("data", {})
            data.update(cache)
            logger.info("大池缓存 %d 只，合并后 %d 只", len(cache), len(data))

    # 截断到统一 end_date + 过滤过短序列（保数据装配一致性）
    out = {}
    for code, df in data.items():
        if len(df) < MIN_BARS:
            continue
        d = df[df["date"] <= cfg_end].sort_values("date").reset_index(drop=True)
        if len(d) >= MIN_BARS:
            out[code] = d
    if max_symbols:
        codes = sorted(out)[:max_symbols]
        out = {c: out[c] for c in codes}
    logger.info("[池 %s] 有效标的 %d 只，日期 %s ~ %s",
                pool, len(out), min(d["date"].min() for d in out.values()).date(),
                max(d["date"].max() for d in out.values()).date())
    return out


def long_metrics(date_va: np.ndarray, prob: np.ndarray,
                 ret: np.ndarray, top_frac: float = 0.2, min_n: int = 10):
    """逐日做多 prob 前 top_frac（Top）/ 后 top_frac（Bottom）vs 全池基准。

    返回 {top_mean, top_win, n_days, avg_pick, base_mean, top_excess,
          top_vs_bot}：
        base_mean     全池当日均值未来收益（beta 基准）
        top_excess   Top 层均值 - 全池均值（真正超额）
        top_vs_bot   Top 层 - Bottom 层
    """
    tops, bots, bases, pick_n = [], [], [], []
    for d in np.unique(date_va):
        m = date_va == d
        if m.sum() < min_n:
            continue
        s, r = prob[m], ret[m]
        valid = np.isfinite(s) & np.isfinite(r)
        if valid.sum() < min_n:
            continue
        order = np.argsort(s[valid])
        k = max(1, int(valid.sum() * top_frac))
        rv = r[valid]
        tops.append(rv[order[-k:]].mean())
        bots.append(rv[order[:k]].mean())
        bases.append(rv.mean())
        pick_n.append(k)
    if not tops:
        return {"top_mean": 0.0, "top_win": 0.0, "n_days": 0, "avg_pick": 0,
                "base_mean": 0.0, "top_excess": 0.0, "top_vs_bot": 0.0}
    t, b, bs = np.asarray(tops), np.asarray(bots), np.asarray(bases)
    return {"top_mean": round(float(t.mean()), 5),
            "top_win": round(float((t > 0).mean()), 3),
            "n_days": len(t),
            "avg_pick": int(np.mean(pick_n)),
            "base_mean": round(float(bs.mean()), 5),
            "top_excess": round(float((t - bs).mean()), 5),
            "top_vs_bot": round(float((t - b).mean()), 5)}


def make_tabular_samples(data: dict, horizon: int, window: int = WINDOW):
    """流式装配 GBM 用 tabular 样本（与 make_samples+tabular_features 语义一致，
    但只保留 4 个窗口聚合块，不构建 (N, window, F) 大数组 → 内存安全)。

    last_i = vals[i-1]（决策日 t 的前一交易日收盘）；mean/std = vals[i-W..i-1] 的
    均值/标准差（rolling shift(1)，ddof 差异可忽略）；slope = vals[i-1]-vals[i-k]。

    Returns: (Xtr, y, rel_ret, dates, symbols, fcols)
    """
    feat_dict, fcols = build_enhanced_features(data, window, True, True)
    y_panel = relative_label_panel(data, horizon)
    ret_panel = fwd_return_panel(data, horizon)
    k = min(6, window)
    Xtrs, ys, rs, ds, ss = [], [], [], [], []
    for s, feat in feat_dict.items():
        if s not in y_panel.columns:
            continue
        last = feat.shift(1)                        # vals[i-1]
        wm = feat.rolling(window).mean().shift(1)   # mean(vals[i-W..i-1])
        ws = feat.rolling(window).std().shift(1)    # std(vals[i-W..i-1])
        slope = last - feat.shift(k)                # vals[i-1]-vals[i-k]
        blk = pd.concat([last, wm, ws, slope], axis=1).values[window:]
        yy = y_panel[s].reindex(feat.index).values[window:]
        rr = ret_panel[s].reindex(feat.index).values[window:]
        m = np.isfinite(yy)
        n_ok = int(m.sum())
        if n_ok == 0:
            continue
        Xtrs.append(blk[m])
        ys.append(yy[m])
        rs.append(rr[m])
        ds.append(feat.index[window:][m].values)
        ss.append(np.full(n_ok, s))
    if not Xtrs:
        raise RuntimeError("样本装配为空：请检查数据覆盖 window+horizon")
    Xtr = np.asarray(np.concatenate(Xtrs), dtype=np.float32)
    y = np.concatenate(ys).astype(np.float32)
    rel_ret = np.concatenate(rs).astype(np.float32)
    dates = np.concatenate(ds)
    symbols = np.concatenate(ss)
    return Xtr, y, rel_ret, dates, symbols, fcols


def run_one(data: dict, tag: str, horizon: int, gbm_cfg: dict,
            seed: int = 42) -> dict:
    """在给定池上跑完整 LightGBM 离线验证，返回验证段指标。"""
    t0 = time.time()
    set_seed(seed)
    Xtr, y, rel_ret, dates, symbols, fcols = make_tabular_samples(data, horizon)
    tidx, vidx = split_by_date(dates, 0.8)
    logger.info("[%s h=%d] 样本 %d (训练 %d / 验证 %d), 特征 %d, 正样本 %.1f%%",
                tag, horizon, len(Xtr), len(tidx), len(vidx), len(fcols),
                float(y.mean()) * 100)

    clf = _make_gbm(gbm_cfg)
    try:
        clf.set_params(n_jobs=-1)
    except Exception:  # noqa: BLE001 - HistGradientBoosting 无 n_jobs
        pass
    logger.info("[%s h=%d] 训练 LightGBM: %s 样本 x %d 特征 ...",
                tag, horizon, len(Xtr), Xtr.shape[1])
    clf.fit(Xtr[tidx], y[tidx])
    prob = clf.predict_proba(Xtr[vidx])[:, 1]
    cm = _cross_metrics(dates[vidx], prob, rel_ret[vidx])
    lm = long_metrics(dates[vidx], prob, rel_ret[vidx])
    # Top-1/3 多头（与 _cross_metrics 的 spread 口径一致的另一只眼）
    top1_3 = []
    for d in np.unique(dates[vidx]):
        m = dates[vidx] == d
        if m.sum() < 6:
            continue
        s, r = prob[m], rel_ret[vidx][m]
        valid = np.isfinite(s) & np.isfinite(r)
        if valid.sum() < 6:
            continue
        order = np.argsort(s[valid])
        k = max(1, int(valid.sum() // 3))
        top1_3.append(r[valid][order[-k:]].mean())
    lm["top13_mean"] = round(float(np.mean(top1_3)), 5) if top1_3 else 0.0

    # 特征重要性 Top 15（tabular 块名映射回原始因子）
    try:
        imp = clf.feature_importances_
        names = [f"{fcols[i % len(fcols)]}::{FEAT_BLOCKS[i // len(fcols)]}"
                 for i in range(len(imp))]
        order = np.argsort(imp)[::-1][:15]
        top_feats = [(names[i], int(imp[i])) for i in order]
    except Exception:  # noqa: BLE001
        top_feats = []
    logger.info("[%s h=%d] 验证段 RankIC=%.4f ICIR=%.3f | 全池均值=%.4f "
                "Top20=%.4f 超额=%.4f Top-Bot=%.4f Top13=%.4f (%.0fs)",
                tag, horizon, cm["rankic_mean"], cm["icir"], lm["base_mean"],
                lm["top_mean"], lm["top_excess"], lm["top_vs_bot"],
                lm["top13_mean"], time.time() - t0)
    return {"pool": tag, "horizon": horizon, "samples": int(len(y)),
            "train_n": int(len(tidx)), "val_n": int(len(vidx)),
            "features": len(fcols), "n_stocks": len(data),
            "rankic_mean": cm["rankic_mean"], "rankic_std": cm["rankic_std"],
            "icir": cm["icir"], "top_bottom": cm["top_bottom"],
            "n_days": cm["n_days"], **lm, "top_feats": top_feats,
            "seconds": round(time.time() - t0, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=20,
                        help="未来收益 horizon（5=周 / 20=月；实证 20 更强）")
    parser.add_argument("--pool", default="compare",
                        choices=["40", "600", "compare"],
                        help="40=现池 / 600=大池 / compare=对照")
    parser.add_argument("--quick", action="store_true", help="冒烟：小池 + 少树")
    parser.add_argument("--max-symbols", type=int, default=None,
                        help="限制每池最大标的总数（冒烟/调试）")
    args = parser.parse_args()

    cfg = load_config()
    gbm_cfg = dict(cfg.get("model_v2", {}).get("gbm", {}))
    if args.quick:
        gbm_cfg.update({"n_estimators": 180, "learning_rate": 0.1,
                        "num_leaves": 31, "max_depth": 6})
    end_date = cfg["data"].get("end_date")
    if not end_date:
        end_date = pd.Timestamp.now().strftime("%Y-%m-%d")

    pools = ["40", "600"] if args.pool == "compare" else [args.pool]
    results = []
    for pool in pools:
        data = load_pool(pool, end_date, args.max_symbols or (60 if args.quick else None))
        if len(data) < 20:
            logger.warning("[池 %s] 标的过少(%d)，跳过", pool, len(data))
            continue
        results.append(run_one(data, f"pool{pool}", args.horizon, gbm_cfg))

    if not results:
        logger.error("无有效结果")
        sys.exit(1)

    print("\n===== LightGBM 离线验证（h=%d） =====" % args.horizon)
    hdr = ["池", "股票数", "样本", "验证日", "RankIC", "ICIR",
           "全池均值", "Top20", "超额", "Top13", "Top20胜率"]
    print("  ".join(f"{h:>9}" for h in hdr))
    for r in results:
        print("  ".join(f"{v:>9}" for v in [
            r["pool"], r["n_stocks"], r["samples"], r["n_days"],
            f"{r['rankic_mean']:.4f}", f"{r['icir']:.3f}",
            f"{r['base_mean']:.4f}", f"{r['top_mean']:.4f}",
            f"{r['top_excess']:+.4f}", f"{r['top13_mean']:.4f}",
            f"{r['top_win']:.2f}"]))
    for r in results:
        if r["top_feats"]:
            print(f"\n[池 {r['pool']}] 特征重要性 Top10:")
            for name, v in r["top_feats"][:10]:
                print(f"    {name:<34} {v}")

    # 保存结果（先写盘再打印，防 print 编码崩溃丢数据）
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"horizon": args.horizon, "quick": args.quick,
                   "results": results}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
