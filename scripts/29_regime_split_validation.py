"""第 29 步：600 池模型的「市场状态切片」验证（回答：是风格问题还是因子问题？）

背景（承 28_lgbm_offline_validation）：
    28 发现 600 池 LightGBM 整体 Top 超额≈0，但那段验证期(~2025-05~2026-08)全池
    等权未来 20 日在涨(beta 强)。无法区分两种可能：
        (a) 风格问题：价量因子其实有 alpha，只是被普涨 beta 淹没（弱市会显形）
        (b) 因子问题：这 51 个价量特征本身就没有真实 alpha，任何行情都不行

本脚本把验证段按「每交易日全池等权未来 h 收益」切成 弱市/中市/强市 三组（tercile），
逐组重算 RankIC / Top20 做多收益 / 相对全池超额：

    - 若「弱市组超额>0 且明显、强市组≈0/负」→ 风格问题：弱市里价量反转/低波是真
      alpha，可做「行情弱→切因子增强」；等 beta 弱化时段即能用
    - 若「三组超额都≈0/负」→ 因子问题：需换更本质的 alpha（基本面/资金流/行业内中性）

用法：
    python scripts/29_regime_split_validation.py                # 600 池 h=20（全量）
    python scripts/29_regime_split_validation.py --quick        # 冒烟（60 只 + 少树）
    python scripts/29_regime_split_validation.py --horizon 5    # 换生产周期
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
WINDOW = 30
MIN_BARS = 90


# ============ 数据装配（与 28_lgbm_offline_validation 保持一致） ============
def load_pool(pool: str, end_date: str, max_symbols: int | None):
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


def make_tabular_samples(data: dict, horizon: int, window: int = WINDOW):
    """流式装配 GBM tabular 样本（语义同 28 的 make_samples+tabular_features，
    不构建 (N, window, F) 大数组，内存安全）。"""
    feat_dict, fcols = build_enhanced_features(data, window, True, True)
    y_panel = relative_label_panel(data, horizon)
    ret_panel = fwd_return_panel(data, horizon)
    k = min(6, window)
    Xtrs, ys, rs, ds, ss = [], [], [], [], []
    for s, feat in feat_dict.items():
        if s not in y_panel.columns:
            continue
        last = feat.shift(1)
        wm = feat.rolling(window).mean().shift(1)
        ws = feat.rolling(window).std().shift(1)
        slope = last - feat.shift(k)
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


# ============ 分组评估 ============
def eval_days(days_subset: np.ndarray, date_va, prob, ret) -> dict:
    """在指定验证日集合上评估：RankIC + Top20 做多 vs 全池基准。"""
    mask = np.isin(date_va, days_subset)
    d, p, r = date_va[mask], prob[mask], ret[mask]
    cm = _cross_metrics(d, p, r)
    tops, bases = [], []
    for dd in np.unique(d):
        m = d == dd
        if m.sum() < 6:
            continue
        s, rr = p[m], r[m]
        valid = np.isfinite(s) & np.isfinite(rr)
        if valid.sum() < 6:
            continue
        order = np.argsort(s[valid])
        k = max(1, int(valid.sum() * 0.2))
        tops.append(rr[valid][order[-k:]].mean())
        bases.append(rr[valid].mean())
    if not tops:
        return {}
    t, b = np.asarray(tops), np.asarray(bases)
    return {"n_days": int(len(t)), "rankic": round(float(cm["rankic_mean"]), 4),
            "icir": round(float(cm["icir"]), 3),
            "base_mean": round(float(b.mean()), 5),
            "top20": round(float(t.mean()), 5),
            "excess": round(float((t - b).mean()), 5),
            "win": round(float((t > b).mean()), 3)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--pool", default="600", choices=["40", "600"])
    parser.add_argument("--quick", action="store_true", help="冒烟：小池 + 少树")
    parser.add_argument("--max-symbols", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    gbm_cfg = dict(cfg.get("model_v2", {}).get("gbm", {}))
    if args.quick:
        gbm_cfg.update({"n_estimators": 180, "learning_rate": 0.1,
                        "num_leaves": 31, "max_depth": 6})
    end_date = cfg["data"].get("end_date") or pd.Timestamp.now().strftime("%Y-%m-%d")

    t0 = time.time()
    set_seed(42)
    data = load_pool(args.pool, end_date,
                     args.max_symbols or (60 if args.quick else None))
    Xtr, y, rel_ret, dates, symbols, fcols = make_tabular_samples(data, args.horizon)
    tidx, vidx = split_by_date(dates, 0.8)
    logger.info("样本 %d (训练 %d / 验证 %d), 特征 %d", len(Xtr), len(tidx), len(vidx), len(fcols))

    clf = _make_gbm(gbm_cfg)
    try:
        clf.set_params(n_jobs=-1)
    except Exception:  # noqa: BLE001
        pass
    logger.info("训练 LightGBM: %d 样本 x %d 特征 ...", len(tidx), Xtr.shape[1])
    clf.fit(Xtr[tidx], y[tidx])
    prob = clf.predict_proba(Xtr[vidx])[:, 1]

    # ---- 市场状态分桶：每个验证日的「全池等权未来 h 收益」→ tercile ----
    date_va = dates[vidx]
    days = np.unique(date_va)
    mkt = np.asarray([rel_ret[vidx][date_va == d].mean() for d in days])
    lo, hi = np.quantile(mkt, [1 / 3, 2 / 3])
    groups = {
        "弱市(全池未来h<%.1f%%)" % (lo * 100): days[mkt <= lo],
        "中市" : days[(mkt > lo) & (mkt < hi)],
        "强市(全池未来h>%.1f%%)" % (hi * 100): days[mkt >= hi],
    }
    overall = eval_days(days, date_va, prob, rel_ret[vidx])
    out = {"horizon": args.horizon, "pool": args.pool, "quick": args.quick,
           "overall": overall, "regimes": {}}
    for name, dset in groups.items():
        out["regimes"][name] = eval_days(dset, date_va, prob, rel_ret[vidx])

    print("\n===== 市场状态切片验证（池=%s h=%d） =====" % (args.pool, args.horizon))
    hdr = ["市场状态", "验证日", "RankIC", "ICIR", "全池均值", "Top20", "超额", "跑赢基准胜率"]
    print("  ".join(f"{h:>12}" for h in hdr))
    def _row(name, r):
        if not r:
            print(f"{name:<24} 无样本")
            return
        print("  ".join(f"{v:>12}" for v in [
            name[:10], r["n_days"], f"{r['rankic']:.4f}", f"{r['icir']:.3f}",
            f"{r['base_mean']:.4f}", f"{r['top20']:.4f}",
            f"{r['excess']:+.4f}", f"{r['win']:.2f}"]))
    _row("整体", overall)
    for name, r in out["regimes"].items():
        _row(name, r)

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/regime_split_validation.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: results/regime_split_validation.json (耗时 {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
