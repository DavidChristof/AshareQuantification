"""横截面增强模型：集成训练 + 推理 + 截面评估（模型 v2）。

组成（config model_v2.members）：
    lstm        序列模型（吃 window×F 窗口）
    transformer 序列模型（自注意力，与小样本 LSTM 互补）
    gbm         LightGBM / sklearn HistGradientBoosting（吃窗口的 tabular 汇总，
                对横截面因子天然友好）

集成 = 各成员 sigmoid 概率的均值（soft-voting）。输出"跑赢当日全池中位数的概率"，
语义是**横截面相对强弱**——与"选 TOP12"任务一致，因此可直接用于排序打分。

评估不用 accuracy（近随机无意义），改用：
    RankIC       模型概率 vs 未来收益的逐日截面秩相关（预测力）
    ICIR         平均 IC / IC 标准差（稳定性）
    Top-Bottom   prob 前 1/3 - 后 1/3 的未来收益差（分层区分度）
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, TensorDataset

from ..factors.analysis import _prepare_panels
from .cross_dataset import build_enhanced_features, fwd_return_panel, make_samples, split_by_date
from .dataset import TimeSeriesDataset
from .metrics import best_threshold, classification_report, precision_recall_f1
from .trainer import _fit_model, build_model, set_seed

logger = logging.getLogger(__name__)


# ============================================================
# GBM 基学习器（LightGBM 优先，缺失回退 sklearn）
# ============================================================
def _make_gbm(gbm_cfg: dict):
    try:
        import lightgbm as lgb  # type: ignore
        logger.info("GBM 基学习器: LightGBM")
        return lgb.LGBMClassifier(
            n_estimators=int(gbm_cfg.get("n_estimators", 800)),
            learning_rate=float(gbm_cfg.get("learning_rate", 0.05)),
            num_leaves=int(gbm_cfg.get("num_leaves", 63)),
            max_depth=int(gbm_cfg.get("max_depth", 8)),
            min_child_samples=int(gbm_cfg.get("min_child_samples", 40)),
            subsample=float(gbm_cfg.get("subsample", 0.9)),
            colsample_bytree=float(gbm_cfg.get("colsample_bytree", 0.8)),
            subsample_freq=1,
            random_state=42, verbose=-1,
        )
    except ImportError:  # pragma: no cover
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
        logger.warning("lightgbm 未安装，回退 sklearn HistGradientBoosting")
        return HistGradientBoostingClassifier(
            max_iter=int(gbm_cfg.get("n_estimators", 300)),
            learning_rate=float(gbm_cfg.get("learning_rate", 0.05)),
            max_leaf_nodes=int(gbm_cfg.get("num_leaves", 63)),
            max_depth=int(gbm_cfg.get("max_depth", 8)),
            min_samples_leaf=int(gbm_cfg.get("min_child_samples", 40)),
            random_state=42,
        )


def tabular_features(X: np.ndarray) -> np.ndarray:
    """窗口序列 → GBM 用 tabular 特征：末帧 + 时序均值/标准差/斜率。

    X: (N, window, F) → (N, 4F)
    """
    last = X[:, -1, :]
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    k = min(6, X.shape[1])
    slope = X[:, -1, :] - X[:, -k, :]
    return np.concatenate([last, mean, std, slope], axis=1)


# ============================================================
# 批量前向
# ============================================================
@torch.no_grad()
def _seq_probs(model: nn.Module, X: np.ndarray, device: str, bs: int = 512) -> np.ndarray:
    """对整批窗口 (N, T, F) 输出 sigmoid 概率。"""
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()),
                        batch_size=bs, shuffle=False)
    probs = []
    for (xb,) in loader:
        logit = model(xb.to(device))
        probs.append(torch.sigmoid(logit).cpu().numpy())
    return np.concatenate(probs)


# ============================================================
# 截面评估：RankIC / ICIR / Top-Bottom spread
# ============================================================
def _cross_metrics(dates: np.ndarray, score: np.ndarray,
                   ret: np.ndarray, min_n: int = 6):
    """逐日截面：RankIC 序列 + Top-Bottom 分层 spread。

    Returns:
        dict{rankic_mean, rankic_std, icir, top_bottom_mean, n_days}
    """
    ics, spreads = [], []
    for d in np.unique(dates):
        m = dates == d
        if m.sum() < min_n:
            continue
        s, r = score[m], ret[m]
        if np.isnan(s).all() or np.isnan(r).all():
            continue
        valid = np.isfinite(s) & np.isfinite(r)
        if valid.sum() < min_n:
            continue
        rho, p = spearmanr(s[valid], r[valid])
        if not np.isfinite(rho):
            continue
        ics.append(rho)
        order = np.argsort(s[valid])
        k = max(1, int(valid.sum() // 3))
        top = r[valid][order[-k:]].mean()
        bot = r[valid][order[:k]].mean()
        spreads.append(top - bot)
    if not ics:
        return {"rankic_mean": 0.0, "rankic_std": 0.0, "icir": 0.0,
                "top_bottom": 0.0, "n_days": 0}
    arr = np.asarray(ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std())
    return {"rankic_mean": round(mean_ic, 4),
            "rankic_std": round(std_ic, 4),
            "icir": round(mean_ic / std_ic, 3) if std_ic > 0 else 0.0,
            "top_bottom": round(float(np.mean(spreads)), 4),
            "n_days": len(ics)}


# ============================================================
# 集成训练
# ============================================================
def train_ensemble(data: dict, cfg) -> dict:
    """全流程：装配样本 → 按日期切分 → 训练成员 → 集成调阈值 → 截面评估。

    cfg: 顶层 Config（读 features/model/model_v2）。
    Returns: dict{member_probs: {name: val_prob}, ... 供保存与报告}
    """
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    mv2 = cfg.get("model_v2", {})
    window = int(feat_cfg["window"])
    horizon = int(feat_cfg["horizon"])
    members = mv2.get("members", ["lstm", "transformer", "gbm"])
    device = model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(mv2.get("seed", model_cfg.get("seed", 42))))
    logger.info("集成训练: 成员=%s 设备=%s window=%d horizon=%d",
                members, device, window, horizon)

    X, y, rel_ret, dates, symbols, fcols = make_samples(
        data, window=window, horizon=horizon,
        use_cross=bool(mv2.get("cross_features", True)),
        use_alpha=bool(mv2.get("alpha_features", True)),
    )
    tidx, vidx = split_by_date(dates, 0.8)
    logger.info("切分: 训练 %d / 验证 %d", len(tidx), len(vidx))

    y_tr, y_va = y[tidx], y[vidx]
    date_va, ret_va = dates[vidx], rel_ret[vidx]

    val_prob: dict[str, np.ndarray] = {}
    member_meta: dict = {}
    models: dict = {}
    train_cfg = {**model_cfg, "epochs": int(mv2.get("epochs", model_cfg.get("epochs", 50)))}

    for name in members:
        if name in ("lstm", "transformer"):
            m_cfg = {**train_cfg, "type": name}
            res = _fit_model(TimeSeriesDataset(X[tidx], y_tr),
                             TimeSeriesDataset(X[vidx], y_va),
                             m_cfg, device)
            prob = _seq_probs(res["model"], X[vidx], device)
            val_prob[name] = prob
            models[name] = res["model"]
            member_meta[name] = {
                "threshold": float(res["best_threshold"]),
                "val_f1": round(float(res["val_f1"]), 4),
                "val_report": res["val_report"],
            }
            logger.info("成员 %s: 验证 F1=%.3f 阈值=%.2f",
                        name, res["val_f1"], res["best_threshold"])
        elif name == "gbm":
            Xtr2 = tabular_features(X[tidx])
            Xva2 = tabular_features(X[vidx])
            clf = _make_gbm(mv2.get("gbm", {}))
            clf.fit(Xtr2, y_tr)
            prob = clf.predict_proba(Xva2)[:, 1]
            val_prob[name] = prob
            models[name] = clf
            best_t, best_f1 = best_threshold(prob, y_va)
            member_meta[name] = {
                "threshold": float(best_t),
                "val_f1": round(float(best_f1), 4),
                "val_report": classification_report(y_va, prob, threshold=best_t),
            }
            logger.info("成员 gbm: 验证 F1=%.3f 阈值=%.2f", best_f1, best_t)
        else:
            raise ValueError(f"未知集成成员: {name}")

    # 软投票集成
    ens_prob = np.mean([val_prob[m] for m in members if m in val_prob], axis=0)
    ens_t, ens_f1 = best_threshold(ens_prob, y_va)
    ens_report = classification_report(y_va, ens_prob, threshold=ens_t)
    logger.info("集成: 验证 F1=%.3f 阈值=%.2f", ens_f1, ens_t)

    # 截面评估（验证段，逐日）
    cm_ens = _cross_metrics(date_va, ens_prob, ret_va)
    cm_mem = {m: _cross_metrics(date_va, val_prob[m], ret_va)
              for m in members if m in val_prob}
    logger.info("集成验证段: RankIC=%.4f ICIR=%.3f Top-Bottom=%.4f",
                cm_ens["rankic_mean"], cm_ens["icir"], cm_ens["top_bottom"])

    return {
        "X": X, "y": y, "rel_ret": rel_ret, "dates": dates, "symbols": symbols,
        "tidx": tidx, "vidx": vidx,
        "feature_columns": fcols, "window": window, "horizon": horizon,
        "members": members, "device": device,
        "val_prob": val_prob, "ens_prob": ens_prob,
        "ens_threshold": float(ens_t), "ens_f1": float(ens_f1),
        "ens_report": ens_report,
        "cross_metrics": {"ensemble": cm_ens, "members": cm_mem},
        "member_meta": member_meta,
        "label_mode": mv2.get("label_mode", "relative"),
        "cross_features": bool(mv2.get("cross_features", True)),
        "alpha_features": bool(mv2.get("alpha_features", True)),
        "_models": models, "_model_cfg": model_cfg,
    }


# ============================================================
# 保存 / 加载
# ============================================================
def save_ensemble(result: dict, out_dir: str | Path):
    """保存成员权重 + meta.json。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_cfg = result.get("_model_cfg", {})
    meta = {
        "feature_columns": result["feature_columns"],
        "window": result["window"],
        "horizon": result["horizon"],
        "members": result["members"],
        "label_mode": result["label_mode"],
        "cross_features": result.get("cross_features", True),
        "alpha_features": result.get("alpha_features", True),
        "hidden_size": model_cfg.get("hidden_size", 64),
        "num_layers": model_cfg.get("num_layers", 2),
        "dropout": model_cfg.get("dropout", 0.2),
        "ens_threshold": result["ens_threshold"],
        "ens_f1": result["ens_f1"],
        "ens_report": result["ens_report"],
        "cross_metrics": result["cross_metrics"],
        "member_meta": result["member_meta"],
        "val_n": int(len(result["vidx"])),
    }
    for name, model in result.get("_models", {}).items():
        if name == "gbm":
            with open(out_dir / "member_gbm.pkl", "wb") as f:
                pickle.dump(model, f)
        else:
            torch.save({"model_state": model.state_dict()},
                       out_dir / f"member_{name}.pt")
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("集成已保存到 %s (阈值 %.2f, 验证 F1 %.3f)",
                out_dir, meta["ens_threshold"], meta["ens_f1"])
    return out_dir


def load_ensemble(dir_path: str | Path) -> "CrossSectionalPredictor":
    """从目录加载集成预测器。"""
    dir_path = Path(dir_path)
    meta = json.loads((dir_path / "meta.json").read_text(encoding="utf-8"))
    pred = CrossSectionalPredictor.__new__(CrossSectionalPredictor)
    pred.dir_path = dir_path
    pred.meta = meta
    pred.window = int(meta["window"])
    pred.horizon = int(meta["horizon"])
    pred.feature_columns = meta["feature_columns"]
    pred.members = meta["members"]
    pred.label_mode = meta["label_mode"]
    pred.cross_features = bool(meta.get("cross_features", True))
    pred.alpha_features = bool(meta.get("alpha_features", True))
    pred.threshold = float(meta["ens_threshold"])
    pred.device = "cuda" if torch.cuda.is_available() else "cpu"
    pred.models = {}
    for name in pred.members:
        if name == "gbm":
            with open(dir_path / "member_gbm.pkl", "rb") as f:
                pred.models[name] = pickle.load(f)
        else:
            ck = torch.load(dir_path / f"member_{name}.pt", map_location="cpu")
            m = build_model(name, len(pred.feature_columns),
                            int(meta.get("hidden_size", 64)),
                            int(meta.get("num_layers", 2)),
                            float(meta.get("dropout", 0.2)))
            m.load_state_dict(ck["model_state"])
            m.eval()
            pred.models[name] = m.to(pred.device)
    return pred


# ============================================================
# 推理预测器（横截面，需一次喂全池）
# ============================================================
class CrossSectionalPredictor:
    """预测"跑赢当日全池中位数的概率"，需一次提供全池日线。

    用法与旧 ModelPredictor 对齐，多一个 make_signals_all：
        sig_all = predictor.make_signals_all(data)   # {symbol: DataFrame}
    """

    def __init__(self, dir_path: str | Path):
        p = load_ensemble(dir_path)
        self.__dict__.update(p.__dict__)

    @torch.no_grad()
    def make_signals_all(self, data: dict, threshold: float | None = None):
        """全池 → {symbol: DataFrame(date/close/prob_up/signal)}。"""
        feat_dict, _ = build_enhanced_features(
            data, self.window, self.cross_features, self.alpha_features,
            self.feature_columns)
        thr = self.threshold if threshold is None else threshold
        out: dict[str, pd.DataFrame] = {}
        for s, feat in feat_dict.items():
            bars = data[s]
            close_s = pd.Series(bars["close"].values,
                                index=pd.to_datetime(bars["date"])).sort_index()
            vals = feat.values.astype(np.float32)
            n = len(vals)
            if n < self.window:
                continue
            # 滑窗 → (n-window, window, F)
            X = np.stack([vals[i - self.window:i]
                          for i in range(self.window, n)]).astype(np.float32)
            probs = self._predict(X)
            dates = feat.index[self.window:]
            sig = pd.DataFrame({
                "close": close_s.reindex(dates).values,
                "prob_up": probs,
                "signal": (probs > thr).astype(int),
            }, index=dates)
            out[s] = sig
        logger.info("截面预测完成 %d 只 (阈值 %.2f)", len(out), thr)
        return out

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """对 (N, window, F) 返回集成概率。"""
        parts = []
        if "gbm" in self.models:
            parts.append(self.models["gbm"].predict_proba(
                tabular_features(X))[:, 1])
        seq_names = [m for m in self.members if m != "gbm"]
        if seq_names:
            for name in seq_names:
                parts.append(_seq_probs(self.models[name], X, self.device))
        return np.mean(parts, axis=0)

    def latest_scores(self, data: dict) -> dict[str, float]:
        """全池最新一天概率（收盘后/盘中均可，截面用当日最新可得价）。"""
        sig_all = self.make_signals_all(data)
        return {s: float(sig["prob_up"].iloc[-1])
                for s, sig in sig_all.items() if len(sig)}
