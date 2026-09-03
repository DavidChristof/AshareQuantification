"""第 25 步：分钟模型 v2 重训 + 概率校准（Platt）。

背景（诊断 24）：旧分钟模型 sigmoid 过饱和——近 2 天 720 个概率 98% 判 sell、
中位 0.003（真实上涨 35%+ 却输出 0.003，属过拟合+评估乱序假象）。分钟数据仅
~25 个交易日（单边下跌段），样本少。

v2 修复：
    1. 严格按时间切分（训练样本时间 < 验证，杜绝旧 train_model 行序乱切的泄漏假象）
    2. 防过拟合：epochs 封顶 + patience 早停（小样本不硬训满）
    3. Platt 校准：p = sigmoid(A*logit + B)，把输出拉回真实频率区间，
       checkpoint 存 calib，预测端自动应用（predict.ModelPredictor 已支持）
    4. 诚实输出：val 上 sigmoid vs 校准后概率分布、AUC、正类基率

用法：python scripts/25_retrain_minute_v2.py [--epochs 30]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from quant.config import load_config                                   # noqa: E402
from quant.features.pipeline import FeaturePipeline                    # noqa: E402
from quant.models.dataset import TimeSeriesDataset                     # noqa: E402
from quant.models.trainer import _fit_model, save_checkpoint, set_seed  # noqa: E402
from quant.realtime.minute_store import MinuteStore                    # noqa: E402

SCALE = 5
WINDOW = 30
HORIZON = 5


def collect(data: dict, symbol_list: list[str]):
    """全股票分钟数据 → (X, y, times)；按时间升序拼好。"""
    pipe = FeaturePipeline(window=WINDOW, horizon=HORIZON)
    Xs, ys, ts = [], [], []
    for symbol in symbol_list:
        df = data.get(symbol)
        if df is None or len(df) < WINDOW + 2:
            continue
        for _, day_df in df.groupby(df["datetime"].dt.date):
            if len(day_df) < WINDOW + 2:
                continue
            d = day_df.copy()
            d["date"] = d["datetime"]
            X, y, D = pipe.make_windows(d)
            mask = ~np.isnan(y)
            if mask.sum() > 0:
                Xs.append(X[mask])
                ys.append(y[mask])
                ts.append(D[mask].astype("datetime64[ns]"))
    X = np.concatenate(Xs).astype(np.float32)
    y = np.concatenate(ys).astype(np.float32)
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return X[order], y[order], t[order]


@torch.no_grad()
def val_logits(model, Xv, bs=512, device="cpu"):
    model.eval()
    out = []
    for i in range(0, len(Xv), bs):
        xb = torch.from_numpy(Xv[i:i + bs]).float().to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="训练 epoch 上限")
    parser.add_argument("--days", type=int, default=600, help="回溯分钟数据天数上限")
    args = parser.parse_args()

    cfg = load_config()
    mc = cfg.get("minute", {})
    store = MinuteStore(cfg.resolve(mc.get("db_path", "data/minute.db")))
    universe = cfg["data"]["universe"]

    print("[1/4] 加载分钟数据（按时间升序切窗）...")
    df_map = {}
    for s in universe:
        try:
            d = store.load_symbol(s, scale=SCALE, days=args.days)
            if d is not None and len(d) >= WINDOW + 2:
                df_map[s] = d
        except Exception:  # noqa: BLE001
            pass
    X, y, t = collect(df_map, list(df_map.keys()))
    print(f"  样本 X={X.shape} 正样本 {y.mean()*100:.1f}%  覆盖 {len(df_map)} 只, "
          f"时间 {np.datetime_as_string(t[0])[:16]} ~ {np.datetime_as_string(t[-1])[:16]}")
    if len(y) < 2000:
        raise RuntimeError("分钟样本过少，无法训练")

    # 按时间切 80/20（训练恒在验证前）
    cut = int(len(y) * 0.8)
    Xtr, Xva, ytr, yva = X[:cut], X[cut:], y[:cut], y[cut:]
    print(f"[2/4] 时间切分: 训练 {len(ytr)} / 验证 {len(yva)} "
          f"(训练末尾 {np.datetime_as_string(t[cut-1])[:16]} < 验证起 {np.datetime_as_string(t[cut])[:16]})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(int(cfg["model"].get("seed", 42)))
    m_cfg = {**cfg["model"], "type": "lstm", "epochs": args.epochs, "patience": 6}
    res = _fit_model(TimeSeriesDataset(Xtr, ytr), TimeSeriesDataset(Xva, yva),
                     m_cfg, device)
    logger.info("验证 F1=%.3f 阈值=%.2f", res["val_f1"], res["best_threshold"])

    print("[3/4] Platt 校准（在验证集拟合 A,B）...")
    logit = val_logits(res["model"], Xva, device=device)
    yva_np = yva
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e9, max_iter=1000)   # 大 C ≈ 无正则
    lr.fit(logit.reshape(-1, 1), yva_np.astype(int))
    A, B = float(lr.coef_[0][0]), float(lr.intercept_[0])
    lo = float(np.percentile(logit, 1))      # 验证集 logit 截断区间（防分布外极端）
    hi = float(np.percentile(logit, 99))
    raw_p = 1.0 / (1.0 + np.exp(-logit))
    clip = np.clip(logit, lo, hi)
    cal_p = 1.0 / (1.0 + np.exp(-(A * clip + B)))
    from sklearn.metrics import roc_auc_score
    auc_raw = roc_auc_score(yva_np.astype(int), raw_p)
    auc_cal = roc_auc_score(yva_np.astype(int), cal_p)
    print(f"  校准 A={A:.3f} B={B:.3f} logit截断区间 [{lo:.2f},{hi:.2f}]"
          f"（校准后 mean={cal_p.mean():.4f}, 正类基率 {yva_np.mean():.3f}）")
    print(f"  验证集 AUC: 校准前={auc_raw:.4f}  校准后={auc_cal:.4f}  （0.5=无信息）")
    for q in (10, 50, 90):
        print(f"    校准后 {q}% 分位 = {np.percentile(cal_p, q):.4f}")

    print("[4/4] 保存 minute_model_v2.pt（含 calib）...")
    pipe = FeaturePipeline(window=WINDOW, horizon=HORIZON)
    out = cfg.resolve("results") / "minute_model.pt"
    if out.exists():                      # 备份旧模型
        bak = out.with_suffix(".pt.bak")
        import shutil
        shutil.copy2(out, bak)
    save_checkpoint(res["model"], out, m_cfg,
                    feature_columns=pipe.feature_columns,
                    window=WINDOW, horizon=HORIZON,
                    best_threshold=0.5,
                    calib={"A": A, "B": B, "lo": lo, "hi": hi})
    print("\n============ 分钟模型 v2 完成 ============")
    print(f"  保存: {out}（旧模型备份 {out.with_suffix('.pt.bak')}）")
    print(f"  校准 A={A:.3f} B={B:.3f} logit[{lo:.2f},{hi:.2f}]  AUC={auc_cal:.4f}")
    print("  请重启 API 服务加载新分钟模型（预测端自动应用 Platt 校准）")


if __name__ == "__main__":
    main()
