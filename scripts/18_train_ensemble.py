"""第 18 步：训练横截面增强模型 v2（相对标签 + 截面/Alpha101 特征 + LSTM/Transformer/GBM 集成），
并在验证段（后 20% 时间）与旧 LSTM 模型做 RankIC / 分层对照。

为什么：旧 LSTM 学"单股绝对涨跌"(acc≈0.5)，与"选股=横截面排序"任务错配。
v2 用"跑赢当日全池中位数"作为标签，特征是当日截面分位 + Alpha101/挖掘因子，
评估改用 RankIC/ICIR/Top-Bottom（accuracy 对近随机无意义）。

用法：
    python scripts/18_train_ensemble.py                  # 用库中数据训练（约 10~30 分钟，CPU）
    python scripts/18_train_ensemble.py --fetch          # 先拉最新日线再训练
    python scripts/18_train_ensemble.py --quick          # 冒烟：短训练快速验证流水线
    python scripts/18_train_ensemble.py --members lstm,gbm   # 只训指定成员

训练完成后重启 API 服务即自动加载 model_v2（api 优先 v2，缺省回退旧模型）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                # noqa: E402
import pandas as pd                                               # noqa: E402

from quant.config import load_config                              # noqa: E402
from quant.data.loader import load_all                            # noqa: E402
from quant.factors.analysis import _prepare_panels                # noqa: E402
from quant.models.cross_dataset import fwd_return_panel           # noqa: E402
from quant.models.cross_model import (                            # noqa: E402
    _cross_metrics, load_ensemble, save_ensemble, train_ensemble,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def _trim_recent(data: dict, recent: int) -> dict:
    """每只只保留最近 N 个交易日（recent<=0 表示全部）——控 make_samples 内存。"""
    if recent <= 0:
        return data
    return {c: df.tail(recent) for c, df in data.items()}


def _load_training_data(cfg, universe: str, recent: int) -> dict:
    """按 universe 载入训练截面：
        base  现池 40（market.db，原行为）
        large 现池40 ∪ data/large_pool.db 全量（约 600 池，阶段实验产物）
    """
    import sqlite3
    data = load_all(cfg)
    if universe == "large":
        db_path = Path(cfg.resolve("data")) / "large_pool.db"
        if not db_path.exists():
            raise SystemExit(
                "缺少 data/large_pool.db —— 请先运行 scripts/23_large_pool_data.py 构建 600 池日线")
        con = sqlite3.connect(str(db_path))
        for code, in con.execute("SELECT DISTINCT symbol FROM large_daily").fetchall():
            rows = con.execute(
                "SELECT date, open, high, low, close, volume, amount FROM large_daily "
                "WHERE symbol=? ORDER BY date", (code,)).fetchall()
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                             "close", "volume", "amount"])
            df["date"] = pd.to_datetime(df["date"])
            data[code] = df
        con.close()
        logger.info("universe=large：现池 %d + 大池 → %d 只", 40, len(data))
    return _trim_recent(data, recent)


def _panel_rankic(prob_panel: pd.DataFrame, ret_panel: pd.DataFrame,
                  val_dates) -> dict:
    """面板版截面评估：prob(date×symbol) vs 未来收益，在验证日期上逐日。"""
    dates = np.asarray([pd.Timestamp(d) for d in val_dates])
    ics, spreads = [], []
    for d in dates:
        if d not in prob_panel.index:
            continue
        p = prob_panel.loc[d].dropna()
        r = ret_panel.loc[d].reindex(p.index).dropna()
        both = p.index.intersection(r.index)
        if len(both) < 6:
            continue
        pp, rr = p[both].values, r[both].values
        from scipy.stats import spearmanr
        rho, _ = spearmanr(pp, rr)
        if not np.isfinite(rho):
            continue
        ics.append(rho)
        order = np.argsort(pp)
        k = max(1, len(pp) // 3)
        spreads.append(rr[order[-k:]].mean() - rr[order[:k]].mean())
    if not ics:
        return {"rankic_mean": 0.0, "icir": 0.0, "top_bottom": 0.0, "n_days": 0}
    arr = np.asarray(ics)
    m, s = float(arr.mean()), float(arr.std())
    return {"rankic_mean": round(m, 4), "icir": round(m / s, 3) if s > 0 else 0.0,
            "top_bottom": round(float(np.mean(spreads)), 4), "n_days": len(ics)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="先拉最新日线再训练")
    parser.add_argument("--quick", action="store_true", help="冒烟：缩短训练快速验证流水线")
    parser.add_argument("--members", default=None, help="覆盖集成成员，逗号分隔 lstm,transformer,gbm")
    parser.add_argument("--universe", default="base", choices=["base", "large"],
                        help="训练截面：base=现池40 | large=现池40∪600大池(需 large_pool.db)")
    parser.add_argument("--tag", default=None,
                        help="保存到 results/model_v2_<tag>（不覆盖/不备份线上 model_v2）")
    parser.add_argument("--recent", type=int, default=0,
                        help="每只只取最近 N 个交易日(0=全部)。large 全历史内存≈5.7GB，建议 900~1100")
    args = parser.parse_args()

    cfg = load_config()
    mv2 = cfg.get("model_v2", {})
    members = (args.members.split(",") if args.members
               else mv2.get("members", ["lstm", "transformer", "gbm"]))

    if args.fetch:
        from quant.data.fetcher import fetch_universe            # noqa: PLC0415
        from quant.data.storage import MarketDB                  # noqa: PLC0415
        data_cfg = cfg["data"]
        logger.info("拉取最新日线（end_date=%s）...", data_cfg["end_date"])
        df = fetch_universe(data_cfg["universe"], data_cfg["start_date"], data_cfg["end_date"])
        MarketDB(cfg.resolve(data_cfg["db_path"])).save_bars(df)

    # ---- 冒烟参数覆盖 ----
    if args.quick:
        cfg.to_dict().setdefault("model_v2", {})["epochs"] = 6
        mv2 = cfg.get("model_v2", {})
        mv2["gbm"] = {**mv2.get("gbm", {}), "n_estimators": 200}
    cfg.to_dict()["model_v2"] = {**cfg.get("model_v2", {}), "members": members}

    logger.info("=" * 64)
    logger.info("模型 v2 集成训练：%s", members)
    logger.info("=" * 64)

    data = _load_training_data(cfg, args.universe, args.recent)
    # 输出目录：--tag 或 universe=large 时另存（不碰线上 model_v2）；否则替换并备份旧版
    live_dir = cfg.resolve(mv2.get("dir", "results/model_v2"))
    tag = args.tag or ("large" if args.universe == "large" else None)
    out_dir = live_dir if tag is None else live_dir.with_name(live_dir.name + "_" + tag)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if tag is None:
        if out_dir.exists():                      # 备份旧 v2
            bak = out_dir.with_name(out_dir.name + ".bak")
            if bak.exists():
                import shutil
                shutil.rmtree(bak)
            import shutil
            shutil.move(str(out_dir), str(bak))
            logger.info("旧 v2 已备份到 %s", bak)
    else:
        logger.info("另存模式：%s（不动线上 model_v2）", out_dir)

    logger.info("训练截面 %d 只 · 输出 %s", len(data), out_dir)
    result = train_ensemble(data, cfg)
    result["_model_cfg"] = cfg["model"]
    save_ensemble(result, out_dir)

    # ---------- 新旧对照（验证段截面评估） ----------
    print("\n" + "=" * 64)
    print("验证段（后 20% 时间）截面评估对照")
    print("=" * 64)
    fcols = result["feature_columns"]
    window, horizon = result["window"], result["horizon"]
    # 构造 prob 面板（ensemble 在 val 段逐样本）→ panel
    vidx = result["vidx"]
    dates, symbols = result["dates"], result["symbols"]
    val_dates = np.unique(dates[vidx])
    ret_panel = fwd_return_panel(data, horizon)
    val_start = pd.Timestamp(min(val_dates))

    ens_panel = pd.DataFrame(
        {"symbol": symbols[vidx], "prob": result["ens_prob"],
         "date": dates[vidx]}).pivot_table(
        index="date", columns="symbol", values="prob", aggfunc="first")
    ens_panel.index = pd.to_datetime(ens_panel.index)

    cm = _panel_rankic(ens_panel, ret_panel, val_dates)
    print(f"  [v2 集成]     RankIC={cm['rankic_mean']:.4f}  ICIR={cm['icir']:.3f}"
          f"  Top-Bottom={cm['top_bottom']:.4f}  ({cm['n_days']}天)")

    # 旧 LSTM 对照
    old_path = cfg.resolve("results") / "lstm_model.pt"
    if old_path.exists() and not args.quick:
        try:
            from quant.models.predict import ModelPredictor        # noqa: PLC0415
            old = ModelPredictor(old_path)
            panel = {}
            for s, bars in data.items():
                sig = old.make_signal(bars)
                if len(sig):
                    panel[s] = sig["prob_up"]
            old_panel = pd.DataFrame(panel).sort_index()
            old_panel.index = pd.to_datetime(old_panel.index)
            cm_old = _panel_rankic(old_panel, ret_panel, val_dates)
            print(f"  [旧 LSTM]      RankIC={cm_old['rankic_mean']:.4f}"
                  f"  ICIR={cm_old['icir']:.3f}  Top-Bottom={cm_old['top_bottom']:.4f}"
                  f"  ({cm_old['n_days']}天)")
            print(f"  >>> 提升: RankIC {cm_old['rankic_mean']:.4f} → {cm['rankic_mean']:.4f}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("旧模型对照失败（可忽略）: %s", exc)
    else:
        print("  [旧 LSTM]      跳过（results/lstm_model.pt 不存在或 --quick）")

    # 各成员明细
    print("\n  成员明细:")
    for m, cmm in result["cross_metrics"]["members"].items():
        print(f"    {m:<12s} RankIC={cmm['rankic_mean']:.4f}  ICIR={cmm['icir']:.3f}"
              f"  Top-Bottom={cmm['top_bottom']:.4f}  F1={result['member_meta'][m]['val_f1']:.3f}")

    # 存报告
    report = {
        "feature_columns": fcols,
        "window": window, "horizon": horizon,
        "members": members,
        "ensemble": {"threshold": result["ens_threshold"], "f1": result["ens_f1"],
                     "report": result["ens_report"],
                     "cross_metrics": result["cross_metrics"]["ensemble"]},
        "members_detail": {m: {"cross_metrics": result["cross_metrics"]["members"][m],
                               **result["member_meta"][m]}
                           for m in members},
    }
    rep_name = "model_v2_report.json" if tag is None else f"model_v2_{tag}_report.json"
    rep_path = cfg.resolve("results") / rep_name
    rep_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n报告已保存: {rep_path}")
    print("\n==================== v2 训练完成 ====================")
    print(f"  集成保存: {out_dir}  (截面 {len(data)} 只)")
    print(f"  验证 F1={result['ens_f1']:.3f} 阈值={result['ens_threshold']:.2f}")
    if tag is None:
        print("\n  >>> 重启 API 服务即自动加载 v2：")
        print("      .venv/Scripts/python.exe -m uvicorn api.main:app --port 8001")
        print("  >>> 回退旧模型：删除/改名 results/model_v2 后重启即可。")
    else:
        print("\n  >>> 另存版本未接入线上。验证通过后切换方式：把 results/model_v2 换掉（先备份 .bak）再重启。")


if __name__ == "__main__":
    main()
