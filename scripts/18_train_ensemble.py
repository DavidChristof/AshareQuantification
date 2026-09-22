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

# pit500 抽样种子 —— **固定**，保证「抽了哪 600 只」可复现、可审计。
# 改动它会换一批票，等于换了实验样本；除非有理由，不要动。
_PIT_SAMPLE_SEED = 20260922


def _trim_recent(data: dict, recent: int) -> dict:
    """每只只保留最近 N 个交易日（recent<=0 表示全部）——控 make_samples 内存。"""
    if recent <= 0:
        return data
    return {c: df.tail(recent) for c, df in data.items()}


def _load_pit500(cfg, n_names: int = 0) -> dict:
    """PIT 中证500 层的**并集**（约 1048 只）—— 无幸存者偏差的训练截面。

    为什么**不**走 load_all()：那会先把「现池 40」装进来，而现池 40 是**今天**的名单，
    等于往无偏池里混进一小撮幸存票。这里起点就是空 dict。

    为什么用**并集**而不是「只按当日成分」：`make_samples` 没有 per-date 掩码接口，
    而**并集本身就消除了幸存者偏差** —— 它包含**后来被调出指数**的票，那正是偏差的来源。
    掩码只影响标签中位数与横截面构成，不影响无偏性，不值得为此改共享模块。

    n_names > 0 时按**固定种子**抽样那么多个。抽样仍无偏（从无偏集合里随机抽），
    且能顺带消掉「只数不同」这个混淆 —— 见下面的内存说明。

    局限（见 docs/2026-09-11-pit-universe.md）：与真实中证500 重合度仅 76.6%，
    它是「**无未来函数的同区间宇宙**」，不是真实指数的复制品。
    """
    import sqlite3
    db_path = Path(cfg.resolve("data")) / "full_market.db"
    if not db_path.exists():
        raise SystemExit("缺少 data/full_market.db —— 请先运行 scripts/35_fetch_full_market.py")
    # 只读打开：full_market.db 是冻结的行情库。绝不能被建表/写入
    # （paper 那次事故就是「打开」会 CREATE TABLE，把账户表建进了行情库）。
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM pit_members WHERE index_code='csi500'").fetchall()]
    if not syms:
        con.close()
        raise SystemExit("pit_members 里没有 csi500 —— 请先运行 "
                         "scripts/36_validate_pit_universe.py")
    total = len(syms)
    if n_names and n_names < total:
        # [!] 为什么要抽样：训练内存大头（因子面板 + make_samples 的 X）随
        # 「**只数 x 天数**」走 —— 全量 1048 只 @ recent500 实测要 ~9.8 GB，
        # 而本机只有 15.6 GB、常驻占用后剩 4~6 GB，装不下。
        # 抽到与对照池（599 只）同规模：内存可跑，且两组只数匹配，判据更干净。
        import random
        syms = sorted(random.Random(_PIT_SAMPLE_SEED).sample(sorted(syms), n_names))
        logger.info("universe=pit500：从 %d 只中抽样 %d 只（seed=%d，可复现）",
                    total, len(syms), _PIT_SAMPLE_SEED)
    data = {}
    for code in syms:
        rows = con.execute(
            "SELECT date, open, high, low, close, volume, amount FROM full_daily "
            "WHERE symbol=? ORDER BY date", (code,)).fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                         "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        data[code] = df
    con.close()
    logger.info("universe=pit500：PIT 中证500 并集 %d 只（无幸存者偏差）", len(data))
    return data


def _trim_window(data: dict, start: str | None, end: str | None) -> dict:
    """按**日历窗口**裁剪每只票的行（闭区间，start/end 为 'YYYY-MM-DD'）。

    [!] 为什么必须有它：`--recent` 是按**每只票自己的最后 N 天**切，而不同池子的数据新鲜度
    不同（large_pool 到 2026-09-22、full_market 到 2026-09-11），于是两组切出来的
    **验证段不是同一段日期** —— 那样「池子」这个变量的比较就不是干净的。
    要升级成**配对级证据**，两组必须钉在同一段日历上。
    """
    if not start and not end:
        return data
    lo = pd.Timestamp(start) if start else None
    hi = pd.Timestamp(end) if end else None
    out: dict = {}
    for c, df in data.items():
        dts = pd.to_datetime(df["date"])          # 有的 loader 给字符串，有的给 Timestamp
        m = pd.Series(True, index=df.index)
        if lo is not None:
            m &= (dts >= lo)
        if hi is not None:
            m &= (dts <= hi)
        d = df[m]
        if len(d):
            out[c] = d
    return out


def _load_training_data(cfg, universe: str, recent: int, pit_names: int = 0,
                        start: str | None = None, end: str | None = None) -> dict:
    """按 universe 载入训练截面：
        base   现池 40（market.db，原行为）
        large  现池40 ∪ data/large_pool.db 全量（约 600 池，阶段实验产物）
        pit500 PIT 中证500 层（无幸存者偏差，见 _load_pit500；pit_names>0 则固定种子抽样）

    start/end：按**日历**裁剪（见 _trim_window），两组对齐时用。
    """
    import sqlite3
    if universe == "pit500":
        d = _trim_window(_load_pit500(cfg, pit_names), start, end)
        return _trim_recent(d, recent)
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
    return _trim_recent(_trim_window(data, start, end), recent)


# ============================================================
# 内存闸门
# ============================================================
# 每「行」的峰值内存系数（MB/行）—— **实测标定**，不是推导出来的。
#
# 标定点（2026-09-22）：universe=large（599 只）x recent500 = 297,195 行，
# 实测训练进程工作集 **4,690 MB**，即 16.2 KB/行。取 15% 余量后为 18.6 KB/行。
#
# 为什么不按「样本数 x X 大小」估：真正的大头是 `build_enhanced_features` 物化的
# Alpha101/挖掘因子面板，它随「**只数 x 天数**」增长（就是行数），**不是**随样本数。
# 我第一版按 X x 1.3 估，对同一个标定点只给出 2,096 MB —— **低估 2.2 倍**，
# 于是闸门形同虚设（那次幸好撞上的是冒烟，不是全量；否则跑到一半 OOM，白等十几分钟）。
_MB_PER_ROW = 0.0162 * 1.15


def _free_mem_mb() -> float:
    """当前可用物理内存（MB）。"""
    import ctypes

    class _MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = _MS()
    m.dwLength = ctypes.sizeof(_MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 1048576.0


def _estimate_peak_mb(data: dict) -> float:
    """估训练峰值内存（MB）—— 按**行数**（只数 x 天数）线性外推，系数实测标定。

    见上面 `_MB_PER_ROW`：内存大头是因子面板，随「只数 x 天数」增长，不是随样本数。
    """
    return sum(len(df) for df in data.values()) * _MB_PER_ROW


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
    parser.add_argument("--universe", default="base", choices=["base", "large", "pit500"],
                        help="训练截面：base=现池40 | large=现池40∪600大池(需 large_pool.db) | "
                             "pit500=PIT中证500层(无幸存者偏差，需 full_market.db，**必须配 --tag**)")
    parser.add_argument("--tag", default=None,
                        help="保存到 results/model_v2_<tag>（不覆盖/不备份线上 model_v2）")
    parser.add_argument("--recent", type=int, default=0,
                        help="每只只取最近 N 个交易日(0=全部)。large 全历史内存≈5.7GB，建议 900~1100")
    parser.add_argument("--pit-names", type=int, default=600,
                        help="universe=pit500 时固定种子抽样多少只（0=全部 1048 只）。"
                             "默认 600：与对照池同规模、内存可跑；"
                             "全量 1048@recent500 实测要 ~9.8GB，本机 15.6GB 装不下")
    parser.add_argument("--allow-tight", action="store_true",
                        help="内存预估不足时只警告、不中断（默认直接拒绝：宁可不跑，也别撞 OOM）")
    parser.add_argument("--start", default=None,
                        help="只保留该日期(含)之后的行 —— 按**日历**裁，用于把两组钉在同一段日期上")
    parser.add_argument("--end", default=None,
                        help="只保留该日期(含)之前的行。与 --start 合用可消掉两组验证段错位")
    args = parser.parse_args()

    # [!] 安全闸：pit500 必须显式给 --tag。
    # 下面的 tag 推导是 `args.tag or ("large" if universe=="large" else None)`，
    # 而 tag=None 会走「**替换并备份线上 results/model_v2**」那条分支 ——
    # 绝不能因为加了个新池子就把线上模型覆盖掉（本项目约定：模型切换只提醒、绝不自动做）。
    if args.universe == "pit500" and not args.tag:
        raise SystemExit("--universe pit500 必须同时指定 --tag（例如 --tag pit500），"
                         "否则会覆盖线上 results/model_v2。")

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

    data = _load_training_data(cfg, args.universe, args.recent, args.pit_names,
                               args.start, args.end)

    # ---- 内存前置检查：宁可不跑，也别撞 OOM ----
    # make_samples 会把整个池子的窗口一次性物化，撞上去就是 MemoryError（或把系统拖垮）。
    # 估不准没关系，只要方向保守：估高一点，宁可让你关个程序，也不要在跑到一半时炸掉。
    need_mb = _estimate_peak_mb(data)
    free_mb = _free_mem_mb()
    logger.info("内存闸门：%d 只 / 峰值预估 %.0f MB / 当前可用 %.0f MB",
                len(data), need_mb, free_mb)
    if free_mb < need_mb:
        if not args.allow_tight:
            raise SystemExit(
                f"可用内存不足：预估需要约 {need_mb:.0f} MB，当前只有 {free_mb:.0f} MB。\n"
                f"  请关掉占内存的程序（浏览器/游戏/多余的编辑器窗口）后重试；\n"
                f"  或减小 --recent（当前 {args.recent or '全部'}）/ --pit-names；\n"
                f"  确认能跑也可以显式加 --allow-tight（风险自担）。")
        logger.warning("内存预估 %.0f MB > 可用 %.0f MB —— 已指定 --allow-tight，继续训练",
                       need_mb, free_mb)

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
