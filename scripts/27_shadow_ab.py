"""第 27 步：600 池影子 A/B —— model_v2(40池·线上) vs model_v2_large(600池·封存) 前向对照。

在**同一个 599 截面上**给两个模型各自打分（同特征、同参考池，苹果对苹果），
用真实未来 5 日收益逐日算 RankIC / ICIR / top-bottom / top命中，比较谁更能排序；
并把近 60 日逐日概率+收益落 results/shadow_ab.db，供日后累积（跑 2~4 周看趋势）。

用法（建议每天收盘后跑一次，先跑 scripts/26 保证大池数据最新）：
    python scripts/27_shadow_ab.py                 # 全近 window，读线上/600 两模型
    python scripts/27_shadow_ab.py --recent 700 --nostore

产物：results/shadow_ab_summary.json、results/shadow_ab.db(table shadow_ab)。
线上 model_v2 与页面完全不受影响。
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))       # noqa: E402

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy.stats import spearmanr                                     # noqa: E402

from quant.config import load_config                                  # noqa: E402
from quant.data.loader import load_all                                # noqa: E402
from quant.models.cross_model import load_ensemble                    # noqa: E402

cfg = load_config()
LARGE_DB = Path(cfg.resolve("data")) / "large_pool.db"
DB_PATH = Path(cfg.resolve("results")) / "shadow_ab.db"
MODELS = {
    "live40": cfg.resolve("results/model_v2"),
    "shadow600": cfg.resolve("results/model_v2_large"),
}
parser = argparse.ArgumentParser()
parser.add_argument("--recent", type=int, default=700,
                    help="每只只用最近 N 交易日（控评分耗时；影子历史验证建议 500~700）")
parser.add_argument("--nostore", action="store_true", help="不写 shadow_ab.db")
parser.add_argument("--model600", default=str(MODELS["shadow600"]), help="600 模型目录")
parser.add_argument("--model40", default=str(MODELS["live40"]), help="40 模型目录")
parser.add_argument("--cutoff", default="2026-09-07",
                    help="两模型的训练截止日。此前为样本内(会过拟合虚高，仅供参考)；"
                         "此后日期才算真正前向 OOS —— 每日跑本脚本，>cutoff 的日子会自然累积")
args = parser.parse_args()


def load_pool(recent: int) -> dict:
    data = load_all(cfg)                                # 现池 40（market.db）
    if LARGE_DB.exists():
        con = sqlite3.connect(str(LARGE_DB))
        for code, in con.execute("SELECT DISTINCT symbol FROM large_daily").fetchall():
            rows = con.execute(
                "SELECT date,open,high,low,close,volume,amount FROM large_daily "
                "WHERE symbol=? ORDER BY date", (code,)).fetchall()
            df = pd.DataFrame(rows, columns=["date", "open", "high", "low",
                                             "close", "volume", "amount"])
            df["date"] = pd.to_datetime(df["date"])
            data[code] = df
        con.close()
    if recent > 0:
        data = {c: df.tail(recent) for c, df in data.items()}
    # 过滤：历史太短无法建窗
    return {c: df for c, df in data.items() if len(df) > 100}


def panel_metrics(prob_panel: pd.DataFrame, ret: pd.DataFrame) -> dict:
    """逐日 IC 等：输入 date×symbol 的概率/未来收益。"""
    ics, spread, hit_top, cnt = [], [], [], 0
    idx = prob_panel.index.intersection(ret.index)
    for d in idx:
        p = prob_panel.loc[d].dropna()
        r = ret.loc[d].dropna()
        both = p.index.intersection(r.index)
        if len(both) < 6:
            continue
        pp, rr = p[both].values, r[both].values
        finite = np.isfinite(pp) & np.isfinite(rr)
        if finite.sum() < 6:
            continue
        pp, rr = pp[finite], rr[finite]
        rho, _ = spearmanr(pp, rr)
        if not np.isfinite(rho):
            continue
        med = np.median(rr)
        k = max(1, len(pp) // 10)
        order = np.argsort(pp)
        top, bot = rr[order[-k:]], rr[order[:k]]
        ics.append(rho)
        spread.append(float(top.mean() - bot.mean()))
        hit_top.append(float((top > med).mean()))
        cnt += 1
    ics = np.asarray(ics)
    return {"n_days": cnt,
            "rankic_mean": round(float(ics.mean()), 4) if cnt else 0.0,
            "icir": round(float(ics.mean() / ics.std()), 3) if cnt > 1 and ics.std() > 0 else 0.0,
            "top_bottom": round(float(np.mean(spread)), 4) if spread else 0.0,
            "hit_top_vs_median": round(float(np.mean(hit_top)), 4) if hit_top else 0.0}


def main():
    t0 = time.time()
    data = load_pool(args.recent)
    print(f"[27] 截面 {len(data)} 只（近 {args.recent} 交易日）", flush=True)

    # 公共收盘价面板（两模型共用未来收益）
    closes = {s: pd.Series(df["close"].values, index=pd.to_datetime(df["date"]))
              for s, df in data.items()}
    close_panel = pd.DataFrame(closes).sort_index()

    probs: dict[str, pd.DataFrame] = {}
    for tag, dpath in MODELS.items():
        dpath = args.model600 if tag == "shadow600" else args.model40
        pred = load_ensemble(dpath)
        sig_all = pred.make_signals_all(data)
        pmap = {s: sig["prob_up"] for s, sig in sig_all.items() if len(sig)}
        probs[tag] = pd.DataFrame(pmap).sort_index()
        print(f"[27] {tag} 打分完成 {len(pmap)} 只，日期 {probs[tag].index.min()}~{probs[tag].index.max()}",
              flush=True)

    # 统一评测日期（两模型共有，且能算出未来5日收益）
    horizon = 5
    pindex = probs["live40"].index.intersection(probs["shadow600"].index)
    cp = close_panel.reindex(pindex)
    ret = cp.shift(-horizon) / cp - 1.0

    cut = pd.Timestamp(args.cutoff)
    oos_index = pindex[pindex > cut]          # 真正前向：模型训练截止之后的日子

    def metrics_for(tag: str, index) -> dict:
        return panel_metrics(probs[tag].loc[index], ret)

    res_all = {tag: metrics_for(tag, pindex) for tag in MODELS}
    res_oos = ({tag: metrics_for(tag, oos_index) for tag in MODELS}
               if len(oos_index) else {})

    print("\n[27] == 全回测窗口（含样本内，仅供参考/易过拟合虚高） ==", flush=True)
    for tag, m in res_all.items():
        print(f"[27]   {tag:<10} RankIC={m['rankic_mean']:.4f} ICIR={m['icir']:.3f} "
              f"Top-Bot={m['top_bottom']:.4f} ({m['n_days']}天)", flush=True)
    if res_oos:
        print(f"\n[27] == 真正前向 OOS（>{args.cutoff}，模型没见过的日子） ==", flush=True)
        for tag, m in res_oos.items():
            print(f"[27]   {tag:<10} RankIC={m['rankic_mean']:.4f} ICIR={m['icir']:.3f} "
                  f"Top-Bot={m['top_bottom']:.4f} ({m['n_days']}天)", flush=True)
        a, b = res_oos["live40"], res_oos["shadow600"]
        print(f"[27] >>> 前向：影子600 ICIR {b['icir']:.3f} vs 线上40 {a['icir']:.3f}"
              f"（{'胜' if b['icir'] > a['icir'] else '负/平'}，n={b['n_days']}天）", flush=True)
    else:
        print(f"\n[27] 目前还没有 >{args.cutoff} 的前向日子（模型刚训到今天）。"
              f"每天收盘后跑一次本脚本，OOS 天数会从明天起累积。", flush=True)

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "n_symbols": len(data), "recent": args.recent, "cutoff": args.cutoff,
        "per_model_full": res_all,
        "per_model_oos": res_oos,
        "window_note": "近N交易日·未来5日真实收益。cutoff 前=样本内(虚高)，cutoff 后=真正前向",
    }
    Path(cfg.resolve("results")).mkdir(exist_ok=True)
    Path(cfg.resolve("results/shadow_ab_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 落库最近 ~60 个交易日的逐日概率（供后续累积/出图）
    if not args.nostore:
        con = sqlite3.connect(str(DB_PATH))
        con.executescript("""CREATE TABLE IF NOT EXISTS shadow_ab(
            date TEXT NOT NULL, symbol TEXT NOT NULL,
            prob40 REAL, prob600 REAL, ret5 REAL,
            PRIMARY KEY(date, symbol));""")
        keep = pindex[-60:]
        rows = []
        for d in keep:
            for s in probs["live40"].columns.intersection(probs["shadow600"].columns):
                try:
                    p40v = float(probs["live40"].loc[d, s]) if pd.notna(probs["live40"].loc[d, s]) else None
                    p60v = float(probs["shadow600"].loc[d, s]) if pd.notna(probs["shadow600"].loc[d, s]) else None
                    r5 = float(ret.loc[d, s]) if pd.notna(ret.loc[d, s]) else None
                except (KeyError, TypeError):
                    continue
                rows.append((d.strftime("%Y-%m-%d"), s, p40v, p60v, r5))
        con.executemany("INSERT OR REPLACE INTO shadow_ab(date,symbol,prob40,prob600,ret5)"
                        " VALUES(?,?,?,?,?)", rows)
        con.commit()
        n = con.execute("SELECT COUNT(*) FROM shadow_ab").fetchone()[0]
        con.close()
        print(f"[27] 落库 shadow_ab.db：{n} 行（近60交易日·双模型概率+ret5）", flush=True)

    print(f"[27] 总耗时 {time.time() - t0:.0f}s → results/shadow_ab_summary.json", flush=True)


if __name__ == "__main__":
    main()
