"""第 28 步：判断「600 池模型是否已在前向 OOS 稳压」→ 决定是否可以提醒上线。

读 scripts/27 累积的 results/shadow_ab.db（date,symbol,prob40,prob600,ret5），
只取 cutoff 之后、且已有未来 5 日收益(ret5)的日子，逐日算两模型的 RankIC，
按「稳压」规则判定：
    - OOS 天数 ≥ min_days（默认 10 天）
    - shadow600 的 OOS 平均 RankIC > 0.015
    - shadow600 平均 RankIC > live40 平均 RankIC（稳定更会排序）
    - shadow600 OOS ICIR > live40 OOS ICIR
全部满足 → STABLE（提醒上线）；否则输出还需多少天/差距。

用法：python scripts/28_shadow_check.py [--cutoff 2026-09-07] [--min-days 10]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))       # noqa: E402

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy.stats import spearmanr                                     # noqa: E402

from quant.config import load_config                                  # noqa: E402

cfg = load_config()
DB = Path(cfg.resolve("results")) / "shadow_ab.db"
parser = argparse.ArgumentParser()
parser.add_argument("--cutoff", default="2026-09-07")
parser.add_argument("--min-days", type=int, default=10)
parser.add_argument("--min-rankic", type=float, default=0.015)
args = parser.parse_args()


def daily_ic(df: pd.DataFrame) -> pd.Series:
    """每日期望排名相关：对 (date,symbol,prob,ret5) 逐日 spearman。"""
    ics = {}
    for d, g in df.groupby("date"):
        g = g.dropna(subset=["prob", "ret5"])
        if len(g) < 6:
            continue
        if g["ret5"].nunique() < 3 or g["prob"].nunique() < 3:
            continue
        rho, _ = spearmanr(g["prob"], g["ret5"])
        if np.isfinite(rho):
            ics[d] = rho
    return pd.Series(ics, dtype=float)


def main():
    if not DB.exists():
        print("NO_DB：shadow_ab.db 还不存在——先跑 scripts/27（每日收盘后）累积。")
        return 1
    con = sqlite3.connect(str(DB))
    rows = con.execute("SELECT date,symbol,prob40,prob600,ret5 FROM shadow_ab").fetchall()
    con.close()
    df = pd.DataFrame(rows, columns=["date", "symbol", "prob40", "prob600", "ret5"])
    df["date"] = pd.to_datetime(df["date"])
    oos = df[df["date"] > pd.Timestamp(args.cutoff)]
    done = oos.dropna(subset=["ret5"])          # 已有未来5日收益 → 可评
    print(f"[28] 累积：OOS 日子 {oos['date'].nunique()}（其中已有收益 {done['date'].nunique()}）",
          flush=True)
    if done.empty:
        print(f"[28] 前向 OOS 尚无可评日子（需 >{args.cutoff} 且已过未来5日）。"
              "请每天收盘后跑 26→27 累积，过几天再看。", flush=True)
        return 1
    ic40 = daily_ic(done[["date", "symbol", "prob40", "ret5"]].rename(columns={"prob40": "prob"}))
    ic60 = daily_ic(done[["date", "symbol", "prob600", "ret5"]].rename(columns={"prob600": "prob"}))
    common = ic40.index.intersection(ic60.index)
    ic40, ic60 = ic40[common], ic60[common]
    nd = len(common)
    m40, m60 = float(ic40.mean()), float(ic60.mean())
    s40 = float(ic40.std()) if nd > 1 else 0.0
    s60 = float(ic60.std()) if nd > 1 else 0.0
    i40 = m40 / s40 if s40 > 0 else 0.0
    i60 = m60 / s60 if s60 > 0 else 0.0
    print(f"[28] 前向 OOS（> {args.cutoff}，{nd} 天）：", flush=True)
    print(f"      live40     平均RankIC={m40:.4f}  ICIR={i40:.3f}", flush=True)
    print(f"      shadow600  平均RankIC={m60:.4f}  ICIR={i60:.3f}", flush=True)

    checks = {
        "OOS天数够": nd >= args.min_days,
        "600有正IC": m60 > args.min_rankic,
        "600>40(平均IC)": m60 > m40,
        "600>40(ICIR)": i60 > i40,
    }
    for k, v in checks.items():
        print(f"      [{('✓' if v else '✗')}] {k}", flush=True)
    stable = all(checks.values())
    if stable:
        print("\nSTABLE：600 池模型已在真正前向 OOS 稳压（>线上40）。→ 提醒用户：可以上线新版600池了。", flush=True)
    else:
        need = max(args.min_days - nd, 0)
        print(f"\nNOT_YET：尚未稳压。预计还需 ≥{need} 个 OOS 可评日子；"
              "继续每天跑 26→27。", flush=True)
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
