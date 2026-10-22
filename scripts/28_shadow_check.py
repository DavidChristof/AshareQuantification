"""第 28 步：判断「600 池模型是否已在前向 OOS 稳压」→ 决定是否可以提醒上线。

读 scripts/27 累积的 results/shadow_ab.db（date,symbol,prob40,prob600,ret5），
只取 cutoff 之后、且已有未来 5 日收益(ret5)的日子，逐日算两模型的 RankIC，
按「稳压」规则判定：
    - OOS 天数 ≥ min_days（默认 10 天）
    - shadow600 的 OOS 平均 RankIC > 0.015
    - shadow600 平均 RankIC > live40 平均 RankIC（稳定更会排序）
    - shadow600 OOS ICIR > live40 OOS ICIR
    - **配对 t（HAC 修正）≥ min_paired_t（默认 2.0）**  <- 2026-10-22 新增
全部满足 → STABLE（提醒上线）；否则输出还需多少天/差距。

[!] 为什么要加最后一条（2026-10-22）：前四条都是**点估计比较**，不含任何显著性检验。
10-19~10-21 连续三天报了 STABLE，而同一批数据的配对检验是 t=+0.82 (p=0.42) ——
所谓"交叉"只是**一次 regime 反转**（09-14~09-24 是 40 完胜，09-28~10-14 是 600 全胜）。
点估计规则在行情反转时必然误报一次，光靠"再多等几天"治不了。

[!] 为什么用 HAC（Newey-West）而不是普通 t：每天的 RankIC 用的是**未来 5 日收益**，
相邻两天共享 4/5 的收益窗口 => 序列自相关，普通 t 把 n 天的重叠样本当成 n 个独立
观测，**系统性高估**显著性。lag 取 horizon-1 = 4，Bartlett 权重。

用法：python scripts/28_shadow_check.py [--cutoff 2026-09-07] [--min-days 10]
                                    [--min-paired-t 2.0] [--hac-lag 4]
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
parser.add_argument("--min-paired-t", type=float, default=2.0,
                    help="配对 t 的门槛（单侧，600 领先）；默认 2.0")
parser.add_argument("--hac-lag", type=int, default=4,
                    help="Newey-West 滞后阶数；默认 4 = 5 日前向收益的重叠补偿")
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


def hac_t(diff: np.ndarray, lag: int) -> tuple[float, float]:
    """差分序列的均值 + Newey-West(Bartlett 权重) HAC t 值。

    Args:
        diff: 逐日差值（600 的 RankIC - 40 的 RankIC）。
        lag: 自相关补偿阶数。5 日前向收益 => 相邻两天共享 4/5 的收益窗口 => lag=4。

    **为什么不能用普通 t**：重叠窗口让序列正自相关，普通 t 把重叠样本当成独立观测，
    系统性**高估**显著性。例（2026-10-22，n=21）：普通 t=+0.82；HAC 修正后更小。
    非独立样本上谈"n 天"，本来就没有那么多信息量。
    """
    d = np.asarray(diff, dtype=float)
    n = len(d)
    if n < 2:
        return (float(d.mean()) if n else 0.0, float("nan"))
    m = float(d.mean())
    x = d - m
    var_plain = float(x @ x) / n
    if var_plain <= 0:
        return m, float("nan")
    var = var_plain
    for k in range(1, min(lag, n - 1) + 1):
        w = 1.0 - k / (lag + 1.0)
        var += 2.0 * w * float(x[k:] @ x[:-k]) / n
    if var <= 0:                 # 短样本下 HAC 可能给出非正方差 -> 回退普通方差
        var = var_plain
    se = float(np.sqrt(var / n))
    return m, (m / se if se > 0 else float("nan"))


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

    # ---- 配对显著性：点估计领先 ≠ 显著领先 ----
    paired = None
    if nd >= 3:
        diffs = (ic60 - ic40).to_numpy(dtype=float)
        mdiff, t_paired = hac_t(diffs, args.hac_lag)
        wins = int((diffs > 0).sum())
        paired = {"mean": mdiff, "t": t_paired, "wins": wins, "n": nd}
        print(f"[28] 配对（逐日 RankIC 差 600-40，n={nd}，HAC lag={args.hac_lag}）：", flush=True)
        print(f"      均值差={mdiff:+.4f}  t(HAC)={t_paired:+.2f}  600赢 {wins}/{nd} 天",
              flush=True)
        print("      注：日窗 5 日重叠 => 有效独立观测远少于 n，HAC 已按 lag 修正自相关。",
              flush=True)

    checks = {
        "OOS天数够": nd >= args.min_days,
        "600有正IC": m60 > args.min_rankic,
        "600>40(平均IC)": m60 > m40,
        "600>40(ICIR)": i60 > i40,
        # 点估计领先**还不够**：必须显著领先。算不出来（样本太少/NaN）也按不通过处理
        # （fail closed）—— 宁可多等，也不要因为"没算"而放行上线。
        f"600>40(配对t>={args.min_paired_t:g})": bool(
            paired and np.isfinite(paired["t"]) and paired["t"] >= args.min_paired_t),
    }
    for k, v in checks.items():
        # 只能用 ASCII：本机控制台是 GBK，checkmark(码位 U+2713) 和 ballot-X(U+2717)
        # 都不在 GBK 里，print 会抛 UnicodeEncodeError —— 而这里恰好挡在 STABLE/NOT_YET
        # 判定**之前**，于是「结论永远打不出来」。2026-09-15 实测就是这样：
        # GBK 下 28 崩在检查表，兜底定时任务因此看不到 STABLE/NOT_YET。
        # 中文本身在 GBK 内，无需改。（此处刻意不写那两个字符本身，见 docs 同名说明。）
        print(f"      [{'OK' if v else '--'}] {k}", flush=True)
    stable = all(checks.values())
    if stable:
        print("\nSTABLE：600 池模型已在真正前向 OOS 稳压（>线上40）。→ 提醒用户：可以上线新版600池了。", flush=True)
    else:
        need = max(args.min_days - nd, 0)
        msg = (f"NOT_YET：尚未稳压。预计还需 ≥{need} 个 OOS 可评日子；"
               "继续每天跑 26→27。")
        if paired and np.isfinite(paired["t"]) and paired["t"] < args.min_paired_t:
            msg += (f"\n  卡在显著性：配对 t={paired['t']:+.2f} < {args.min_paired_t:g}"
                    " —— 点估计领先不等于显著领先（日窗 5 日重叠，别按天数算信息量）。")
        print("\n" + msg, flush=True)
    return 0 if stable else 1


if __name__ == "__main__":
    sys.exit(main())
