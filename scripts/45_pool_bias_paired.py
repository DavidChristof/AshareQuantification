"""第 45 步：池子偏差的**配对**检验 —— 逐日 IC 之差做单样本 t。

## 为什么需要它

`scripts/44` 用的是**非配对**公式 `se = sqrt(se_A^2 + se_B^2)`，它丢掉了两个模型的
逐日 IC 序列之间的**相关性**。而两组是在**同一天**评价的（日期窗口已对齐到同一天起止），
两串 IC 必然同向波动（同一个市场、同一批日子、排序的票也高度重叠）。

正确的做法（与 `scripts/37_risk_rerun_pit.py` 的 `_paired` 同一手法）：

    d_t = IC_B(t) - IC_A(t)          # 逐日之差
    t   = mean(d) / (std(d) / sqrt(n))

丢掉相关性会**高估** se、从而**低估**显著性 —— 44 报的 z=1.24 可能因此偏保守。

## 它不做的事

- **不重训**：只加载两个已训好的模型做推理（几十秒）。训练由 `scripts/18` 负责。
- **不动线上模型**：只读 `results/model_v2_<tag>`。

## 用法

    .venv/Scripts/python.exe scripts/45_pool_bias_paired.py \\
        --a large500al --b pit500al --start 2024-09-01 --end 2026-09-11
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from quant.config import load_config                                # noqa: E402
from quant.models.cross_model import load_ensemble                  # noqa: E402

# 与 scripts/27 同一门槛：一天至少要有 6 只可比才值得算 IC
_MIN_NAMES = 6


def _s18():
    """导入 scripts/18 —— 保证两组用的是**完全相同**的载入/裁剪/抽样逻辑。"""
    p = Path(__file__).resolve().parent / "18_train_ensemble.py"
    spec = importlib.util.spec_from_file_location("s18", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _daily_ic(prob: pd.DataFrame, close: pd.DataFrame, horizon: int) -> pd.Series:
    """逐日截面 Spearman IC（prob 与未来 horizon 日收益）。"""
    from scipy.stats import spearmanr

    ret = close.shift(-horizon) / close - 1.0
    idx = prob.index.intersection(ret.index)
    out = {}
    for d in idx:
        p, r = prob.loc[d].dropna(), ret.loc[d].dropna()
        both = p.index.intersection(r.index)
        if len(both) < _MIN_NAMES:
            continue
        pp, rr = p[both].to_numpy(), r[both].to_numpy()
        ok = np.isfinite(pp) & np.isfinite(rr)
        if ok.sum() < _MIN_NAMES:
            continue
        rho, _ = spearmanr(pp[ok], rr[ok])
        if np.isfinite(rho):
            out[d] = float(rho)
    return pd.Series(out).sort_index()


def _val_dates(all_dates: pd.DatetimeIndex, window: int, horizon: int) -> pd.DatetimeIndex:
    """复刻 `split_by_date(dates, 0.8)` 的**验证段**日期。

    [!] 为什么必须做这一步：`make_signals_all` 会对**整个窗口**出信号，若不裁到验证段，
    算出来的 IC 就**大部分是训练段的记忆** —— 实测不裁时均值 IC = +0.52
    （真实水平 0.03，0.52 是个假数），而且配对 t 会给出「显著」的假结论。
    这正是 27 报 shadow600「样本内 0.3702」的同一个陷阱。

    样本日期 ≈ 数据日期去掉开头的 window（建窗）与结尾的 horizon（前向收益未落地）。
    """
    sample_dates = all_dates[window: len(all_dates) - horizon]
    if len(sample_dates) < 10:
        return sample_dates
    cut = sample_dates[int(len(sample_dates) * 0.8)]
    return sample_dates[sample_dates > cut]


def _score(cfg, tag: str, universe: str, pit_names: int,
           start: str, end: str, recent: int) -> pd.Series:
    """载入池子 → 推理 → **验证段**逐日 IC。用完即释放（省内存）。"""
    m = _s18()
    print(f"  [{tag}] 载入 {universe} ...", flush=True)
    data = m._load_training_data(cfg, universe, recent, pit_names, start, end)
    print(f"  [{tag}] {len(data)} 只，开始推理 ...", flush=True)
    pred = load_ensemble(cfg.resolve(f"results/model_v2_{tag}"))
    sig = pred.make_signals_all(data)
    prob = pd.DataFrame({s: t["prob_up"] for s, t in sig.items() if len(t)}).sort_index()
    prob.index = pd.to_datetime(prob.index)
    close = pd.DataFrame({s: pd.Series(df["close"].to_numpy(),
                                       index=pd.to_datetime(df["date"]))
                          for s, df in data.items()}).sort_index()
    feat = cfg["features"]
    horizon, window = int(feat["horizon"]), int(feat["window"])
    n_all = len(prob.index)
    ics_all = _daily_ic(prob, close, horizon)
    vd = _val_dates(close.index, window, horizon)
    ics = ics_all.reindex(vd).dropna()
    mean_all = float(ics_all.mean()) if len(ics_all) else 0.0
    print(f"  [{tag}] 全窗口 {n_all} 天（均值 IC={mean_all:+.4f}，含训练段）"
          f" → 验证段 {len(ics)} 天，均值 IC={ics.mean():+.4f}", flush=True)
    # 护栏：验证段 IC 高得离谱 => 多半是没裁干净，别拿它下结论
    if abs(ics.mean()) > 0.15:
        print(f"  [!] 验证段均值 IC={ics.mean():+.4f} 大得不合常理（真实水平约 0.03）——"
              "很可能是**训练段泄漏进了评价**，本结果不可用。", flush=True)
    del data, sig, prob, close
    return ics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="large500al", help="对照（有偏）的 tag")
    ap.add_argument("--b", default="pit500al", help="处理（无偏）的 tag")
    ap.add_argument("--universe-a", default="large")
    ap.add_argument("--universe-b", default="pit500")
    ap.add_argument("--pit-names", type=int, default=600)
    ap.add_argument("--start", default="2024-09-01")
    ap.add_argument("--end", default="2026-09-11")
    ap.add_argument("--recent", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config()
    print(f"[45] 配对检验  A={args.a}  B={args.b}  窗口 {args.start}~{args.end}")
    # 顺序载入、各自用完释放 —— 避免两个池子同时在内存里
    ic_a = _score(cfg, args.a, args.universe_a, 0, args.start, args.end, args.recent)
    ic_b = _score(cfg, args.b, args.universe_b, args.pit_names,
                  args.start, args.end, args.recent)

    j = pd.concat([ic_a.rename("a"), ic_b.rename("b")], axis=1).dropna()
    n = len(j)
    if n < 3:
        print(f"\n[45] 共同可评日只有 {n} 天，无法配对检验。")
        return 1

    d = j["b"] - j["a"]
    mean_d, sd_d = float(d.mean()), float(d.std(ddof=1))
    se = sd_d / np.sqrt(n) if sd_d > 0 else 0.0
    t = mean_d / se if se > 0 else 0.0
    corr = float(j["a"].corr(j["b"]))

    # 同口径的「非配对」se，用来看丢掉相关性到底高估了多少
    se_a, se_b = float(j["a"].std(ddof=1)), float(j["b"].std(ddof=1))
    se_un = np.sqrt(se_a ** 2 + se_b ** 2) / np.sqrt(n)

    print(f"\n[45] 共同可评日 n = {n}   （IC 序列相关 r = {corr:.3f}）")
    print(f"  A 平均 IC = {j['a'].mean():+.4f}   B 平均 IC = {j['b'].mean():+.4f}")
    print(f"  逐日差 mean = {mean_d:+.4f}   sd = {sd_d:.4f}")
    print(f"  **配对**   se = {se:.4f}   t = {t:+.2f}")
    print(f"  非配对     se = {se_un:.4f}   z = {mean_d / se_un:+.2f}   <- scripts/44 用的口径")
    print(f"  丢掉相关性把 se 高估了 {se_un / se:.2f} 倍" if se > 0 else "")

    verdict = ("**配对级证据成立**：无偏组显著优于有偏组（|t| > 2）" if abs(t) > 2 else
               "方向一致（B 更好），但**配对后仍未达显著**（|t| < 2）")
    print(f"\n  => {verdict}")
    if abs(t) <= 2:
        need = int(np.ceil((2.0 / abs(t)) ** 2 * n)) if t else 0
        print(f"     要让 |t| 到 2，同样效应量下需要约 {need} 个共同可评日（现在 {n}）。")
        print("     但那要求更长的训练窗口，内存放不下 —— 见 docs/2026-09-22-pool-bias-retrain.md。")
    print("\n[45] 注：两组横截面不同（599 vs 600 只、名单不同），这是实验本身的性质；")
    print("     配对只保证「同一天」可比。PIT 与真实中证500 重合度 76.6%，只用于相对比较。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
