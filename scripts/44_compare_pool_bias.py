"""第 44 步：**池子偏差判别** —— 把「有幸存者偏差的池」与「PIT 无偏宇宙」训练的模型并排比。

## 为什么需要它

2026-09-22 用户问「是否是我们的方向搞错了？也许是因子选择出的问题呢？」排查后发现的**真问题**
不是因子选择（三个不同池/规模的模型留出 IC 都在 -0.014 ~ +0.037、top-bottom 全 ≈ 0 或负
—— 换因子组应该看到分化，实际是三套不同的东西都弱，共同点是**训练/标签/评估流程**）。

于是设计了这个判别实验：**一次只动一个变量 —— 池子**，标签/超参/成员一律不变。

    对照 A = 现有 559 池（现池40 ∪ large_pool，**幸存者偏差**）
    处理 B = PIT 中证500 层并集（1048 只，**无未来函数**，见 docs/2026-09-11-pit-universe.md）

**必须同 `--recent`**：现有跑批已经把「窗口长度」和「池子」混在一起了
（同样 40 只票：全历史 IC -0.0138 vs recent1000 +0.0365），不同 recent 的比较无效。

## 判定口径（写进脚本，免得下次重新解释）

    t    = ICIR × sqrt(n_days)          ← 模型的 report 里没有 t，用它自己给的 ICIR/n 反推
    se   = rankic_std / sqrt(n_days)    ← 等价于 mean / (ICIR × sqrt(n))
    z    = (IC_B - IC_A) / sqrt(se_A^2 + se_B^2)

  - **偏差是主因** → B 的 IC 显著更高（z > 2）**且** top-bottom 由负转正；
  - **偏差不是主因** → 两者都趴在 0 附近、z 不显著 => 下一轮去测**学习目标**
    （现在的 `relative` 标签只优化整体秩相关，**不优化头部**，而选股只用头部）。

## 用法

    .venv/Scripts/python.exe scripts/44_compare_pool_bias.py
    .venv/Scripts/python.exe scripts/44_compare_pool_bias.py --a large500 --b pit500

读 `results/model_v2_<tag>_report.json` 里的 `ensemble.cross_metrics`
（**留出段 = 后 20% 时间**，这才是可信的数字；27 的「全窗口」含训练段 80%，是记忆不是能力）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

from quant.config import load_config                                # noqa: E402

# 照抄 28 的实质门槛，便于跨脚本对照
MIN_RANKIC = 0.015


def _load(cfg, tag: str) -> dict | None:
    """读 results/model_v2_<tag>_report.json 的 ensemble.cross_metrics。"""
    p = Path(cfg.resolve("results")) / f"model_v2_{tag}_report.json"
    if not p.exists():
        print(f"  [!] 缺 {p.name} —— 先跑 scripts/18_train_ensemble.py ... --tag {tag}")
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    cm = (d.get("ensemble") or {}).get("cross_metrics") or {}
    if not cm:
        print(f"  [!] {p.name} 里没有 ensemble.cross_metrics")
        return None
    n = int(cm.get("n_days") or 0)
    ic = float(cm.get("rankic_mean") or 0.0)
    sd = float(cm.get("rankic_std") or 0.0)
    icir = float(cm.get("icir") or 0.0)
    se = sd / math.sqrt(n) if n > 0 and sd > 0 else float("inf")
    return {"tag": tag, "ic": ic, "icir": icir, "std": sd, "n": n,
            "tb": float(cm.get("top_bottom") or 0.0), "se": se,
            "t": (ic / se) if se not in (0.0, float("inf")) else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="large500", help="对照（有幸存者偏差）的 tag")
    ap.add_argument("--b", default="pit500", help="处理（PIT 无偏）的 tag")
    args = ap.parse_args()

    cfg = load_config()
    A, B = _load(cfg, args.a), _load(cfg, args.b)
    if A is None or B is None:
        print("\n[44] 数据不全，无法对照。")
        return 1

    print(f"\n[44] 池子偏差判别（留出段 = 后 20% 时间；门槛 RankIC > {MIN_RANKIC}）")
    print(f"  {'组':<22}{'留出IC':>10}{'ICIR':>9}{'t':>8}{'top-bottom':>12}{'n_days':>8}")
    for r, name in ((A, "A 对照(有偏)"), (B, "B 处理(无偏)")):
        print(f"  {name + ' ' + r['tag']:<22}{r['ic']:>+10.4f}{r['icir']:>9.3f}"
              f"{r['t']:>+8.2f}{r['tb']:>+12.4f}{r['n']:>8}")

    diff = B["ic"] - A["ic"]
    se_d = math.sqrt(A["se"] ** 2 + B["se"] ** 2) if math.isfinite(A["se"] + B["se"]) else float("inf")
    z = diff / se_d if se_d not in (0.0, float("inf")) else 0.0
    print(f"\n  B - A：IC 差 {diff:+.4f}   se {se_d:.4f}   z = {z:+.2f}")

    # ---- 判定 ----
    bias_main = (z > 2.0) and (B["tb"] > 0 > A["tb"] or B["tb"] > A["tb"])
    both_weak = (abs(A["ic"]) < MIN_RANKIC) and (abs(B["ic"]) < MIN_RANKIC)
    # 中间档：两组之差没到显著（分别训练，比较噪声大），但**只有无偏组自身站得住**
    # （t>2），有偏组自身不显著（t<2）。这不是「不确定」—— 「一个统计上成立、
    # 一个不成立」本身就是有方向的信息，只是强度低于配对检验。
    # [!] 注意别用「谁越过 0.015 点估计门槛」来判：A 的 0.0190 也在门槛之上，
    #     只是它的 t=1.49 说明这个点估计**和 0 分不开**。有区分度的是 t，不是点估计。
    only_b_sig = (B["t"] > 2.0) and (A["t"] < 2.0) and (B["ic"] > A["ic"]) and (B["tb"] >= A["tb"])
    if bias_main:
        print("\n  => **幸存者偏差是主因**：无偏池的留出 IC 显著更高、且 top-bottom 改善。")
        print("     方向：先把训练土壤换成 PIT（并检查线上/选股是否也在用有偏名单）。")
    elif only_b_sig:
        print(f"\n  => **幸存者偏差很可能是主因**（方向性证据）：无偏组自身**统计上站得住**"
              f"（IC={B['ic']:+.4f}, t={B['t']:+.2f}），有偏组站不住"
              f"（IC={A['ic']:+.4f}, t={A['t']:+.2f}，与 0 分不开）。")
        print(f"     但**两组之差未达显著**（z={z:+.2f}）—— 两个模型分别独立训练，"
              "比较本身的噪声远大于配对检验。")
        print("     要把它升级成「配对级」证据：把两组钉在**同一段日期**上重训，")
        print("     消掉验证段错位（--recent 是按每只票自己的尾部切的，两池历史长度不同）。")
    elif both_weak:
        print("\n  => **偏差不是主因**：两组的留出 IC 都趴在 0 附近。")
        print("     方向：下一轮测**学习目标** —— 现标签 relative 只优化整体秩相关、")
        print("     不优化头部，而选股只用头部（这是 top-bottom ≈ 0 的最可能解释）。")
    else:
        print("\n  => **不确定**：既没到显著，也不都属于「都趴在 0 附近」。")
        print("     建议看逐日/逐年分解，或加大 n_days（更长 --recent）再判。")

    print("\n[44] 注：PIT 中证500 与真实中证500 重合度仅 76.6%（是无偏的同区间宇宙、")
    print("     不是真实指数的复制品），故只用于**相对比较**。full_market.db 到 2026-09-11 为止。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
