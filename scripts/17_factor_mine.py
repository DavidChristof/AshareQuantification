"""第 17 步：Alpha101 因子 + 自动因子挖掘——寻找有效因子。

受 FinHack 启发：内置世界经典 Alpha101 因子集 + 自动因子挖掘。
我们实现候选因子池 = Alpha101 子集（~20 个）+ 自动挖掘因子（~30 个），
全部接入现有的因子检验流水线（Rank IC / ICIR / 多因子回归）：

    1. 对每个候选因子 × 预测期算 Rank IC / ICIR（复用 analysis）
    2. 多因子回归（Fama-MacBeth + Pooling，复用 regression，自动剔除共线）
    3. 输出报告：哪些因子 |IC|>0.03 有效、回归显著性

用法：
    python scripts/17_factor_mine.py                  # 预测期 5/20
    python scripts/17_factor_mine.py --horizons 5 20 60 --top 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd                                    # noqa: E402

from quant.config import load_config                   # noqa: E402
from quant.data.loader import load_all                 # noqa: E402
from quant.factors.alpha101 import build_all_candidate_panels  # noqa: E402
from quant.factors.analysis import (_prepare_panels,   # noqa: E402
                                    forward_returns, judge_factor,
                                    rank_ic_series, summarize_ic)
from quant.factors.regression import (                 # noqa: E402
    _format_report, fama_macbeth, pooled_ols,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

# 有效判定阈值
EFFECTIVE_IC = 0.03


def ic_report(data: dict, factor_panels: dict, horizons: tuple[int, ...]
              ) -> pd.DataFrame:
    """对候选因子池每个因子 × 预测期算 Rank IC / ICIR 报告。"""
    close_panel, _ = _prepare_panels(data)
    rows = []
    for name, panel in factor_panels.items():
        for h in horizons:
            rets = forward_returns(close_panel, h)
            ics = rank_ic_series(panel, rets)
            rep = summarize_ic(ics)
            if rep is None:
                continue
            rep["factor"] = name
            rep["horizon"] = h
            rep["judge"] = judge_factor(rep)
            rows.append(rep)
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["factor", "horizon", "mean_ic", "icir", "ic_positive",
                 "abs_ic", "n_days", "judge"]]
        df = df.sort_values(["horizon", "abs_ic"], ascending=[True, False])
    return df.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--top", type=int, default=20,
                        help="多因子回归纳入的因子数（按 |IC| 取前 N）")
    args = parser.parse_args()

    cfg = load_config()
    data = load_all(cfg)
    logger.info("已加载 %d 只股票", len(data))

    # 1. 候选因子池 = Alpha101 子集 + 自动挖掘
    panels = build_all_candidate_panels(data)
    n_alpha = len([k for k in panels if k.startswith("alpha")])
    logger.info("候选因子池: %d 个（Alpha101 子集 %d + 挖掘 %d）",
                len(panels), n_alpha, len(panels) - n_alpha)

    # 2. Rank IC / ICIR 检验
    report = ic_report(data, panels, tuple(args.horizons))
    print("\n==================== Alpha101 + 因子挖掘 IC 检验 ====================")
    print("判定：|IC|>0.05 有效 · >0.10 优秀 · ICIR>0.5 高质量")
    print(report.to_string(index=False))

    # 3. 高亮有效因子
    eff = report[report["abs_ic"] >= EFFECTIVE_IC]
    if len(eff):
        print("\n>>> 候选有效因子（|IC| >= %.2f）：" % EFFECTIVE_IC)
        print(eff[["factor", "horizon", "mean_ic", "icir", "judge"]].to_string(index=False))
    else:
        print(f"\n（候选因子池在所选预测期下均无明显 IC，可扩展更多挖掘组合）")

    # 4. 多因子回归（|IC| 前 top 个因子，Fama-MacBeth + Pooling）
    if len(report):
        best = (report.sort_values("abs_ic", ascending=False)
                ["factor"].drop_duplicates().head(args.top).tolist())
        logger.info("多因子回归纳入 top%d 因子: %s", len(best), best)
        fm = fama_macbeth(data, horizon=args.horizons[0], factor_names=best,
                          factor_panels=panels)
        pool = pooled_ols(data, horizon=args.horizons[0], factor_names=best,
                          factor_panels=panels)
        print("\n============ 多因子回归（预测期 %d 日 · top%d 因子） ============" % (
            args.horizons[0], len(best)))
        print(_format_report({"fama_macbeth": {args.horizons[0]: fm},
                              "pooled": {args.horizons[0]: pool}}))

    # 5. 保存
    out = cfg.resolve("results")
    out.mkdir(parents=True, exist_ok=True)
    report.to_csv(out / "factor_mine_report.csv", index=False, encoding="utf-8-sig")
    logger.info("报告已保存到 %s", out / "factor_mine_report.csv")


if __name__ == "__main__":
    main()
