"""第 12 步：因子有效性检验报告——计算每个因子的 Rank IC / 平均 IC / ICIR。

对应教材「因子检验模块」：评估各因子对未来收益的预测能力与稳定性。

用法：
    python scripts/12_factor_ic.py                    # 预测期 5/20 日
    python scripts/12_factor_ic.py --horizons 5 20 60
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import load_config                        # noqa: E402
from quant.data.loader import load_all                      # noqa: E402
from quant.factors.analysis import factor_ic_report         # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 20],
                        help="未来收益预测期（日）")
    args = parser.parse_args()

    cfg = load_config()
    data = load_all(cfg)
    logger.info("已加载 %d 只股票", len(data))

    report = factor_ic_report(data, horizons=tuple(args.horizons))

    print("\n==================== 因子 IC / ICIR 检验报告 ====================")
    print("判定标准：|IC|>0.05 有效 · >0.10 优秀 · ICIR>0.5 高质量")
    print(report.to_string(index=False))

    out = cfg.resolve("results") / "factor_ic_report.csv"
    report.to_csv(out, index=False, encoding="utf-8-sig")
    logger.info("报告已保存到 %s", out)

    # 高亮有效因子
    good = report[report["judge"] != "弱/无效"]
    if len(good):
        print("\n有效因子汇总：")
        print(good[["factor", "horizon", "mean_ic", "icir", "judge"]].to_string(index=False))
    else:
        print("\n（当前因子集在所选预测期下均无明显预测力，可扩展新因子）")


if __name__ == "__main__":
    main()
