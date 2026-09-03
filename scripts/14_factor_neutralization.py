"""第 14 步：因子中性化——剔除风格暴露（规模/波动率）后重算 IC。

对应教材「因子中性化」：一个因子显著可能只是暴露在风格上，
把因子对风格变量做横截面回归取残差（纯因子），看 IC 是否保留。

用法：
    python scripts/14_factor_neutralization.py            # 预测期 5/20 日
    python scripts/14_factor_neutralization.py --horizons 5 20 60
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import load_config                        # noqa: E402
from quant.data.loader import load_all                      # noqa: E402
from quant.factors.neutralize import _format_report, neutralize_report  # noqa: E402

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

    report = neutralize_report(data, horizons=tuple(args.horizons))

    print("\n==================== 因子中性化报告 ====================")
    print(_format_report(report))

    out = cfg.resolve("results")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "factor_neutralization.csv"
    report.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("报告已保存到 %s", path)


if __name__ == "__main__":
    main()
