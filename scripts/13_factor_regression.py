"""第 13 步：多因子回归——检验多因子对收益的联合解释力与边际显著性。

对应教材「多因子回归 / Fama-MacBeth」：比单因子 IC 更进一步，
把全部因子放进同一回归，看控制其他因子后各因子是否仍显著。

用法：
    python scripts/13_factor_regression.py                # 预测期 5/20 日
    python scripts/13_factor_regression.py --horizons 5 20 60
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import load_config                        # noqa: E402
from quant.data.loader import load_all                      # noqa: E402
from quant.factors.regression import (                      # noqa: E402
    _format_report, multi_factor_report,
)

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

    report = multi_factor_report(data, horizons=tuple(args.horizons))
    text = _format_report(report)

    print("\n==================== 多因子回归报告 ====================")
    print(text)

    out = cfg.resolve("results")
    out.mkdir(parents=True, exist_ok=True)
    # 保存 CSV：每个预测期一张 FM 表
    for h, fm in report["fama_macbeth"].items():
        if not fm.empty:
            path = out / f"factor_fm_h{h}.csv"
            fm.to_csv(path, index=False, encoding="utf-8-sig")
            logger.info("Fama-MacBeth(h=%d) 已保存到 %s", h, path)
    for h, p in report["pooled"].items():
        if p:
            path = out / f"factor_pooled_h{h}.csv"
            p["factor"].to_csv(path, index=False, encoding="utf-8-sig")
            logger.info("Pooling(h=%d) 已保存到 %s", h, path)


if __name__ == "__main__":
    main()
