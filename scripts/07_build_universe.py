"""第 7 步：自动生成股票池并写入 config.yaml。

从沪深300成分股随机抽取 N 只（固定种子可复现），更新配置中的 universe。

用法：
    python scripts/07_build_universe.py --n 40 --seed 42
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml                                                      # noqa: E402

from quant.data.selector import select_universe                  # noqa: E402
from quant.data.universe import pick_universe                    # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def main():
    parser = argparse.ArgumentParser(description="生成股票池并写入 config.yaml")
    parser.add_argument("--mode", choices=["quality", "random"], default="quality",
                        help="quality=流动性+PE/ROE基本面选优质股(慢但优质); random=随机抽样(快)")
    parser.add_argument("--n", type=int, default=40, help="股票池数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 1. 生成股票池
    if args.mode == "quality":
        logger.info("选股模式: quality（排除ST + 流动性过滤 + PE/ROE 基本面打分）")
        universe = select_universe(n=args.n, seed=args.seed)
    else:
        logger.info("选股模式: random（沪深300 随机抽样，可复现）")
        universe = pick_universe(n=args.n, seed=args.seed)
    codes = [s["code"] for s in universe]
    logger.info("生成股票池 %d 只: %s", len(codes), codes)

    # 2. 备份并更新 config.yaml
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    backup = CONFIG_PATH.with_suffix(".yaml.bak")
    backup.write_text(open(CONFIG_PATH, encoding="utf-8").read(), encoding="utf-8")
    logger.info("已备份原配置到 %s", backup)

    cfg["data"]["universe"] = codes
    cfg["data"]["universe_names"] = {s["code"]: s["name"] for s in universe}
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    logger.info("config.yaml 股票池已更新为 %d 只（含名称映射）", len(codes))
    print("\n请接着运行：")
    print("  python scripts/01_fetch_data.py    # 下载新股票池日线")
    print("  python scripts/03_train.py         # 用更大样本重训模型")


if __name__ == "__main__":
    main()
