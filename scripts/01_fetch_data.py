"""第 1 步：下载 A 股日线数据到 SQLite。

用法：
    python scripts/01_fetch_data.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# 允许从项目根目录 import quant
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import load_config                      # noqa: E402
from quant.data.fetcher import fetch_universe             # noqa: E402
from quant.data.storage import MarketDB                   # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    cfg = load_config()
    data_cfg = cfg["data"]

    logger.info("股票池: %s", data_cfg["universe"])
    logger.info("时间范围: %s ~ %s", data_cfg["start_date"], data_cfg["end_date"])

    # 1. 从 akshare 拉取数据
    df = fetch_universe(
        data_cfg["universe"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
    )
    logger.info("共获取 %d 行", len(df))

    # 2. 存入 SQLite
    db = MarketDB(cfg.resolve(data_cfg["db_path"]))
    db.save_bars(df)

    # 3. 展示入库情况
    logger.info("数据库统计: %s", db.stats())

    # 4. 可选：导出 CSV 备份
    csv_dir = cfg.resolve(data_cfg["csv_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)
    for symbol, group in df.groupby("symbol"):
        group.to_csv(csv_dir / f"{symbol}.csv", index=False, encoding="utf-8-sig")
    logger.info("CSV 备份已保存到 %s", csv_dir)


if __name__ == "__main__":
    main()
