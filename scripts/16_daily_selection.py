"""第 16 步：每日选股——寻找大盘（沪深300）优质股。

在流动性 + 基本面（PE/ROE）基础上叠加技术面（动量/趋势/波动），
每天收盘后跑一次，找出「基本面好且当前技术面走强」的优质候选股。
**不修改训练股票池**（避免频繁换池影响模型），只做推荐。

用法：
    python scripts/16_daily_selection.py            # 默认 top 12
    python scripts/16_daily_selection.py --n 20 --basic 100 --tech 50
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.config import load_config                        # noqa: E402
from quant.data.selector import select_daily                # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=12, help="展示优质股数量")
    parser.add_argument("--basic", type=int, default=80, help="抓基本面的候选数")
    parser.add_argument("--tech", type=int, default=40, help="算技术面的候选数")
    args = parser.parse_args()

    logger.info("每日选股开始（n=%d, basic=%d, tech=%d）...", args.n, args.basic, args.tech)
    rows = select_daily(n=args.n, basic_topk=args.basic, tech_topk=args.tech)

    today = datetime.now().strftime("%Y-%m-%d")

    # 先保存结果（JSON + CSV），避免打印异常导致结果丢失
    cfg = load_config()
    out = cfg.resolve("results")
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "date": today,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "candidates": rows,
    }
    json_path = out / "daily_selection.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    import pandas as pd
    csv_path = out / "daily_selection.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info("选股结果已保存到 %s / %s", json_path, csv_path)

    print("\n================ 每日选股（%s） ================" % today)
    print("综合分 = 基本面(PE/ROE)x0.6 + 技术面 x0.4 (动量25/趋势15/低波动30/反转30)")
    print(f"{'排名':<4}{'代码':<8}{'名称':<10}{'综合':<7}{'基本面':<7}"
          f"{'技术':<7}{'PE':<8}{'ROE':<6}{'动量20':<8}{'反转60':<8}{'池':<3}")
    for i, r in enumerate(rows, 1):
        mom = f"{r['mom20'] * 100:+.1f}%" if r["mom20"] is not None else "-"
        rev = f"{r['rev60'] * 100:+.1f}%" if r["rev60"] is not None else "-"
        mark = "Y" if r["in_universe"] else "-"
        print(f"{i:<4}{r['code']:<8}{r['name']:<10}{r['total_score']:<7}"
              f"{r['fund_score']:<7}{str(r['tech_score']):<7}{r['pe']:<8}"
              f"{r['roe']:<6}{mom:<8}{rev:<8}{mark:<3}")


if __name__ == "__main__":
    main()
