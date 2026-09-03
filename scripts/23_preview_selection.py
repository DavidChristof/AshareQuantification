"""第 23 步预览：在 600 只大池 + 实证因子（低波动/60日反转）下跑一次「今日选股」。

用法：python scripts/23_preview_selection.py [--n 12] [--universe large]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.data.selector import select_daily          # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=12)
parser.add_argument("--universe", default="large", choices=["large", "hs300"])
args = parser.parse_args()

t0 = time.time()
rows = select_daily(n=args.n, universe=args.universe)
print(f"\n候选宇宙={args.universe} · 耗时 {time.time() - t0:.1f}s · 选出 {len(rows)} 只\n")
print(f"{'#':<3}{'代码':<8}{'名称':<10}{'综合':>7}{'基本面':>7}{'技术':>7}"
      f"{'动量20':>8}{'反转60':>8}{'波动':>7}{'PE':>6}{'ROE':>6}  池")
for i, r in enumerate(rows, 1):
    tag = "✓" if r["in_universe"] else "-"
    print(f"{i:<3}{r['code']:<8}{r['name'][:9]:<10}{r['total_score']:>7.1f}"
          f"{r['fund_score']:>7.1f}{(r['tech_score'] or 0):>7.1f}"
          f"{(r['mom20'] or 0):>+8.2f}{(r['rev60'] or 0):>+8.2f}"
          f"{(r['vol20'] or 0):>7.3f}{r['pe']:>6.1f}{r['roe']:>6.1f}  {tag}")
