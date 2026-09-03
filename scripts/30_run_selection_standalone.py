"""第 30 步：独立子进程执行每日选股（隔离 py_mini_racer 崩溃）。

背景：在 API 服务内直接跑大池选股时，抓基本面会调用 akshare 的百度估值接口
（内部用 py_mini_racer 执行 JS 解密）——该库偶发 **native 进程级崩溃**（无法
try/except 捕获），会连带把整个 uvicorn 服务杀死。因此把选股放进独立子进程：
崩了只杀本进程，API 侧用 subprocess 捕获退出码即可，服务永不被拖垮。

用法：
    python scripts/30_run_selection_standalone.py            # 用 config selection.n
    python scripts/30_run_selection_standalone.py --n 12

成功：写 results/daily_selection.json（含 regime 门控的市场状态与权重）+ 打印 OK。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    from quant.config import load_config  # noqa: PLC0415
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None,
                        help="选股数量（默认 config selection.n）")
    args = parser.parse_args()
    n = args.n or int((cfg.get("selection") or {}).get("n", 12))

    from quant.data.selector import fetch_market_regime, select_daily  # noqa: PLC0415

    rows = select_daily(n=n)
    now = datetime.now()
    regime = None
    try:
        regime = fetch_market_regime()
    except Exception as exc:  # noqa: BLE001
        logger.warning("市场状态判定失败: %s", exc)

    result = {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "candidates": rows,
        "regime": regime,
    }
    path = cfg.resolve("results") / "daily_selection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("选股完成 %d 只，regime=%s，已写入 %s",
                len(rows), (regime or {}).get("regime"), path)
    print("OK")


if __name__ == "__main__":
    main()
