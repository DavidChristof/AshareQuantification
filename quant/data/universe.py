"""股票池生成工具：从指数成分股构建自定义股票池。

阶段三：扩展规模。用「沪深300 成分股」作为候选池，按需抽取 N 只。
- 沪深300 无 ST 股、流动性好，适合作为股票池来源
- 固定随机种子保证可复现（同一 seed 每次生成相同股票池）
"""
from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)


def fetch_hs300() -> list[dict]:
    """拉取沪深300成分股（中证指数官网）。

    Returns:
        [{code, name}]：code 为 6 位代码，name 为中文名称。
    """
    import akshare as ak

    df = ak.index_stock_cons_csindex(symbol="000300")
    if df is None or df.empty:
        raise RuntimeError("沪深300成分股获取失败")

    # 列名是中文，需精确定位「成分证券代码/名称」（避免误匹配「指数代码/名称」）
    code_col = next((c for c in df.columns if "成分" in c and "代码" in c), None)
    name_col = next((c for c in df.columns if "成分" in c and "名称" in c), None)
    if not code_col or not name_col:
        raise RuntimeError(f"成分股列名异常: {list(df.columns)}")

    stocks = []
    for _, row in df.iterrows():
        code = str(row[code_col]).zfill(6)
        name = str(row[name_col])
        if code and code.isdigit():
            stocks.append({"code": code, "name": name})
    logger.info("沪深300成分股: %d 只", len(stocks))
    return stocks


def pick_universe(n: int = 40, seed: int = 42,
                  exclude_prefix: tuple[str, ...] = ("bj",)) -> list[dict]:
    """从沪深300随机抽取 n 只（固定种子可复现）。

    Args:
        n: 目标股票数。
        seed: 随机种子。
        exclude_prefix: 排除的交易所前缀（默认排除北交所）。
    """
    stocks = fetch_hs300()
    if not stocks:
        raise RuntimeError("成分股为空")

    # 排除指定前缀（代码转前缀判断）
    def _prefix(code: str) -> str:
        if code.startswith(("60", "68")):
            return "sh"
        if code.startswith(("00", "30")):
            return "sz"
        return "bj"

    candidates = [s for s in stocks if _prefix(s["code"]) not in exclude_prefix]
    if len(candidates) < n:
        logger.warning("候选 %d 只 < 目标 %d，取全部", len(candidates), n)
        n = len(candidates)

    rng = random.Random(seed)
    sample = rng.sample(candidates, n)
    # 按代码排序，便于阅读和对比
    sample.sort(key=lambda s: s["code"])
    return sample
