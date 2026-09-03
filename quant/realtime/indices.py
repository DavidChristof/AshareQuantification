"""大盘指数实时行情（腾讯源）：作为组合决策的市场参考。

使用腾讯行情接口的指数通道（s_ 前缀），一次请求批量返回：
    https://qt.gtimg.cn/q=s_sh000001,s_sz399001,...

返回字段（v_s_ 前缀，~ 分隔，GBK 编码）：
    1 名称 · 2 当前点位 · 3 涨跌额 · 4 涨跌幅% · 5 今开 · 6 昨收 · 7 最高 · 8 最低
    9-... 略 · 31 成交量(手) · 32 成交额(万元)
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# 主要大盘指数（代码 → 名称）
INDEX_CODES = {
    "sh000001": "上证指数",
    "sz399001": "深证成指",
    "sz399006": "创业板指",
    "sh000300": "沪深300",
}

_TENCENT_URL = "https://qt.gtimg.cn/q={codes}"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_TIMEOUT = 6
_CACHE_TTL = 10          # 缓存秒数（实时行情）
_DAILY_TTL = 1800        # 日 K 线缓存秒数（30 分钟）

# 日 K 线内存缓存: {code: (timestamp, DataFrame)}
_DAILY_CACHE: dict = {}


def fetch_index_daily(code: str) -> pd.DataFrame:
    """获取指数日 K 线（新浪源，带 30 分钟缓存）。

    返回 DataFrame(date/open/high/low/close/volume)，date 为 datetime64。
    失败时回退到上次缓存；无缓存则抛异常由调用方处理。
    """
    import akshare as ak

    now = time.time()
    if code in _DAILY_CACHE and now - _DAILY_CACHE[code][0] < _DAILY_TTL:
        return _DAILY_CACHE[code][1]
    df = ak.stock_zh_index_daily(symbol=code)
    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    _DAILY_CACHE[code] = (now, df)
    return df


class IndexQuoter:
    """指数行情抓取器（带短缓存，避免每次请求都打网络）。"""

    def __init__(self, codes: dict[str, str] | None = None):
        self.codes = codes or INDEX_CODES
        self._cache: list[dict] = []
        self._cached_at = 0.0

    def _parse(self, raw: str, code: str) -> dict | None:
        """解析单条腾讯指数（s_ 通道）：
        v_s_sh000001="1~上证指数~000001~3979.89~-6.41~-0.16~量~额~...~ZS";
        索引：1名称 · 2代码 · 3当前点位 · 4涨跌额 · 5涨跌幅% · 6量(手) · 7额(万)
        """
        start = raw.find('="')
        if start < 0:
            return None
        end = raw.rfind('"')
        body = raw[start + 2:end] if end > start else ""
        f = body.split("~")
        if len(f) < 6 or not f[1]:
            return None
        point = float(f[3]); change = float(f[4])
        return {
            "code": code,
            "name": self.codes.get(code, f[1]),
            "point": point,
            "prev_close": round(point - change, 4),
            "change": change,
            "change_pct": float(f[5]),
            "volume": float(f[6]) if len(f) > 6 and f[6] else 0.0,   # 手
            "amount": float(f[7]) if len(f) > 7 and f[7] else 0.0,   # 万元
            "source": "tencent",
        }

    def fetch(self, force: bool = False) -> list[dict]:
        """返回指数列表；TTL 内走缓存。失败返回空列表。"""
        now = time.time()
        if not force and self._cache and now - self._cached_at < _CACHE_TTL:
            return self._cache
        try:
            codes = ",".join(f"s_{c}" for c in self.codes)
            resp = requests.get(
                _TENCENT_URL.format(codes=codes),
                headers={"User-Agent": _UA},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            resp.encoding = "gbk"
            result = []
            lines = resp.text.strip().splitlines()
            for raw, code in zip(lines, self.codes):
                q = self._parse(raw, code)
                if q:
                    result.append(q)
            if result:
                self._cache = result
                self._cached_at = now
            return result
        except Exception as exc:  # noqa: BLE001
            logger.warning("指数行情获取失败: %s", exc)
            return self._cache if self._cache else []


# 全局单例（供 API 复用）
QUOTER = IndexQuoter()
