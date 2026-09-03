"""实时行情抓取：新浪 / 腾讯双源，自动回退。

统一输出（每只股票的 quote dict）：
    symbol/name/price/prev_close/open/high/low/volume/amount
    change/change_pct/bid/ask(五档)/time/source
"""
from __future__ import annotations

import logging
import re
from typing import Iterable

import requests

logger = logging.getLogger(__name__)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_TIMEOUT = 6

_SINA_URL = "https://hq.sinajs.cn/list={codes}"
_TENCENT_URL = "https://qt.gtimg.cn/q={codes}"


def _prefix(symbol: str) -> str:
    """6 位代码 → 带交易所前缀（60/68→sh，00/30→sz，其他→bj）。"""
    if symbol.startswith(("sh", "sz", "bj")):
        return symbol
    if symbol.startswith(("60", "68")):
        return f"sh{symbol}"
    if symbol.startswith(("00", "30")):
        return f"sz{symbol}"
    return f"bj{symbol}"


def _f(val: str, default: float = 0.0) -> float:
    """安全转 float，空串/异常返回 default。"""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ---------- 新浪源 ----------
def _parse_sina_line(raw: str, symbol: str):
    """解析新浪单条：var hq_str_sh600519="..."; """
    m = re.search(r'="(.*)"', raw)
    if not m:
        return None
    f = m.group(1).split(",")
    if len(f) < 32 or not f[0]:
        return None
    price = _f(f[3]); prev = _f(f[2])
    bid = [( _f(f[11]), _f(f[10])), (_f(f[13]), _f(f[12])), (_f(f[15]), _f(f[14])),
           (_f(f[17]), _f(f[16])), (_f(f[19]), _f(f[18]))]
    ask = [( _f(f[21]), _f(f[20])), (_f(f[23]), _f(f[22])), (_f(f[25]), _f(f[24])),
           (_f(f[27]), _f(f[26])), (_f(f[29]), _f(f[28]))]
    return {
        "symbol": symbol, "name": f[0].replace(" ", ""),
        "price": price, "prev_close": prev, "open": _f(f[1]),
        "high": _f(f[4]), "low": _f(f[5]),
        "volume": _f(f[8]), "amount": _f(f[9]),      # 股 / 元
        "change": price - prev if prev else 0.0,
        "change_pct": (price / prev - 1) if prev else 0.0,
        "bid": bid, "ask": ask,
        "time": f"{f[30]} {f[31]}", "source": "sina",
    }


def fetch_sina(symbols: Iterable[str]) -> dict[str, dict]:
    """新浪批量快照。返回 {symbol: quote}。"""
    sym_list = list(symbols)
    if not sym_list:
        return {}
    codes = ",".join(_prefix(s) for s in sym_list)
    resp = requests.get(
        _SINA_URL.format(codes=codes),
        headers={"Referer": "https://finance.sina.com.cn", "User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    resp.encoding = "gbk"
    result = {}
    for s, line in zip(sym_list, resp.text.strip().splitlines()):
        q = _parse_sina_line(line, s)
        if q:
            result[s] = q
    return result


# ---------- 腾讯源 ----------
def _parse_tencent_line(raw: str, symbol: str):
    """解析腾讯单条：v_sh600519="1~name~code~..."; """
    m = re.search(r'="(.*)"', raw)
    if not m:
        return None
    f = m.group(1).split("~")
    if len(f) < 35 or not f[1]:
        return None
    price = _f(f[3]); prev = _f(f[4])
    # 腾讯五档量单位为「手」，统一转成「股」与新浪保持一致
    bid = [( _f(f[9]), _f(f[10]) * 100), (_f(f[11]), _f(f[12]) * 100),
           (_f(f[13]), _f(f[14]) * 100), (_f(f[15]), _f(f[16]) * 100),
           (_f(f[17]), _f(f[18]) * 100)]
    ask = [( _f(f[19]), _f(f[20]) * 100), (_f(f[21]), _f(f[22]) * 100),
           (_f(f[23]), _f(f[24]) * 100), (_f(f[25]), _f(f[26]) * 100),
           (_f(f[27]), _f(f[28]) * 100)]
    return {
        "symbol": symbol, "name": f[1].replace(" ", ""),
        "price": price, "prev_close": prev, "open": _f(f[5]),
        "high": _f(f[33]), "low": _f(f[34]),
        "volume": _f(f[6]) * 100, "amount": _f(f[37]) * 10000,  # 手→股 / 万元→元
        "change": _f(f[31]), "change_pct": _f(f[32]) / 100,      # % → 小数
        "bid": bid, "ask": ask,
        "time": f[30], "source": "tencent",
    }


def fetch_tencent(symbols: Iterable[str]) -> dict[str, dict]:
    """腾讯批量快照。返回 {symbol: quote}。"""
    sym_list = list(symbols)
    if not sym_list:
        return {}
    codes = ",".join(_prefix(s) for s in sym_list)
    resp = requests.get(
        _TENCENT_URL.format(codes=codes),
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    resp.encoding = "gbk"
    result = {}
    for s, line in zip(sym_list, resp.text.strip().splitlines()):
        q = _parse_tencent_line(line, s)
        if q:
            result[s] = q
    return result


# ---------- 双源合并（新浪优先，缺失回退腾讯） ----------
def fetch_quotes(symbols: Iterable[str]) -> dict[str, dict]:
    """获取实时快照：新浪批量优先，缺失的用腾讯补齐。"""
    sym_list = list(symbols)
    quotes: dict[str, dict] = {}

    try:
        quotes.update(fetch_sina(sym_list))
    except Exception as exc:  # noqa: BLE001
        logger.warning("新浪实时接口失败: %s", exc)

    missing = [s for s in sym_list if s not in quotes]
    if missing:
        try:
            quotes.update(fetch_tencent(missing))
        except Exception as exc:  # noqa: BLE001
            logger.warning("腾讯实时接口失败: %s", exc)

    logger.debug("实时快照获取 %d/%d 只", len(quotes), len(sym_list))
    return quotes
