"""大盘指数行情（腾讯源）：实时报价 + 日 K 线。作为组合决策的市场参考。

- **实时报价**用腾讯行情接口的指数通道（s_ 前缀），一次请求批量返回：
      https://qt.gtimg.cn/q=s_sh000001,s_sz399001,...
  返回字段（v_s_ 前缀，~ 分隔，GBK 编码）：
      1 名称 · 2 当前点位 · 3 涨跌额 · 4 涨跌幅% · 5 今开 · 6 昨收 · 7 最高 · 8 最低
      9-... 略 · 31 成交量(手) · 32 成交额(万元)
- **日 K 线**用腾讯 K 线接口（见 `fetch_index_daily`）。

## 为什么日 K 线也从新浪换到了腾讯（2026-09-15）

新浪源走的是 `akshare.stock_zh_index_daily`，而它内部是
`py_mini_racer.MiniRacer()` **跑 JS 解密** —— 这是本机 V8 原生崩溃
（`[FATAL:partition_address_space.cc(243)] Check failed: !IsConfigurablePoolInitialized()`）
唯一的来源。近 6 天 15 次崩溃**全部**落在 `mini_racer.dll`（Windows 事件日志确证），
反复把 api 服务整个打死。而 `fetch_index_daily` 就跑在 **api 服务进程内**，
因此服务必崩。腾讯 K 线接口是**纯 requests、完全不碰 V8**，换过去等于从根上消灭这类崩溃。

[!] **注意：换源不是为了换数据** —— 已逐值核对 6 个指数 × 2000 根重叠 bar，
价格最大相对误差 1.2e-06（纯分位取整），成交量 ×100 后与新浪**完全相等**。
另有两个源间差异已在 `fetch_index_daily` 里逐条抹平（字段顺序、当日 bar、成交量单位）。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

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
_TENCENT_KLINE_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_TIMEOUT = 6
_CACHE_TTL = 10          # 缓存秒数（实时行情）
_DAILY_TTL = 1800        # 日 K 线缓存秒数（30 分钟）
_KLINE_COUNT = 2000      # 腾讯单次最多返回的 bar 数（实测 2500 起返回空）
_SESSION_CLOSE = (15, 0)  # A 股连续竞价收盘时间（用于判定当日 bar 是否已走完）

# 日 K 线内存缓存: {code: (timestamp, DataFrame)}
_DAILY_CACHE: dict = {}


def _intraday(now: datetime | None = None) -> bool:
    """现在是否「当日尚未收盘」——决定要不要剔除腾讯的当日实时 bar。

    只在今天恰是交易日、且在 15:00 之前时才起作用（其它情况当日 bar 本就不存在）。
    """
    n = now or datetime.now()
    return (n.hour, n.minute) < _SESSION_CLOSE


def _fetch_tencent_kline(code: str, count: int) -> pd.DataFrame:
    """拉腾讯日 K 线并规整成与旧新浪源**完全一致**的列与单位。

    [!] 两处源间差异必须在这里抹平（都已逐值核对）：
    1. **字段顺序不同**：腾讯一行是 `[date, open, close, high, low, volume]`，
       新浪是 `open/high/low/close`。照搬新浪的顺序会把 high 当 close 用。
    2. **成交量单位不同**：腾讯给「手」，新浪给「股」→ ×100 与新源等值（比值恰为 1.000000）。
    """
    resp = requests.get(
        _TENCENT_KLINE_URL,
        params={"param": f"{code},day,,,{int(count)},"},
        headers={"User-Agent": _UA},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    node = ((resp.json() or {}).get("data") or {}).get(code)
    rows = node.get("day") if isinstance(node, dict) else None
    if not rows:
        # 腾讯在参数越界/代码不存在时会把 data[code] 返回成 None 或 list，别让它静默变成空表
        raise ValueError(f"腾讯日线返回异常({code}): {str(node)[:140]}")

    df = pd.DataFrame([r[:6] for r in rows],
                      columns=["date", "open", "close", "high", "low", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = df["volume"] * 100.0
    return (df[["date", "open", "high", "low", "close", "volume"]]
            .sort_values("date").reset_index(drop=True))


def fetch_index_daily(code: str, count: int = _KLINE_COUNT,
                      include_today: bool | None = None) -> pd.DataFrame:
    """获取指数日 K 线（腾讯源，带 30 分钟缓存）。

    返回 DataFrame(date/open/high/low/close/volume)，date 为 datetime64。
    失败时回退到上次缓存；无缓存则抛异常由调用方处理。

    ## 与旧新浪源的行为对齐（换源不能改语义）

    腾讯**盘中就返回当天的实时 bar**，而新浪日线盘中只到昨天。
    若不处理，`fetch_market_regime`/`_market_trend_gate` 会在盘中把
    「还没走完的当日 bar」当成收盘价 —— 正是 2026-09-14 修掉的那类口径漂移。
    所以这里默认**盘中剔除当天**（收盘后保留），与新浪源逐日一致。
    需要强制时用 `include_today=True/False` 覆盖（测试用）。

    ## 历史长度

    腾讯单次上限 2000 根 => 只能回溯到约 2018-06（新浪能到 2002）。
    本项目所有回测起点都是 2020-01-01，够用；若将来要更早的历史，
    得改用带起止日期的 `param=code,day,<start>,<end>,<count>,` 并分批拼。
    """
    now = time.time()
    if code in _DAILY_CACHE and now - _DAILY_CACHE[code][0] < _DAILY_TTL:
        return _DAILY_CACHE[code][1]
    try:
        df = _fetch_tencent_kline(code, count)
    except Exception as exc:  # noqa: BLE001 - 网络/接口波动回退到上次缓存（docstring 承诺的行为）
        logger.warning("指数日线获取失败(%s): %s", code, exc)
        if code in _DAILY_CACHE:
            return _DAILY_CACHE[code][1]
        raise
    if include_today is None:
        include_today = not _intraday()
    if not include_today:
        cutoff = pd.Timestamp(datetime.now().date())
        df = df[df["date"] < cutoff].reset_index(drop=True)
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
