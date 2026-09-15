"""大盘指数模块 单元测试：腾讯行解析 + 日 K 线换源 + 缓存。

运行：python -m pytest tests/test_indices.py -v  或  python tests/test_indices.py
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.realtime import indices as idx              # noqa: E402
from quant.realtime.indices import IndexQuoter         # noqa: E402


def _line(point="3979.89", chg="-6.41", pct="-0.16", name="上证指数", code="sh000001"):
    return f'v_s_{code}="1~{name}~{code[2:]}~{point}~{chg}~{pct}~573538949~94430756~~704688.33~ZS~";'


def test_parse_index_line():
    """腾讯指数行解析出正确点位/涨跌/涨跌幅。"""
    q = IndexQuoter()._parse(_line(), "sh000001")
    assert q is not None
    assert q["name"] == "上证指数"
    assert q["point"] == 3979.89
    assert q["change"] == -6.41
    assert q["change_pct"] == -0.16
    assert abs(q["prev_close"] - (3979.89 + 6.41)) < 1e-6   # 昨收 = 点 - 涨跌


def test_parse_positive_change():
    """上涨行：正涨跌、正涨跌幅。"""
    q = IndexQuoter()._parse(_line(point="3000.0", chg="+12.5", pct="0.42"), "sh000001")
    assert q["change"] == 12.5
    assert q["change_pct"] == 0.42


def test_parse_bad_line():
    """非法行返回 None（不崩溃）。"""
    assert IndexQuoter()._parse("garbage", "sh000001") is None
    assert IndexQuoter()._parse('v_s_sh000001="1~";', "sh000001") is None


def test_fetch_with_cache():
    """fetch：TTL 内第二次走缓存，不重复请求。"""
    resp = mock.Mock()
    resp.text = _line() + "\n" + _line(point="13872", chg="-142", pct="-1.02",
                                       name="深证成指", code="sz399001")
    q = IndexQuoter({"sh000001": "上证指数", "sz399001": "深证成指"})
    with mock.patch("quant.realtime.indices.requests.get", return_value=resp) as m:
        r1 = q.fetch(force=True)
        r2 = q.fetch()          # 命中缓存，不再请求
        assert len(r1) == 2
        assert r1[0]["point"] == 3979.89
        assert r2 is r1         # 同一缓存对象
        assert m.call_count == 1


def test_fetch_failure_returns_cache():
    """网络失败时返回上次缓存（有缓存时）。"""
    resp = mock.Mock()
    resp.text = _line()
    q = IndexQuoter({"sh000001": "上证指数"})
    with mock.patch("quant.realtime.indices.requests.get", return_value=resp):
        q.fetch(force=True)
    with mock.patch("quant.realtime.indices.requests.get",
                    side_effect=RuntimeError("net down")):
        r = q.fetch(force=True)
        assert r and r[0]["name"] == "上证指数"


# ============================================================
# 日 K 线：腾讯源（2026-09-15 由新浪换入，为消除 py_mini_racer V8 崩溃）
# ============================================================

def _kline_resp(code="sh000300", rows=None):
    """构造腾讯 K 线接口的 mock 响应。"""
    resp = mock.Mock()
    resp.json.return_value = {"code": 0, "msg": "", "data": {code: {"day": rows}}}
    return resp


# 真实一行（2026-09-14 沪深300，取自新浪，用于逐值核对）：
#   腾讯行顺序 = [date, open, close, high, low, volume]
_REAL_ROW = ["2026-09-14", "4474.947", "4480.082", "4504.489", "4473.275", "149488418"]


def _clear_cache():
    idx._DAILY_CACHE.clear()


def test_kline_row_order_is_tencent_not_sina():
    """**核心回归**：腾讯行是 open/close/high/low，新浪是 open/high/low/close。

    照搬新浪顺序会把 high(4504.489) 当成 close 用。这里直接断言每个字段的落位。
    """
    _clear_cache()
    with mock.patch("quant.realtime.indices.requests.get",
                    return_value=_kline_resp(rows=[_REAL_ROW])):
        df = idx.fetch_index_daily("sh000300", include_today=True)
    assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
    r = df.iloc[0]
    assert abs(r["open"] - 4474.947) < 1e-6
    assert abs(r["close"] - 4480.082) < 1e-6      # 第 3 列才是 close
    assert abs(r["high"] - 4504.489) < 1e-6       # 第 4 列是 high
    assert abs(r["low"] - 4473.275) < 1e-6        # 第 5 列是 low
    # 顺序若搞错，这个不变量立刻破：high 必须 >= open/close
    assert r["high"] >= max(r["open"], r["close"])
    assert r["low"] <= min(r["open"], r["close"])


def test_kline_volume_converted_to_shares():
    """腾讯给「手」、旧新浪源给「股」-> 必须 ×100 才能等值（实测比值恰为 1.000000）。"""
    _clear_cache()
    with mock.patch("quant.realtime.indices.requests.get",
                    return_value=_kline_resp(rows=[_REAL_ROW])):
        df = idx.fetch_index_daily("sh000300", include_today=True)
    assert abs(df.iloc[0]["volume"] - 149488418 * 100) < 1e-3


def test_kline_sorted_and_date_is_datetime():
    """日期升序 + datetime64（调用方普遍 set_index('date').sort_index()）。"""
    _clear_cache()
    rows = [["2026-09-11", "4514.295", "4510.155", "4520.170", "4461.570", "204230027"],
            _REAL_ROW]
    with mock.patch("quant.realtime.indices.requests.get",
                    return_value=_kline_resp(rows=rows)):
        df = idx.fetch_index_daily("sh000300", include_today=True)
    assert str(df["date"].dtype).startswith("datetime64")
    assert df["date"].is_monotonic_increasing


def test_intraday_drops_today_but_close_keeps_it():
    """腾讯盘中就带当日实时 bar，新浪不带 -> 默认盘中剔除，收盘后保留。"""
    _clear_cache()
    today = datetime.now().strftime("%Y-%m-%d")
    rows = [_REAL_ROW, [today, "4500", "4520", "4530", "4495", "1000"]]
    with mock.patch("quant.realtime.indices.requests.get",
                    return_value=_kline_resp(rows=rows)):
        dropped = idx.fetch_index_daily("sh000300", include_today=False)
        _clear_cache()
        kept = idx.fetch_index_daily("sh000300", include_today=True)
    assert len(dropped) == 1 and len(kept) == 2
    assert today not in set(dropped["date"].dt.strftime("%Y-%m-%d"))
    assert today in set(kept["date"].dt.strftime("%Y-%m-%d"))


def test_intraday_rule_boundary():
    """14:59 算盘中、15:00 起算收盘（当日 bar 已走完）。"""
    d = datetime(2026, 9, 15)
    assert idx._intraday(d.replace(hour=9, minute=31)) is True
    assert idx._intraday(d.replace(hour=14, minute=59)) is True
    assert idx._intraday(d.replace(hour=15, minute=0)) is False
    assert idx._intraday(d.replace(hour=20, minute=0)) is False


def test_kline_bad_payload_raises():
    """腾讯越界/代码不存在时 data[code] 会变成 None 或 list —— 必须显式报错，不能静默变空表。"""
    _clear_cache()
    for bad in [None, [], "oops"]:
        resp = mock.Mock()
        resp.json.return_value = {"code": 0, "data": {"sh000300": bad}}
        with mock.patch("quant.realtime.indices.requests.get", return_value=resp):
            try:
                idx.fetch_index_daily("sh000300")
                raise AssertionError(f"应当抛异常: {bad!r}")
            except ValueError:
                pass


def test_kline_failure_falls_back_to_cache():
    """首次成功后再失败 -> 回退到上次缓存（docstring 承诺的行为）。"""
    _clear_cache()
    with mock.patch("quant.realtime.indices.requests.get",
                    return_value=_kline_resp(rows=[_REAL_ROW])):
        first = idx.fetch_index_daily("sh000300", include_today=True)
    with mock.patch("quant.realtime.indices.requests.get",
                    side_effect=RuntimeError("net down")):
        again = idx.fetch_index_daily("sh000300")          # 缓存已过期 -> 必须回退而非抛
    assert len(again) == len(first) == 1


def test_kline_no_akshare_import():
    """**根因护栏**：本模块不得再 import akshare / py_mini_racer。

    2026-09-15 的崩溃根因就是 akshare 新浪日线内部新建 MiniRacer 跑 JS。
    用 AST 检查（而不是搜字符串），这样 docstring 里解释「为什么离开 akshare」不会误报。
    """
    import ast

    src = (Path(__file__).resolve().parents[1]
           / "quant" / "realtime" / "indices.py").read_text(encoding="utf-8")
    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "akshare" not in imported, "indices.py 不能再 import akshare（会带进 py_mini_racer/V8）"
    assert "py_mini_racer" not in imported


if __name__ == "__main__":
    tests = [test_parse_index_line, test_parse_positive_change,
             test_parse_bad_line, test_fetch_with_cache,
             test_fetch_failure_returns_cache,
             test_kline_row_order_is_tencent_not_sina,
             test_kline_volume_converted_to_shares,
             test_kline_sorted_and_date_is_datetime,
             test_intraday_drops_today_but_close_keeps_it,
             test_intraday_rule_boundary,
             test_kline_bad_payload_raises,
             test_kline_failure_falls_back_to_cache,
             test_kline_no_akshare_import]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"all {len(tests)} passed")
