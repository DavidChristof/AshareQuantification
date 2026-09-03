"""大盘指数模块 单元测试：腾讯行解析 + 缓存。

运行：python -m pytest tests/test_indices.py -v  或  python tests/test_indices.py
"""
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.realtime.indices import IndexQuoter     # noqa: E402


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


if __name__ == "__main__":
    tests = [test_parse_index_line, test_parse_positive_change,
             test_parse_bad_line, test_fetch_with_cache,
             test_fetch_failure_returns_cache]
    for fn in tests:
        fn()
        print(f"PASS {fn.__name__}")
    print("全部通过")
