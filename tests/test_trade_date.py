"""成交/记账日期口径测试：`trade_date()`（今天）不能与行情数据日期混用。

## 背景（2026-09-14 事故）

行情数据是**收盘后**才刷新的（config `auto_refresh.update_time: 15:30`），所以
「信号表最后一天」在盘中/盘前还停在**上一个交易日**。早期代码到处用
`str(sig.index[-1].date())` 当"今天"，于是周一盘中做的调仓被记成上周五，
`INSERT OR REPLACE` 还把上周五的收盘净值点用今天的账户状态覆盖了。

运行：python tests/test_trade_date.py
"""
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker, trade_date      # noqa: E402
from quant.trading.real_account import RealBroker            # noqa: E402

_TMP_UID = [0]
REAL_KW = dict(initial_capital=3000.0, commission=0.0003, slippage=0.0,
               stamp_tax=0.0005, lot_size=100, min_commission=5.0,
               transfer_fee=0.00001)


# 每次**运行**唯一的后缀。只用 pid 会被操作系统复用 -> 撞上 paper/ 下的残留同名库
# -> PaperBroker 打开到有历史状态的旧库 -> 测试间歇性失败（2026-09-22 实测）。
_RUN_TAG = os.urandom(4).hex()


def _tmp_db():
    _TMP_UID[0] += 1
    return f"paper/_test_tradedate_{os.getpid()}_{_RUN_TAG}_{_TMP_UID[0]}.db"


def _rm(tmp):
    try:
        os.remove(tmp)
    except OSError:
        pass


def _broker(tmp):
    return PaperBroker(tmp, initial_capital=100_000.0, commission=0.0003,
                       slippage=0.0, stamp_tax=0.0005, lot_size=1,
                       min_commission=0.0, transfer_fee=0.0)


def test_trade_date_is_calendar_date():
    """`trade_date()` 必须等于**日历上的今天**，与行情数据覆盖到哪天无关。"""
    assert trade_date() == date.today().isoformat()


def test_trade_date_accepts_explicit_day():
    assert trade_date(date(2026, 9, 14)) == "2026-09-14"
    assert trade_date(date(2026, 1, 5)) == "2026-01-05"


def test_trade_date_zero_pads():
    """月份/日期必须补零 —— 否则和库里 'YYYY-MM-DD' 的字符串比较会错位。"""
    assert trade_date(date(2026, 9, 4)) == "2026-09-04"


def test_t1_check_must_use_same_date_as_recording():
    """T+1 用日期字符串**精确匹配**当日买入 => 记账与校验必须用同一个日期来源。

    这是本事故里最容易漏的连带坑：只把「成交日期」改成日历日期、而 T+1 校验
    还在用行情数据日期，当天买入就会被判成可卖（T+1 静默失效）。
    这里同时断言"用错日期会误判"，把这个耦合钉死。（sellable_shares 在 RealBroker 上。）
    """
    tmp = _tmp_db()
    b = RealBroker(tmp, **REAL_KW)
    r = b.record_execution("601600", "buy", 100, 10.0, trade_date(), fee=0.0)
    assert r.success, f"买入应成功: {getattr(r, 'message', '')}"
    # 用同一个日期来源 → 正确判为不可卖
    assert b.sellable_shares("601600", trade_date()) == 0.0
    # 反证：换一个日期（等价于"T+1 用了行情数据日期"）→ 误判为全部可卖
    assert b.sellable_shares("601600", "2020-01-02") == 100.0
    _rm(tmp)


def test_snapshot_equity_writes_the_date_given():
    """`snapshot_equity` 完全按传入日期落库（INSERT OR REPLACE）。

    => 调用方给错日期就会**覆盖**那一天已有的点 —— 这正是 09-11 日点被今天
    账户状态覆盖的机制。日期必须由调用方保证正确。
    """
    tmp = _tmp_db()
    b = _broker(tmp)
    b.buy("601600", 1000, 10.0, "2026-09-11")
    b.snapshot_equity("2026-09-11", {"601600": 9.44})
    first = [r for r in b.equity_history() if r["date"] == "2026-09-11"]
    assert len(first) == 1
    # 用同一天再写一次（模拟"今天的账户状态用了昨天的日期"）→ 覆盖
    b.snapshot_equity("2026-09-11", {"601600": 99.0})
    again = [r for r in b.equity_history() if r["date"] == "2026-09-11"]
    assert len(again) == 1, "同一天应只有一行（INSERT OR REPLACE）"
    assert again[0]["market_value"] > first[0]["market_value"], "旧点被覆盖了"
    _rm(tmp)


def test_equity_history_orders_day_point_after_intraday():
    """同日：盘中点在前，**日点（收盘点）在最后**。

    `paper_equity.date` 混存 `'YYYY-MM-DD'`（日点）与 `'YYYY-MM-DD HH:00'`（盘中点），
    字符串比较 `'2026-09-11' < '2026-09-11 09:00'` => 只写 `ORDER BY date` 会把
    **当日日点排到当日 09:00 之前**，净值曲线上表现为「收盘点画在开盘点前面」。
    这里**故意先写日点再写盘中点**，确保排序不依赖写入顺序。
    """
    tmp = _tmp_db()
    b = _broker(tmp)
    b.buy("601600", 100, 10.0, "2026-09-11")
    b.snapshot_equity("2026-09-11", {"601600": 9.44})          # 日点（先写）
    b.snapshot_equity("2026-09-11 09:00", {"601600": 9.50})
    b.snapshot_equity("2026-09-11 15:00", {"601600": 9.60})
    b.snapshot_equity("2026-09-14", {"601600": 9.70})          # 次日日点
    dates = [r["date"] for r in b.equity_history()]
    assert dates == ["2026-09-11 09:00", "2026-09-11 15:00", "2026-09-11",
                     "2026-09-14"], dates
    _rm(tmp)


if __name__ == "__main__":
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    ok = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
            ok += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{ok}/{len(fns)} passed")
    sys.exit(0 if ok == len(fns) else 1)
