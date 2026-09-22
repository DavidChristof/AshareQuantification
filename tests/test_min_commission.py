"""最低佣金 / 过户费 / 实际手续费回报 的回归护栏（quant/trading/paper.py）。

核心要求：`min_commission=0`（默认）时新旧公式**逐值相等** —— 自动纸面盘、模拟炒股、
回测全部不受影响；只有实盘账户（min_commission=5.0, transfer_fee=0.00001）走真实口径。

运行：python tests/test_min_commission.py
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker

_TMP_UID = [0]
# 本次进程的唯一标识。**只用 pid 会被复用** —— 上一轮残留的 paper/_test_fee_<pid>_1.db
# 会被新进程当成自己的库打开，于是 query_cash() 不是初值，测试间歇性失败
# （2026-09-22 实测：单跑 5/5 通过、放进全量套件里偶发失败）。
# 加一个时间戳把这条路堵死。
_RUN_TAG = f"{os.getpid()}_{time.time_ns() % 10_000_000}"


def _tmp_db():
    _TMP_UID[0] += 1
    return f"paper/_test_fee_{_RUN_TAG}_{_TMP_UID[0]}.db"


def _rm(tmp):
    """删主库**与三个边车文件**（-journal/-wal/-shm）。

    [!] 这个函数**实际上删不掉**，是已知的：调用它时测试里的 broker 变量还活着，
    连接还开着，**Windows 会锁住文件** -> `os.remove` 抛 `PermissionError`（WinError 32）
    -> 被下面的 `except OSError: pass` **静默吞掉**。实测：`del b` + `gc.collect()` 之后
    就能删（2026-09-22 验证）。所以 paper/ 下会持续攒残留（当时 1481 个）。

    **这不会让测试出错**（因为 `_tmp_db()` 的名字带本进程唯一标识，撞不上旧文件），
    但文件会越积越多。要真正清理，得在每个测试里 `del b` 之后再调 `_rm`，
    或者改成 `atexit` 统一清 —— 那是另一件事，本次没做。
    """
    for suf in ("", "-journal", "-wal", "-shm"):
        try:
            os.remove(tmp + suf)
        except OSError:
            pass


def test_default_matches_legacy_formula_exactly():
    """默认（无最低佣金/过户费）：与旧公式 amount*commission / proceeds*(c+stamp) 完全一致。"""
    tmp = _tmp_db()
    b = PaperBroker(tmp, initial_capital=100_000.0,
                    commission=0.0003, slippage=0.0002, stamp_tax=0.0005)
    r = b.buy("600519", 100, 10.0, "2026-09-10")
    buy_price = 10.0 * (1 + 0.0002)
    assert r.success
    assert abs(r.fee - 100 * buy_price * 0.0003) < 1e-12
    s = b.sell("600519", 100, 10.0, "2026-09-11")
    sell_price = 10.0 * (1 - 0.0002)
    assert abs(s.fee - 100 * sell_price * (0.0003 + 0.0005)) < 1e-12
    _rm(tmp)


def test_min_commission_floors():
    tmp = _tmp_db()
    b = PaperBroker(tmp, initial_capital=100_000.0, commission=0.0003,
                    slippage=0.0, stamp_tax=0.0005, lot_size=100,
                    min_commission=5.0, transfer_fee=0.00001)
    r = b.buy("600519", 100, 10.0, "2026-09-10")          # 万3 仅 ￥0.30 → 取 ￥5
    assert r.success
    assert abs(r.fee - (5.0 + 1000.0 * 0.00001)) < 1e-9
    # 现金按「金额 + 实收费」扣减
    assert abs(b.query_cash() - (100_000.0 - 1000.0 - r.fee)) < 1e-6
    # 大额走费率而非最低佣金
    b2 = PaperBroker(_tmp_db(), initial_capital=1_000_000.0, commission=0.0003,
                     slippage=0.0, min_commission=5.0)
    r2 = b2.buy("600519", 1000, 10.0, "2026-09-10")       # 万3 = ￥3.0 < 5 → 仍取 5
    assert abs(r2.fee - 5.0) < 1e-9
    b3 = PaperBroker(_tmp_db(), initial_capital=10_000_000.0, commission=0.0003,
                     slippage=0.0, min_commission=5.0)
    r3 = b3.buy("600519", 100_000, 10.0, "2026-09-10")    # 万3 = ￥300 > 5
    assert abs(r3.fee - 300.0) < 1e-9
    _rm(tmp)


def test_fee_override_used_for_reported_fee():
    """实盘按券商回报的实际手续费记账；非法值（NaN/负）退回估算。"""
    tmp = _tmp_db()
    b = PaperBroker(tmp, initial_capital=100_000.0, commission=0.0003,
                    slippage=0.0, stamp_tax=0.0005, lot_size=100,
                    min_commission=5.0)
    r = b.buy("600519", 100, 10.0, "2026-09-10", fee_override=6.37)
    assert r.success and abs(r.fee - 6.37) < 1e-9
    bad = b.buy("600519", 100, 10.0, "2026-09-10", fee_override=float("nan"))
    assert bad.success and abs(bad.fee - 5.0) < 1e-9        # 退回估算
    neg = b.buy("600519", 100, 10.0, "2026-09-10", fee_override=-1)
    assert neg.success and abs(neg.fee - 5.0) < 1e-9
    _rm(tmp)


def test_nan_price_rejected_cash_intact_with_real_fees():
    """实盘费用口径下，NaN 价仍被拒、现金分毫不动（2026-09-09 事故口径不回归）。"""
    tmp = _tmp_db()
    b = PaperBroker(tmp, initial_capital=3000.0, lot_size=100,
                    min_commission=5.0, transfer_fee=0.00001)
    assert b.buy("600519", 100, float("nan"), "2026-09-10").success is False
    assert b.query_cash() == 3000.0
    assert b.query_positions() == []
    _rm(tmp)


def test_transfer_fee_charged_both_sides():
    tmp = _tmp_db()
    b = PaperBroker(tmp, initial_capital=100_000.0, commission=0.0003,
                    slippage=0.0, stamp_tax=0.0005, lot_size=100,
                    min_commission=0.0, transfer_fee=0.00002)
    r = b.buy("600519", 100, 100.0, "2026-09-10")
    assert abs(r.fee - (10_000 * 0.0003 + 10_000 * 0.00002)) < 1e-9
    s = b.sell("600519", 100, 100.0, "2026-09-11")
    assert abs(s.fee - (10_000 * 0.0003 + 10_000 * 0.0005 + 10_000 * 0.00002)) < 1e-9
    _rm(tmp)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
