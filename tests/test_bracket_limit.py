"""止损/止盈线「板块涨跌停封顶」单元测试。

背景：ATR 动态线（止损≤15%/止盈≤30%）不看板块，可能超出主板 ±10% / 创·科 ±20% 的涨跌停幅。
- 止盈超涨停 → 当天到不了价、只能死等多日（纸面能成交、实盘未必）；
- 止损超跌停 → 触发被跌停天然延后，实际亏得比线更多（假止损）。

修复：quant/trading/paper.py:apply_stop_rules 与 api/main.py:_position_risk 在算出
止损/止盈/移动止损百分比后，再按个股板块限幅（rules.limit_pct）封顶。

运行：python -m pytest tests/test_bracket_limit.py -v  或  python tests/test_bracket_limit.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.trading.paper import PaperBroker
from quant.trading.rules import limit_pct

_TMP_UID = [0]


def _tmp_db():
    _TMP_UID[0] += 1
    return f"paper/_test_bl_{os.getpid()}_{_TMP_UID[0]}.db"


def _broker(tmp):
    return PaperBroker(tmp, initial_capital=100_000.0, lot_size=100)


def _rm(tmp):
    try:
        os.remove(tmp)
    except OSError:
        pass


def test_main_board_high_vol_lines_capped_at_10pct():
    """主板高波动股：ATR 给出的原始线远超 ±10%，封顶后止盈在 +10% 附近即触发。

    成本≈100.02，ATR=20/日 → 原始止损≈50%→钳到15%、原始止盈≈70%→钳到30%；
    再按主板 ±10% 封顶 → 止损≈9.5%、止盈≈10%：
      - 现价 90.0（旧代码原始止损 85 不会触发；封顶后 90.52 会触发）→ 止损
      - 现价 110.5（旧代码原始止盈 130 不会触发；封顶后 110.02 会触发）→ 止盈
    """
    vol_cfg = dict(atr_stop_mult=2.5, atr_take_mult=3.5, atr_trailing_mult=2.5,
                   vol_min_pct=0.03, vol_max_pct=0.15,
                   take_min_pct=0.05, take_max_pct=0.30, trailing_enabled=True)
    vol = {"605117": {"atr": 20.0, "close": 100.0, "atr_pct": 0.2}}
    assert limit_pct("605117") == 0.10

    # --- 止损侧：90.0 触发（封顶止损 ≈ cost×0.905 ≈ 90.52）---
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("605117", 100, 100.0, "2026-08-01")
        tr = b.apply_stop_rules("2026-08-10", {"605117": 90.0}, vol=vol, vol_cfg=vol_cfg)
        assert tr and tr[0]["symbol"] == "605117" and "止损" in tr[0]["reason"], tr
    finally:
        _rm(tmp)

    # --- 止盈侧：110.5 触发（封顶止盈 ≈ cost×1.10 ≈ 110.02）---
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("605117", 100, 100.0, "2026-08-01")
        tr = b.apply_stop_rules("2026-08-10", {"605117": 110.5}, vol=vol, vol_cfg=vol_cfg)
        assert tr and tr[0]["symbol"] == "605117" and "止盈" in tr[0]["reason"], tr
    finally:
        _rm(tmp)


def test_chinext_allows_up_to_20pct():
    """创业板(300)/科创板(688) ±20%：高波动线可到 20%，不受主板 10% 误伤。

    成本≈100.02，ATR=40/日 → 原始止损钳到15%（≤20 不再收窄），原始止盈→钳到20%。
      - 现价 118（介于 +10%~+20% 之间，只该是“+20% 封顶”的名称才会触发区间外的止盈）：
        20% 封顶止盈 ≈ cost×1.20 ≈ 120.02 → 118 不触发
      - 现价 121 → 触发止盈
    """
    vol_cfg = dict(atr_stop_mult=2.5, atr_take_mult=3.5, atr_trailing_mult=2.5,
                   vol_min_pct=0.03, vol_max_pct=0.15,
                   take_min_pct=0.05, take_max_pct=0.30, trailing_enabled=True)
    vol = {"688111": {"atr": 40.0, "close": 100.0, "atr_pct": 0.4}}
    assert limit_pct("688111") == 0.20

    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("688111", 100, 100.0, "2026-08-01")
        assert b.apply_stop_rules("2026-08-10", {"688111": 118.0}, vol=vol, vol_cfg=vol_cfg) == []
        tr = b.apply_stop_rules("2026-08-10", {"688111": 121.0}, vol=vol, vol_cfg=vol_cfg)
        assert tr and "止盈" in tr[0]["reason"], tr
    finally:
        _rm(tmp)


def test_fixed_fallback_15pct_take_capped_on_main_board():
    """固定百分比回退（无 vol 时）：止盈默认 15% 对主板超涨停，同样被压到 +10%。"""
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("600519", 100, 100.0, "2026-08-01")   # 主板，无 vol → 固定回退 8%/15%
        # 封顶后止盈 ≈ cost×1.10 ≈ 110.02：112 触发（旧代码 15%→需 115.03 才触发）
        tr = b.apply_stop_rules("2026-08-10", {"600519": 112.0},
                                stop_loss_pct=0.08, take_profit_pct=0.15)
        assert tr and "止盈" in tr[0]["reason"], tr
        # 止损固定 8%（≤10% 不受影响）：93 不触发（cost×0.92≈92.02 才触发）
        tmp2 = _tmp_db()
        b2 = _broker(tmp2)
        try:
            b2.buy("600519", 100, 100.0, "2026-08-01")
            assert b2.apply_stop_rules("2026-08-10", {"600519": 93.0},
                                       stop_loss_pct=0.08, take_profit_pct=0.15) == []
        finally:
            _rm(tmp2)
    finally:
        _rm(tmp)


def test_separate_stop_take_flags():
    """apply_stop=False → 只止盈不触止损；apply_take=False → 只止损不止盈。

    用于「止盈盘中(实时价)、止损收盘」的分开调用（2026-09-08 手动盘口径）。
    成本≈100.02，固定止损 8%、止盈 10%：
      - apply_stop=False + 现价 90(≤止损92) → 不卖
      - apply_stop=False + 现价 115(≥止盈110) → 止盈卖出
      - apply_take=False + 现价 90 → 止损卖出
      - apply_take=False + 现价 115 → 不卖
    """
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("600519", 100, 100.0, "2026-08-01")   # 主板，固定 8%/10%
        args = dict(stop_loss_pct=0.08, take_profit_pct=0.10)
        assert b.apply_stop_rules("2026-08-10", {"600519": 90.0},
                                  apply_stop=False, apply_take=True, **args) == []
        tr = b.apply_stop_rules("2026-08-10", {"600519": 115.0},
                                apply_stop=False, apply_take=True, **args)
        assert tr and "止盈" in tr[0]["reason"], tr
    finally:
        _rm(tmp)
    tmp = _tmp_db()
    b = _broker(tmp)
    try:
        b.buy("600519", 100, 100.0, "2026-08-01")
        args = dict(stop_loss_pct=0.08, take_profit_pct=0.10)
        tr = b.apply_stop_rules("2026-08-10", {"600519": 90.0},
                                apply_stop=True, apply_take=False, **args)
        assert tr and "止损" in tr[0]["reason"], tr
        assert b.apply_stop_rules("2026-08-10", {"600519": 115.0},
                                  apply_stop=True, apply_take=False, **args) == []
    finally:
        _rm(tmp)


if __name__ == "__main__":
    tests = [test_main_board_high_vol_lines_capped_at_10pct,
             test_chinext_allows_up_to_20pct,
             test_fixed_fallback_15pct_take_capped_on_main_board,
             test_separate_stop_take_flags]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {fn.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
