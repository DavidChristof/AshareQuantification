"""时点财务指标（`quant/data/fundamentals.py`）单元测试。

重点钉住三件**错了就会静默产生未来函数**的事：
  1. **去累计**：接口给的是年内累计值，直接当季度值用会错。
  2. **披露滞后**：报告期 ≠ 公告日。2020-06-30 的半年报，7 月**看不到**，8/31 之后才看得到。
  3. **篡改未来报告不能影响历史面板**（与宇宙重建同款的最有力断言）。

运行：python tests/test_fundamentals.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.fundamentals import (disclosure_date, pit_pe_roe_panels,  # noqa: E402
                                     prepare_financials, ttm_panel)


def _raw(symbol="600000", rows=None) -> pd.DataFrame:
    """构造原始财务表（eps/roe 为**年内累计**值，模拟接口口径）。"""
    rows = rows or [
        ("2020-03-31", 1.0, 5.0),      # Q1 累计
        ("2020-06-30", 2.5, 11.0),     # H1 累计
        ("2020-09-30", 4.0, 17.0),     # Q3 累计
        ("2020-12-31", 5.0, 22.0),     # 全年累计
        ("2021-03-31", 1.2, 6.0),      # 次年 Q1：**累计重置**
        ("2021-06-30", 2.8, 13.0),
        ("2021-09-30", 4.5, 19.5),
        ("2021-12-31", 6.0, 26.0),
    ]
    return pd.DataFrame([{"symbol": symbol, "report_date": d, "eps_diluted": e,
                          "eps_weighted": e, "roe": r, "roe_weighted": r}
                         for d, e, r in rows])


# ============================================================
# 1. 披露滞后
# ============================================================
def test_disclosure_date_known():
    assert disclosure_date("2020-03-31") == date(2020, 4, 30)    # 一季报
    assert disclosure_date("2020-06-30") == date(2020, 8, 31)    # 半年报
    assert disclosure_date("2020-09-30") == date(2020, 10, 31)   # 三季报
    assert disclosure_date("2020-12-31") == date(2021, 4, 30)    # 年报 → 次年 4/30
    assert disclosure_date("2021-12-31") == date(2022, 4, 30)
    assert disclosure_date(None) is None
    assert disclosure_date("not-a-date") is None


# ============================================================
# 2. 去累计
# ============================================================
def test_decumulate_single_quarter():
    fin = prepare_financials(_raw())
    got = dict(zip(fin["report_date"].dt.strftime("%Y-%m-%d"), fin["eps_q"]))
    # 累计 1.0 → 2.5 → 4.0 → 5.0 ⇒ 单季 1.0 → 1.5 → 1.5 → 1.0
    assert got["2020-03-31"] == 1.0
    assert got["2020-06-30"] == 1.5
    assert got["2020-09-30"] == 1.5
    assert got["2020-12-31"] == 1.0
    # 次年 Q1 必须**重置**（不是 1.2 - 5.0 = -3.8）
    assert got["2021-03-31"] == 1.2


def test_decumulate_roe_and_roundtrip():
    fin = prepare_financials(_raw())
    q = dict(zip(fin["report_date"].dt.strftime("%Y-%m-%d"), fin["roe_q"]))
    assert q["2020-03-31"] == 5.0
    assert q["2020-06-30"] == 6.0
    assert q["2020-12-31"] == 5.0
    # 单季之和应还原累计值
    assert abs(sum(q[f"2020-{m}"] for m in ("03-31", "06-30", "09-30", "12-31")) - 22.0) < 1e-9


def test_prepare_handles_empty_and_missing_column():
    assert prepare_financials(pd.DataFrame()).empty
    df = _raw()
    df["eps_diluted"] = np.nan                 # 摊薄缺失 → 回退加权
    fin = prepare_financials(df)
    assert fin["eps_ytd"].notna().all()


# ============================================================
# 3. TTM 与可见性（无未来函数）
# ============================================================
def _dates(a: str, b: str) -> pd.DatetimeIndex:
    return pd.date_range(a, b, freq="D")


def test_ttm_needs_four_quarters():
    fin = prepare_financials(_raw())
    d = _dates("2020-01-01", "2022-06-30")
    panel = ttm_panel(fin, "eps_q", d, ["600000"])
    s = panel["600000"]
    # 2020-12-31 的报告要到 2021-04-30 才可见 ⇒ 之前只有 3 个单季，凑不满 TTM
    assert np.isnan(s.loc["2021-01-15"])
    assert np.isnan(s.loc["2021-04-29"])
    # 2021-04-30 起年报可见（当天一季报也同时可见，取最新）⇒ TTM = 1.5+1.5+1.0+1.2 = 5.2
    assert abs(s.loc["2021-04-30"] - 5.2) < 1e-9
    # 2021-08-31 起 H1 可见，TTM = 1.5+1.0+1.2+1.6 = 5.3
    assert abs(s.loc["2021-08-31"] - 5.3) < 1e-9


def test_report_invisible_before_deadline():
    """半年报 6/30 的数据，8/31 之前绝不可见（这是最容易出的未来函数）。

    注意断言的是**数值**而不是 NaN：7 月仍有值，但那个值必须仍基于一季报
    （TTM=5.2），**不能**是含半年报的 5.3。若这里查成 5.3，就是偷看了未来。
    """
    fin = prepare_financials(_raw())
    d = _dates("2021-06-01", "2021-10-01")
    s = ttm_panel(fin, "eps_q", d, ["600000"])["600000"]
    assert abs(s.loc["2021-06-15"] - 5.2) < 1e-9      # 半年报前：基于一季报
    assert abs(s.loc["2021-07-01"] - 5.2) < 1e-9
    assert abs(s.loc["2021-08-30"] - 5.2) < 1e-9      # 截止日前一天仍是 5.2
    assert abs(s.loc["2021-08-31"] - 5.3) < 1e-9      # 8/31 起才换成 5.3


def test_no_future_leak_on_mutation():
    """篡改**未来**的报告，历史面板必须逐值不变。"""
    d = _dates("2021-01-01", "2022-12-31")
    fin1 = prepare_financials(_raw())
    p1 = ttm_panel(fin1, "eps_q", d, ["600000"])

    raw2 = _raw()
    m = raw2["report_date"] > "2021-06-30"          # 只改未来
    raw2.loc[m, "eps_diluted"] *= 10
    raw2.loc[m, "eps_weighted"] *= 10
    fin2 = prepare_financials(raw2)
    p2 = ttm_panel(fin2, "eps_q", d, ["600000"])

    cut = pd.Timestamp("2021-08-31")                # 第一期不可见的最后一天
    a, b = p1.loc[:cut, "600000"], p2.loc[:cut, "600000"]
    assert a.equals(b), "历史面板被未来数据影响了 → 存在未来函数"
    # 而未来那段确实应该变（证明测试本身有效，不是恒等）
    assert not p1.loc["2022-01-01":, "600000"].equals(p2.loc["2022-01-01":, "600000"])


def test_pit_pe_roe_panels_shapes_and_pe_sign():
    fin = prepare_financials(_raw())
    idx = _dates("2021-01-01", "2022-02-03")
    close = pd.DataFrame({"600000": np.linspace(10, 20, len(idx))}, index=idx)
    pe, roe = pit_pe_roe_panels(fin, close)
    assert pe.shape == close.shape and roe.shape == close.shape
    assert bool((pe.dropna() > 0).to_numpy().all()), "亏损（EPS<=0）应置 NaN，PE 只留正值"
    # 边界（本用例最能说明时点性）：年报的法定截止日就是 4/30，
    # 所以 2021-04-29 连 2020 年报都看不到 → 只剩 3 个季度 → TTM 不可算 → PE 为 NaN。
    assert pd.isna(pe["600000"].loc["2021-04-29"]), "年报在 4/30 之前不该可见"
    # 2021-04-30 当天：年报与一季报**同时**可见，取最新（一季报）
    # ⇒ TTM = 1.5+1.5+1.0+1.2 = 5.2
    px = close["600000"].loc["2021-04-30"]
    assert abs(pe["600000"].loc["2021-04-30"] - px / 5.2) < 1e-6
    # 次一个可见日（2021-08-31 中报）之后 TTM 换成 1.5+1.0+1.2+1.6 = 5.3
    px2 = close["600000"].loc["2021-08-31"]
    assert abs(pe["600000"].loc["2021-08-31"] - px2 / 5.3) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
