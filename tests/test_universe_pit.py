"""时点宇宙重建（`quant/data/universe_pit.py`）单元测试。

重点不是"跑通"，而是钉住三条**容易错且后果严重**的性质：
  1. **无未来函数**：篡改截止日之后的数据，历史快照必须逐票不变。
  2. **调仓日顺延**：必须查真实交易日历（2021-06-11 的下一交易日是 06-15，因 6/14 端午休市）。
  3. **剔除顺序**：成交额后 20% 的分母必须是**上一步剔除之后**的剩余集合。

运行：python tests/test_universe_pit.py
"""
import os
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant.data.universe_pit import (CSI1000, CSI500, HS300, PitRules,  # noqa: E402
                                     build_mask, build_snapshot, build_snapshots,
                                     cut_bottom_by_amount, load_snapshots,
                                     next_trading_day, pick_members,
                                     review_effective_dates, save_snapshots,
                                     second_friday, semi_annual_cutoffs)

_UID = [0]


def _tmp_db() -> Path:
    _UID[0] += 1
    d = Path(__file__).resolve().parents[1] / "paper"
    d.mkdir(exist_ok=True)
    return d / f"_test_pit_{os.getpid()}_{_UID[0]}.db"


def _rm(p: Path):
    for suf in ("", "-wal", "-shm"):
        try:
            os.remove(str(p) + suf)
        except OSError:
            pass


def _trading_days(start="2019-01-01", n=700) -> list[str]:
    """合成交易日（跳过周末；够长以满足 windows 回看）。"""
    out, d = [], date(*map(int, start.split("-")))
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def _make_db(symbols: list[str], days: list[str], *, skip_before: int = 0) -> Path:
    """建临时 full_daily：每只票每天一行（用序号制造有区分度的 mcap/amount）。"""
    p = _tmp_db()
    con = sqlite3.connect(str(p))
    con.executescript("""
        CREATE TABLE full_daily (
            symbol TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
            volume REAL, amount REAL, outstanding_share REAL, turnover REAL,
            float_mcap REAL, PRIMARY KEY (symbol, date));
    """)
    rows = []
    for i, s in enumerate(symbols):
        for j, d in enumerate(days):
            if j < skip_before and i == 0:          # 让首只票上市晚一点
                continue
            mcap = 1e10 * (len(symbols) - i)        # 序号越靠前市值越大
            amt = 1e8 * (1 + (j % 7))               # 成交额随时间波动
            rows.append((s, d, 10.0, 10.0, 10.0, 10.0, 1e6, amt, 1e8, amt / 1e8 / 10, mcap))
    con.executemany("INSERT INTO full_daily VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return p


# ============================================================
# 1. 调仓日
# ============================================================
def test_second_friday_known():
    """各年 6/12 月第二个星期五（已用日历核对）。"""
    cases = {(2020, 6): date(2020, 6, 12), (2020, 12): date(2020, 12, 11),
             (2021, 6): date(2021, 6, 11), (2021, 12): date(2021, 12, 10),
             (2024, 6): date(2024, 6, 14), (2024, 12): date(2024, 12, 13),
             (2026, 6): date(2026, 6, 12), (2026, 12): date(2026, 12, 11)}
    for (y, m), want in cases.items():
        got = second_friday(y, m)
        assert got == want, f"{y}-{m}: {got} != {want}"
        assert got.weekday() == 4, "必须是星期五"


def test_semi_annual_cutoffs_sorted():
    cs = semi_annual_cutoffs([2021])
    assert cs == [date(2021, 6, 11), date(2021, 12, 10)]


def test_next_trading_day_skips_holiday():
    """2021-06-11（周五）之后 6/12~6/14 休市（端午）→ 下一个交易日是 06-15。"""
    days = [d for d in _trading_days("2021-06-01", 40) if d not in ("2021-06-14",)]
    assert next_trading_day(date(2021, 6, 11), days) == date(2021, 6, 15)
    # 国庆：2021-09-30 之后连休 → 下一个交易日 2021-10-08
    days2 = [d for d in _trading_days("2021-09-20", 40)
             if not ("2021-10-01" <= d <= "2021-10-07")]
    assert next_trading_day(date(2021, 9, 30), days2) == date(2021, 10, 8)


def test_review_effective_dates_all_trading_days():
    days = _trading_days("2020-01-01", 1500)
    eff = review_effective_dates(semi_annual_cutoffs([2020, 2021, 2022]), days)
    assert len(eff) == 6
    for cutoff, e in eff.items():
        assert e.isoformat() in days, f"生效日 {e} 不是交易日"
        assert e > cutoff, "生效日必须晚于截止日"


# ============================================================
# 2. 选样规则
# ============================================================
def _stats(pairs: dict[str, tuple[float, float]], n_days: int = 250) -> pd.DataFrame:
    """{symbol: (mcap, amt)} → stats 表。"""
    df = pd.DataFrame(
        {"mcap_avg": {s: v[0] for s, v in pairs.items()},
         "amt_avg": {s: v[1] for s, v in pairs.items()},
         "n_days": n_days})
    return df


def test_cut_bottom_by_amount_uses_passed_series():
    """分母 = 传进来的 Series（调用方必须先做上一步剔除）。"""
    amt = pd.Series({"a": 100.0, "b": 80.0, "c": 60.0, "d": 40.0, "e": 20.0})
    keep = set(cut_bottom_by_amount(amt, 0.20))          # 5 只剔 1 只 → 剔 e
    assert keep == {"a", "b", "c", "d"}
    # 若只传 4 只（已剔掉 e），则剔掉的是 d
    keep2 = set(cut_bottom_by_amount(amt[["a", "b", "c", "d"]], 0.25))
    assert keep2 == {"a", "b", "c"}


def test_pick_members_respects_exclude_and_size():
    mcap = pd.Series({"a": 5.0, "b": 4.0, "c": 3.0, "d": 2.0, "e": 1.0})
    amt = pd.Series({k: 10.0 for k in mcap.index})       # 成交额相同 → 不触发剔除
    got = pick_members(mcap, amt, PitRules(cut_bottom_pct=0.0), exclude={"a"}, size=2)
    assert got == ["b", "c"]


def test_build_snapshot_rule_order_exact():
    """手工构造 8 只票，逐条断言名单。

    构造（mcap 降序 A>B>C>D>E>F>G>H；**B 的成交额最低**，H 次低）：
        A(100,100) B(90,20) C(80,90) D(70,80) E(60,70) F(50,60) G(40,50) H(30,40)
    规则：exclude_top_mcap=2（剔市值前 2 = A,B）、cut_bottom_pct=20%、
          hs300=2 / csi500=3 / csi1000=4。
    """
    stats = _stats({"A": (100.0, 100.0), "B": (90.0, 20.0), "C": (80.0, 90.0),
                    "D": (70.0, 80.0), "E": (60.0, 70.0), "F": (50.0, 60.0),
                    "G": (40.0, 50.0), "H": (30.0, 40.0)})
    rules = PitRules(exclude_top_mcap=2, cut_bottom_pct=0.20,
                     hs300_size=2, csi500_size=3, csi1000_size=4,
                     min_days_in_window=1)
    snap = build_snapshot(stats, rules)

    # HS300：8 只里剔成交额后 20%（垫底的 B）→ 剩 7 只按市值取前 2 → A、C
    assert snap[HS300] == ["A", "C"], snap[HS300]

    # CSI500：剔 (HS300 ∪ 市值前2) = {A,C} ∪ {A,B} = {A,B,C}
    #         → 候选 {D,E,F,G,H}（5 只），**在这个 5 只集合内**剔成交额后 20%
    #         → 垫底的变成 H（rank 0.2，不 > 0.2）→ 剩 {D,E,F,G}
    #         → 市值前 3 → D、E、F
    #   ↑ 这一步正是"分母用剩余集合"：全局垫底的 B 已被剔除，于是轮到 H 被剔。
    assert snap[CSI500] == ["D", "E", "F"], snap[CSI500]

    # CSI1000：剔 (HS300 ∪ CSI500 ∪ 市值前2) = {A,B,C,D,E,F} → 候选 {G,H}（2 只）
    #          → 剔成交额后 20%：2 只里 H 的 rank 0.5 > 0.2 → 保留
    #          → 市值前 4（不足）→ G、H
    assert snap[CSI1000] == ["G", "H"], snap[CSI1000]

    # 三档互斥
    assert not (set(snap[HS300]) & set(snap[CSI500]))
    assert not (set(snap[CSI500]) & set(snap[CSI1000]))
    assert not (set(snap[HS300]) & set(snap[CSI1000]))
    # 高市值但成交额垫底的 B，被"市值前 2"这条规则挡住（不进 500/1000）
    assert "B" not in snap[CSI500] and "B" not in snap[CSI1000]


def test_snapshot_sizes_when_enough_candidates():
    """候选足够时三档都要选满（4000 只足够：1000 档要剔 800 + 前300 后再剔 20%）。"""
    n = 4000
    stats = _stats({f"{i:06d}": (float(n - i), 1e8 + i) for i in range(n)})
    rules = PitRules(min_days_in_window=1)
    snap = build_snapshot(stats, rules)
    assert len(snap[HS300]) == 300
    assert len(snap[CSI500]) == 500
    assert len(snap[CSI1000]) == 1000


def test_min_days_in_window_filters_new_and_suspended():
    stats = _stats({"A": (10.0, 10.0), "B": (9.0, 9.0)})
    stats.loc["B", "n_days"] = 5                       # B 数据太少（次新/长期停牌）
    snap = build_snapshot(stats, PitRules(min_days_in_window=120, hs300_size=5,
                                          csi500_size=5, csi1000_size=5))
    assert "B" not in snap[HS300] and "B" not in snap[CSI500] and "B" not in snap[CSI1000]
    assert "A" in snap[HS300]


def test_bj_excluded_by_default():
    stats = _stats({"600000": (100.0, 100.0), "920001": (99.0, 99.0),
                    "830001": (98.0, 98.0)})
    snap = build_snapshot(stats, PitRules(min_days_in_window=1, hs300_size=5))
    assert "920001" not in snap[HS300] and "830001" not in snap[HS300]
    assert "600000" in snap[HS300]
    # 关掉排除后就会入选
    snap2 = build_snapshot(stats, PitRules(min_days_in_window=1, hs300_size=5,
                                           exclude_bj=False))
    assert "920001" in snap2[HS300]


def test_empty_input_does_not_crash():
    snap = build_snapshot(_stats({}), PitRules(min_days_in_window=1))
    assert snap[HS300] == [] and snap[CSI500] == [] and snap[CSI1000] == []


# ============================================================
# 3. 无未来函数（最关键的断言）
# ============================================================
def test_no_future_leak_on_data_mutation():
    """**篡改截止日之后的数据，历史快照必须逐票不变。**

    这是"无未来函数"最有力的证据——不是看代码像不像，而是直接改未来数据看结果动不动。
    """
    days = _trading_days("2019-01-01", 700)
    syms = [f"6000{i:02d}" for i in range(20)]
    p = _make_db(syms, days)
    try:
        con = sqlite3.connect(str(p))
        rules = PitRules(window=250, min_days_in_window=1, min_list_days=0)
        cutoffs = [c for c in semi_annual_cutoffs([2020, 2021])
                   if c.isoformat() <= days[-1]]
        assert len(cutoffs) >= 2, "合成数据要覆盖至少两期"
        first_cut = cutoffs[0]

        snaps1 = build_snapshots(con, days[0], days[-1], rules, progress=False)
        # 把第一期截止日**之后**的数据全部改掉（价格×3、成交额×1000）
        con.execute("UPDATE full_daily SET float_mcap = float_mcap * 3, "
                    "amount = amount * 1000 WHERE date > ?", (first_cut.isoformat(),))
        con.commit()
        snaps2 = build_snapshots(con, days[0], days[-1], rules, progress=False)

        eff = review_effective_dates(cutoffs, days)
        first_eff = eff[first_cut]
        assert first_eff in snaps1 and first_eff in snaps2
        for code in (HS300, CSI500, CSI1000):
            assert snaps1[first_eff][code] == snaps2[first_eff][code], (
                f"{code} 在 {first_eff} 的快照被未来数据影响了 → 存在未来函数")
        con.close()
    finally:
        _rm(p)


def test_suspended_symbol_drops_out_of_window():
    """某票在窗口内数据不足（停牌/次新）→ 不该进那一期名单。"""
    days = _trading_days("2019-01-01", 700)
    syms = [f"6000{i:02d}" for i in range(10)]
    p = _make_db(syms, days)
    try:
        con = sqlite3.connect(str(p))
        rules = PitRules(window=250, min_days_in_window=120, min_list_days=0,
                         hs300_size=5, csi500_size=5, csi1000_size=5)
        # 把 600000 的数据砍到只剩极少几天（模拟长期停牌/次新）
        con.execute("DELETE FROM full_daily WHERE symbol='600000' AND date < ?",
                    (days[690],))
        con.commit()
        snaps = build_snapshots(con, days[0], days[-1], rules, progress=False)
        for snap in snaps.values():
            assert "600000" not in snap[HS300]
            assert "600000" not in snap[CSI500]
            assert "600000" not in snap[CSI1000]
        con.close()
    finally:
        _rm(p)


# ============================================================
# 4. 掩码传播
# ============================================================
def test_mask_uses_latest_snapshot_not_future():
    """掩码：t 取 `review_date <= t` 的最新快照，绝不提前切换。"""
    days = _trading_days("2020-01-01", 40)
    syms = [f"6000{i:02d}" for i in range(6)]
    p = _make_db(syms, days)
    try:
        con = sqlite3.connect(str(p))
        d1, d2 = date(2020, 1, 15), date(2020, 2, 3)
        save_snapshots(con, {d1: {CSI500: ["600000", "600001"]},
                             d2: {CSI500: ["600002", "600003"]}})
        idx = [d for d in days if d <= "2020-02-10"]
        m = build_mask(con, (CSI500,), idx, syms)
        pre = [d for d in m.index if d.date() < d1]        # 第一期生效之前：无快照
        before = [d for d in m.index if d1 <= d.date() < d2]
        after = [d for d in m.index if d.date() >= d2]
        assert pre and before and after, "测试数据要覆盖三个阶段"
        assert not m.loc[pre].to_numpy().any(), "生效日之前不该有任何成员"
        assert all(m.loc[d, "600000"] and not m.loc[d, "600002"] for d in before)
        # 第二期生效当天就切换，且绝不提前
        assert all(m.loc[d, "600002"] and not m.loc[d, "600000"] for d in after)
        con.close()
    finally:
        _rm(p)


def test_snapshots_roundtrip():
    p = _tmp_db()
    try:
        con = sqlite3.connect(str(p))
        snap = {date(2021, 6, 15): {HS300: ["600000"], CSI500: ["600001"],
                                    CSI1000: ["600002"]}}
        n = save_snapshots(con, snap)
        assert n == 3
        back = load_snapshots(con)
        assert back[date(2021, 6, 15)][CSI500] == ["600001"]
        con.close()
    finally:
        _rm(p)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    ok = 0
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
        ok += 1
    print(f"\n{ok}/{len(fns)} passed")
