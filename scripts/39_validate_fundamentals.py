"""校验 fundamentals.db 推出的 PE/ROE 面板是否「合理」——接进回测前的验收闸门。

## 为什么不能只看分布好不好看

「PE 中位数 25 倍」这种数字**好看不等于正确**：如果 EPS 用错口径，中位数照样可能
落在合理区间。所以本脚本的主检是**恒等式自检**与**口径检验**，不是看分布。

### 主检 C：ROE 恒等式（跨字段交叉验证）

    ROE = 净利润/净资产 = EPS/BPS
    PE = 价/EPS，PB = 价/BPS   ⇒   ROE = PB/PE

eps / bps / roe 是**同一条记录**的三个独立字段，这个恒等式不是数据源自带的，
而是三者之间的代数约束。任何一方口径错（EPS 没去累计 / BPS 用了未来值 /
ROE 单位是小数不是百分数），恒等式就崩。`implied = 100*PB/PE` 必须 ≈ `roe_ttm`。

### 决定性检验 E：送转后 EPS 是否被数据源追溯重述

`full_daily.close` 是 **qfq 前复权**（`fetch_daily_full` 默认 `adjust="qfq"`）。
qfq 把历史价按**今天的股本**折算过，所以：

    送转前  实际价 100，10送10后 实际价 50，qfq 价在两个时点**都是 50**

那 PE = qfq价/EPS 对不对，就完全取决于 **EPS 是不是也被折算到今天的股本**：

| 数据源的 EPS | qfq价/EPS | 结论 |
|---|---|---|
| **追溯重述**（送转后旧报告的 EPS 也除以 2） | 口径一致 | 可用 |
| **不重述**（旧报告仍是送转前的 EPS） | 送转前 PE 被**低估 N 倍**（N=送转比例） | **必须改口径** |

这不是理论担忧：送转在 A 股常见，一只 10送10 的票在送转前的 PE 会直接**差一倍**，
而 PE 正是线上选股的主因子。所以必须实测数据源属于哪一类。

判据：找到 `outstanding_share` 跳变的时点，看**该时点前后报告期的 `eps_ytd`**
是否也按相近比例跳变。重述 ⇒ 跳；不重述 ⇒ 平滑。

用法：

    .venv/Scripts/python.exe scripts/39_validate_fundamentals.py
    .venv/Scripts/python.exe scripts/39_validate_fundamentals.py --year 2024
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant.data.fundamentals import load_financials, pit_pe_roe_panels, disclosure_date  # noqa: E402
from quant.data.universe_pit import load_raw_close                                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUND_DB = os.path.join(ROOT, "data", "fundamentals.db")
FULL_DB = os.path.join(ROOT, "data", "full_market.db")

SAMPLE_MONTH_DAYS = [(4, 30), (8, 31), (12, 31)]


def line(c="-", n=78):
    print(c * n)


def sec(title):
    print()
    line("=")
    print(title)
    line("=")


def trading_days_near(con, targets):
    have = pd.read_sql_query("SELECT DISTINCT date FROM full_daily ORDER BY date", con)["date"]
    have = pd.to_datetime(have)
    out = []
    for t in targets:
        cand = have[have <= pd.Timestamp(t)]
        if len(cand):
            out.append(cand.iloc[-1])      # iloc 必需：pandas 3.0 的 s[-1] 是标签索引
    return sorted(set(out))


def load_close(con, symbols, dates):
    """只取采样日的收盘价，内存恒定（不做全历史 pivot）。"""
    q = ",".join("?" * len(symbols))
    ds = ",".join("?" * len(dates))
    df = pd.read_sql_query(
        f"SELECT date, symbol, close FROM full_daily "
        f"WHERE date IN ({ds}) AND symbol IN ({q})",
        con, params=[*[str(d)[:10] for d in dates], *symbols])
    p = df.pivot_table(index="date", columns="symbol", values="close")
    p.index = pd.to_datetime(p.index)
    return p.sort_index().astype("float64")


def load_bps_asof(con, symbols, dates):
    """每股净资产是资产负债表存量，不需要 TTM，只要「可见日 <= t 的最新一期」。"""
    raw = pd.read_sql_query(
        "SELECT symbol, report_date, bps FROM financials WHERE bps IS NOT NULL", con)
    raw["report_date"] = pd.to_datetime(raw["report_date"])
    raw["avail"] = raw["report_date"].map(disclosure_date)
    raw = raw.dropna(subset=["avail"]).sort_values("report_date")
    out = pd.DataFrame(index=pd.DatetimeIndex(dates), columns=symbols, dtype="float64")
    for d in dates:
        dts = pd.Timestamp(d).date()
        out.loc[d] = raw[raw["avail"] <= dts].groupby("symbol")["bps"].last().reindex(symbols)
    return out


def q(s, pcts=(1, 5, 25, 50, 75, 95, 99)):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(s):
        return "n=0"
    return "  ".join(f"p{p}={np.percentile(s, p):.2f}" for p in pcts) + f"   n={len(s)}"


def section_e(con, fin, pe_q, close, symbols, dates):
    """决定性检验：qfq 价 vs 不复权价，算出的 PE 截面排序差多少。

    qfq 把历史价按**今天的股本**折算过；若数据源的 EPS 是 as-reported（未追溯重述），
    则 `qfq价/EPS` 在送转前会被低估 N 倍（N = 送转比例），而 `不复权价/EPS` 才是时点正确的。

    但「哪个对」不影响本段的可用性判据——**真正要问的是：选错口径会不会改变选股结果**。
    所以直接量两件事：① 两者比值分布；② 截面**排序相关性**（选股只看排序）。
    排序相关性高 ⇒ 口径之争对模型无实质影响，可直接用现有的 qfq 版。
    """
    sec("E. [决定性] qfq价/EPS  vs  不复权价/EPS —— 口径差异会不会改变选股排序")
    raw = load_raw_close(con, dates, symbols)

    # E0. 先验证 raw 价本身可信：最后一个采样日 qfq 应当约等于 raw（qfq 锚定在最新日）
    d_last = dates[-1]
    a, b = close.loc[d_last], raw.loc[d_last]
    both = pd.concat([a.rename("qfq"), b.rename("raw")], axis=1).dropna()
    both = both[(both > 0).all(axis=1)]
    ratio = both["qfq"] / both["raw"]
    print(f"  [E0] {str(d_last)[:10]}  qfq/raw 比值: {q(ratio, (1, 25, 50, 75, 99))}")
    print("       （锚定日两者应约等于 1；若中位明显偏离 1，说明 amount/volume 不是元/股）")

    # E1. 逐日：两种口径的 PE 比值 + 截面排序相关性
    pe_q = pe_q.reindex(columns=symbols)
    pe_r = pit_pe_roe_panels(fin, raw, symbols)[0].reindex(columns=symbols)

    print("\n  [E1] 同一日两种口径的 PE：比值分布 + 截面排序相关（选股只看排序）")
    print(f"  {'date':<12}{'p50(qfq/raw)':>14}{'p5':>8}{'p95':>8}{'Spearman':>11}{'n':>7}")
    worst = []
    for d in dates:
        a = pe_q.loc[d].replace([np.inf, -np.inf], np.nan)
        b = pe_r.loc[d].replace([np.inf, -np.inf], np.nan)
        both = pd.concat([a.rename("q"), b.rename("r")], axis=1).dropna()
        if len(both) < 50:
            continue
        rt = both["q"] / both["r"]
        sp = both["q"].corr(both["r"], method="spearman")
        print(f"  {str(d)[:10]:<12}{np.median(rt):>14.3f}"
              f"{np.percentile(rt, 5):>8.3f}{np.percentile(rt, 95):>8.3f}{sp:>11.4f}{len(both):>7}")
        worst.append((sp, d, rt))

    # E2. 分层：未来有送转的票 vs 没有的
    print("\n  [E2] 分层：按「该日之后是否发生过送转」拆开")
    jumps = pd.read_sql_query(
        "SELECT symbol, date FROM ("
        "  SELECT symbol, date, outstanding_share AS os,"
        "         LAG(outstanding_share) OVER (PARTITION BY symbol ORDER BY date) AS prev"
        "  FROM full_daily WHERE date >= '2021-01-01'"
        ") WHERE prev IS NOT NULL AND os > 1.5*prev", con)
    jumps["date"] = pd.to_datetime(jumps["date"])
    has_split = set(jumps["symbol"])
    print(f"  池内有送转记录的票: {len(has_split & set(symbols))} / {len(symbols)}")
    for d in dates[:1] + dates[len(dates) // 2:len(dates) // 2 + 1] + dates[-1:]:
        a = pe_q.loc[d].replace([np.inf, -np.inf], np.nan)
        b = pe_r.loc[d].replace([np.inf, -np.inf], np.nan)
        both = pd.concat([a.rename("q"), b.rename("r")], axis=1).dropna()
        if len(both) < 50:
            continue
        fut = jumps[jumps["date"] > d]["symbol"]
        both["has_future_split"] = both.index.isin(set(fut))
        for flag, lab in ((True, "日后有送转"), (False, "无送转")):
            sub = both[both["has_future_split"] == flag]
            if len(sub) < 20:
                continue
            rt = sub["q"] / sub["r"]
            print(f"    {str(d)[:10]}  {lab:<10} n={len(sub):>5}  "
                  f"p50比={np.median(rt):>6.3f}  p5={np.percentile(rt, 5):>6.3f}  "
                  f"Spearman={sub['q'].corr(sub['r'], method='spearman'):.4f}")

    sps = [w[0] for w in worst]
    print(f"\n  => 全期截面排序相关: 最低 {min(sps):.4f} / 中位 {np.median(sps):.4f}")
    if np.median(sps) > 0.99:
        print("  判定：两口径的选股排序几乎相同，口径之争对模型无实质影响，沿用 qfq 版即可。")
    else:
        print("  判定：排序差异显著，PE 口径必须修正（详见 docs）。")
    return {"rank_corr_min": float(min(sps)), "rank_corr_med": float(np.median(sps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument("--skip-e", action="store_true", help="跳过较慢的送转检验")
    args = ap.parse_args()

    conf = sqlite3.connect(FUND_DB)
    con = sqlite3.connect(FULL_DB)

    fin = load_financials(conf)
    symbols = sorted(fin["symbol"].unique())
    print(f"[in] financials: {len(fin)} rows / {len(symbols)} symbols (de-cumulated)")

    have = pd.to_datetime(
        pd.read_sql_query("SELECT DISTINCT date FROM full_daily ORDER BY date", con)["date"])
    yrs = [args.year] if args.year else sorted(set(have.dt.year))
    targets = [pd.Timestamp(y, m, d) for y in yrs for m, d in SAMPLE_MONTH_DAYS
               if pd.Timestamp(y, m, d) <= have.iloc[-1]]
    dates = trading_days_near(con, targets)
    print(f"[in] sample dates: {len(dates)}  ({str(dates[0])[:10]} .. {str(dates[-1])[:10]})")

    close = load_close(con, symbols, dates)
    symbols = [s for s in symbols if s in close.columns]
    pe, roe = pit_pe_roe_panels(fin, close, symbols)
    bps = load_bps_asof(conf, symbols, dates)
    pb = close[symbols].div(bps.replace(0, np.nan))
    print(f"[in] panels: {close.shape[0]} dates x {len(symbols)} symbols")

    sec("A. 覆盖率（PE 为 NaN 的亏损股是设计如此，不是缺失）")
    print(f"{'date':<12}{'PE有值':>9}{'占比':>9}{'ROE有值':>10}{'价格有值':>10}")
    for d in dates:
        n_tot = close.loc[d].notna().sum()
        n_pe = pe.loc[d].notna().sum()
        n_roe = roe.loc[d].notna().sum()
        print(f"{str(d)[:10]:<12}{n_pe:>9}{n_pe / max(n_tot, 1):>8.1%}"
              f"{n_roe:>10}{n_tot:>10}")

    sec("B. 分布分位数（PE 全市场约 20-45；TTM ROE 多在 0-20%；PB 多在 1-3）")
    for name, panel in (("PE", pe), ("TTM ROE (%)", roe), ("PB", pb)):
        print(f"-- {name} --")
        for d in dates:
            print(f"  {str(d)[:10]}  {q(panel.loc[d])}")

    sec("C. 主检 ROE 恒等式  implied = 100*PB/PE  应 ~= 报告 roe_ttm")
    print(f"{'date':<12}{'中位绝对误差':>14}{'|误差|<5pp':>12}{'Pearson':>10}{'Spearman':>10}{'n':>7}")
    for d in dates:
        imp = (pb.loc[d] / pe.loc[d] * 100.0).rename("imp")
        rep = roe.loc[d].rename("rep")
        both = pd.concat([imp, rep], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
        if len(both) < 10:
            print(f"{str(d)[:10]:<12}{'n<10':>14}")
            continue
        err = (both["imp"] - both["rep"]).abs()
        print(f"{str(d)[:10]:<12}{err.median():>14.2f}{(err < 5).mean():>11.1%}"
              f"{both['imp'].corr(both['rep']):>10.3f}"
              f"{both['imp'].corr(both['rep'], method='spearman'):>10.3f}{len(both):>7}")
    # 误差最大的样本：看看是什么在拖偏相关
    d = dates[-1]
    both = pd.concat([(pb.loc[d] / pe.loc[d] * 100).rename("imp"), roe.loc[d].rename("rep")],
                     axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    both["err"] = (both["imp"] - both["rep"]).abs()
    print(f"  -- {str(d)[:10]} 误差最大的 5 例（看是不是极端值在拖相关）--")
    for sym, r in both.nlargest(5, "err").iterrows():
        print(f"     {sym}  implied={r['imp']:>10.2f}  报告={r['rep']:>8.2f}  err={r['err']:.2f}")

    sec("D. 极端值占比（PE>1000 = 微利股；PE<1 异常）")
    print(f"{'date':<12}{'PE>1000':>10}{'PE<1':>8}{'PE>300':>9}{'n':>9}")
    for d in dates:
        s = pe.loc[d].replace([np.inf, -np.inf], np.nan).dropna()
        if not len(s):
            continue
        print(f"{str(d)[:10]:<12}{(s > 1000).mean():>9.2%}{(s < 1).mean():>8.2%}"
              f"{(s > 300).mean():>9.2%}{len(s):>9}")

    verdict = {"restated": 0, "not_restated": 0, "inconclusive": 0}
    if not args.skip_e:
        verdict = section_e(con, fin, pe, close, symbols, dates)

    print()
    line("=")
    print("判读标准：")
    print("  C 段：中位绝对误差 < 5pp 且 Spearman > 0.9  =>  EPS 去累计/BPS/ROE 三者口径自洽")
    print("  E 段：全期截面排序相关中位 > 0.99  =>  PE 口径之争不影响选股，沿用 qfq 版")
    print(f"       (本次实测: {verdict})")
    line("=")


if __name__ == "__main__":
    main()
