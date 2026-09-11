"""第 33 步：流动性门槛专项 —— 结论：**现有股票池有严重幸存者偏差，这个问题答不了**。

起因（scripts/32 的副产品）：同一套因子，只因流动性口径不同，回测结果从 +16.7% 跳到 -22.8%。
原计划查「是门槛设错了，还是要把不可交易性显式建模进打分」。查下来发现**根因在数据**：

    559 池 = 「现池40 ∪ **当前**中证500(260) ∪ **当前**中证1000(300)」，用今天的成分名单回填历史。
    ⇒ 被指数剔除的、退市的（多数是跌下去的）统统不在样本里。
    ⇒ 更糟的是：**偏差不是均匀的，它正好落在「流动性」这根轴上** ——
       2021 年冷门、今天却还在中证500/1000 里的票，必然是涨了很多倍的票。

本脚本给出证据链（A 偏差量级 → B 偏差挂在流动性上 → C 冲击成本 → D 门槛扫描作废）：

    A. 池内流通市值加权 vs 真实中证500/1000 指数（同窗口）
    B. 按 2021-06 的流动性分组，看之后的买入持有收益 + 收益的尾部集中度
    C. Amihud 冲击成本（按我们真实账户规模，回答「门槛是不是为了防冲击」）
    D. 门槛扫描（保留数字，但**已作废**，不作为决策依据）

用法：
    python scripts/33_liquidity_gate_study.py
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    stream=sys.stdout)
logger = logging.getLogger(__name__)

from quant.config import load_config                                # noqa: E402
from quant.realtime.indices import fetch_index_daily                # noqa: E402

START = "2021-07-01"


def load(db: Path):
    """返回 close / amount / turnover 三个 date x symbol 宽表。"""
    con = sqlite3.connect(str(db))
    bars = pd.read_sql_query(
        "SELECT symbol, date, close, volume, amount FROM large_daily", con)
    tur = pd.read_sql_query("SELECT symbol, date, turnover FROM large_turnover", con)
    con.close()

    def piv(df, col):
        p = df.pivot_table(index="date", columns="symbol", values=col).sort_index()
        p.index = pd.to_datetime(p.index)
        return p

    return piv(bars, "close"), piv(bars, "amount"), piv(tur, "turnover")


def index_close(sym: str, dates: pd.DatetimeIndex) -> pd.Series:
    df = fetch_index_daily(sym)
    s = df.set_index("date")["close"].sort_index()
    s.index = pd.to_datetime(s.index)
    return s.reindex(dates).ffill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=START)
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--order-yuan", type=float, default=10_000.0,
                    help="单笔委托金额（模拟盘 10 万 / 12 仓 ≈ 1 万；实盘 3000 更小）")
    args = ap.parse_args()

    cfg = load_config()
    close, amount, tv = load(Path(cfg.resolve("data")) / "large_pool.db")
    start = pd.Timestamp(args.start)
    cl, am, tv = close.loc[start:], amount.loc[start:], tv.loc[start:]
    dates = cl.index
    logger.info("窗口 %s ~ %s（%d 个交易日，%d 只）",
                dates[0].date(), dates[-1].date(), len(dates), cl.shape[1])

    # ================= A. 幸存者偏差量级 =================
    print("\n==================== A. 幸存者偏差量级 ====================")
    ret1 = cl.pct_change()
    # 真实流通市值 = 成交额 / 换手率 = 价格 x 流通股本（只用当日量，规避复权口径问题）
    mcap = (am / tv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)

    def capw_return(msk=None) -> float:
        w = mcap.where(msk) if msk is not None else mcap
        w = w.div(w.sum(axis=1), axis=0)
        return float((1 + (w * ret1).sum(axis=1).fillna(0)).prod() - 1)

    idx500, idx1000 = index_close("sh000905", dates), index_close("sh000852", dates)
    r500 = float(idx500.iloc[-1] / idx500.iloc[0] - 1)
    r1000 = float(idx1000.iloc[-1] / idx1000.iloc[0] - 1)
    pool_all, pool_liq = capw_return(), capw_return(am >= 1e8)
    blend = (r500 + r1000) / 2
    print("口径：流通市值加权、每日再平衡（近似指数编制），与真实指数同窗口")
    print(f"  池内 559 只（流通市值加权）  : {pool_all * 100:+7.1f}%")
    print(f"  池内仅「当日额>=1亿」        : {pool_liq * 100:+7.1f}%")
    print(f"  真实 中证500 指数            : {r500 * 100:+7.1f}%")
    print(f"  真实 中证1000 指数           : {r1000 * 100:+7.1f}%")
    print(f"  两者均值（池子~两者的混合）  : {blend * 100:+7.1f}%")
    print(f"  -> 超额（池子 - 混合）       : {(pool_all - blend) * 100:+.1f} pp"
          "   <= 这就是偏差量级，不可能是真 alpha")

    # ================= B. 偏差挂在流动性上 =================
    print("\n==================== B. 偏差正好挂在「流动性」这根轴上 ====================")
    end = cl.index[-1]
    base_amt = am.loc[:start].tail(20).median()          # 起点前 20 日中位成交额
    hold = (cl.loc[end] / cl.loc[start] - 1)
    d = pd.DataFrame({"ret": hold, "amt0": base_amt}).dropna()
    d["grp"] = np.where(d["amt0"] >= 1e8, "起点已活跃(>=1亿)", "起点冷门(<1亿)")
    print("按 2021-06 的流动性分组，看之后 5.2 年的买入持有收益：")
    g = d.groupby("grp")["ret"]
    for k in ["起点冷门(<1亿)", "起点已活跃(>=1亿)"]:
        s = g.get_group(k)
        print(f"  {k:<18} n={len(s):>3d}  平均 {s.mean():+.3f}  中位 {s.median():+.3f}  "
              f"翻倍以上 {(s > 1).mean() * 100:>4.1f}%")
    print(f"  {'全池':<18} n={len(d):>3d}  平均 {d['ret'].mean():+.3f}  "
          f"中位 {d['ret'].median():+.3f}")
    print(f"\n  [对照] 真实指数同期：中证500 {r500 * 100:+.1f}%、中证1000 {r1000 * 100:+.1f}%")
    print("  => 池子的**中位数**几乎等于指数，但**平均值/加权**高出一大截，说明是右偏长尾：")
    for k in ["起点冷门(<1亿)", "起点已活跃(>=1亿)"]:
        r = g.get_group(k).sort_values(ascending=False)
        top = ", ".join(f"{i}({v * 100:+.0f}%)" for i, v in r.head(3).items())
        print(f"     {k}: Top5 占总收益 {r.head(5).sum() / r.sum() * 100:.0f}%，前 3 名 {top}")
    print("  => 冷门组在 2021 年是小票、如今仍在中证500/1000，等于**被未来涨幅选进来**的。")

    # ================= C. 冲击成本 =================
    print("\n==================== C. 冲击成本：门槛不是为了防冲击 ====================")
    print(f"单笔委托 {args.order_yuan:,.0f}（模拟盘 10 万/12 仓 ≈ 8.3k，实盘 3k 更小）")
    print("Amihud 冲击（单边比例） ~ mean(|日收益| / 日成交额) x 委托金额")
    liq_rank = am.rank(axis=1, pct=True)
    buckets = [("Q1 最冷门 20%", liq_rank <= 0.2), ("Q2", (liq_rank > 0.2) & (liq_rank <= 0.4)),
               ("Q3", (liq_rank > 0.4) & (liq_rank <= 0.6)),
               ("Q4", (liq_rank > 0.6) & (liq_rank <= 0.8)), ("Q5 最活跃 20%", liq_rank > 0.8)]
    print(f"  {'档':<14}{'中位成交额(亿)':>14}{'单边冲击(bp)':>14}{'往返(bp)':>10}")
    for name, msk in buckets:
        med = float(am.where(msk).median(axis=1).median())
        ill = float((ret1.abs() / am.replace(0, np.nan)).where(msk).mean(axis=1).median())
        imp = ill * args.order_yuan
        print(f"  {name:<14}{med / 1e8:>14.2f}{imp * 1e4:>14.3f}{imp * 2e4:>10.3f}")
    print("  => 我们的账户规模下冲击成本是**零点几个 bp**，比最低佣金(5)小两个数量级。")
    print("     流动性门槛的真实作用不是防冲击，而是**防买卖价差 + 保证能成交**"
          "（价差历史数据我们没有）。")

    # ================= D. 门槛扫描（作废） =================
    print("\n==================== D. 门槛扫描（数字保留，但**已作废**）====================")
    print("原因：B 已证明偏差与流动性直接相关 => 这个扫描是在比较「偏差大的子样本」")
    print("      和「偏差小的子样本」，不是比较策略。**不作为决策依据。**")
    f = {"mom20": cl.pct_change(20), "rev60": cl.shift(60) / cl - 1,
         "trd": cl / cl.rolling(20).mean() - 1, "-vol20": -ret1.rolling(20).std()}

    def z(p, msk):
        p = p.where(msk)
        return p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0)

    def bt(score: pd.DataFrame, topn: int, rebal: int = 5, cost: float = 0.0016) -> dict:
        d_ = score.index
        nav, curve = 1.0, []
        for i in range(0, len(d_) - rebal, rebal):
            dd, d2 = d_[i], d_[i + rebal]
            s = score.loc[dd].dropna()
            s = s[np.isfinite(s)]
            if len(s) < topn:
                continue
            picks = s.nlargest(topn).index
            p0, p1 = cl.loc[dd, picks], cl.loc[d2, picks]
            if (p0 <= 0).any():
                continue
            seg = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
            if not len(seg):
                continue
            nav *= (1 + float(seg.mean()) - cost)
            curve.append((d2, nav))
        if not curve:
            return {}
        ser = pd.Series(dict(curve))
        yrs = (ser.index[-1] - ser.index[0]).days / 365.25
        tot = ser.iloc[-1] - 1
        per = ser.pct_change().dropna()
        return {"total": round(tot * 100, 1),
                "annual": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else 0.0,
                "maxdd": round(float((ser / ser.cummax() - 1).min()) * 100, 1),
                "sharpe": round(float(per.mean() / per.std() * np.sqrt(252 / rebal)), 2)
                if per.std() else 0.0}

    for thr in (0.0, 2e7, 5e7, 1e8, 2e8, 5e8):
        msk = (am >= thr) if thr > 0 else am.notna()
        score = (0.25 * z(f["mom20"], msk) + 0.15 * z(f["trd"], msk)
                 + 0.30 * z(f["-vol20"], msk) + 0.30 * z(f["rev60"], msk))
        m = bt(score, args.topn)
        tag = "无门槛" if thr == 0 else f">= {thr / 1e8:.1f} 亿"
        if m:
            print(f"  {tag:<10} 可交易中位 {int(msk.sum(axis=1).median()):>3d} 只 | "
                  f"总收益 {m['total']:>7.1f}%  年化 {m['annual']:>6.2f}%  "
                  f"回撤 {m['maxdd']:>6.1f}%  Sharpe {m['sharpe']:>5.2f}")

    print("\n==================== 结论 ====================")
    print("1) 门槛问题**用现有数据答不了** —— 必须先修股票池（时点成分股 / 含退市股）。")
    print("2) 偏差不只影响本脚本：**此前所有基于这 559 池的回测**（风控/止损/入场闸门/换手率）")
    print("   的绝对水平都不可信；涉及流动性暴露的相对比较也受污染。")
    print("3) 冲击成本不是设门槛的理由（我们的账户太小）；真实理由是价差与成交概率。")


if __name__ == "__main__":
    main()
