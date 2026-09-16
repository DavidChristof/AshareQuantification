"""第 37 步：把此前的**风控回测**放到 PIT 无偏宇宙上重跑一遍（老池 vs 无偏池对照）。

## 为什么必须重跑

`docs/2026-09-11-risk-control.md`（回撤熔断 / 弱势减仓）与同日的趋势闸门、入场择时闸门，
**全部是在有幸存者偏差的 559 池上测的**。而 `docs/2026-09-11-pit-universe.md` 已证明：
换手率的结论在无偏宇宙上**直接反转**（老池「无改善」→ 无偏池「+61pp」）。
=> 风控结论的绝对水平同样可疑，必须重测。

## 重跑什么

同一套组合、同一套阈值，只换宇宙：

    策略   = 现有 4 因子综合分（mom20 / trd / -vol20 / rev60）→ top12 等权
    调仓   = 每 5 个交易日，含换手费 0.16%/期
    A 基准（无风控）
    B + 账户回撤熔断 8%/4%（带滞回；触发则仓位上限降到 50% 且停开新仓）
    C + 大盘趋势闸门（指数 < MA20 → 当日不开新仓，保留持仓）
    D + 弱势减半（大盘当日 ≤ -1% → 仓位腰斩）

## 口径对齐（2026-09-16，第二层）

原实现有两个**实盘无法落地**的口径错误（`docs/2026-09-14-trend-gate-semantics.md` 第二层）：

1. `ma_ok = idx >= idx.rolling(20).mean()` —— 用的是 **今收 vs 今日 MA20**。
   但闸门在 09:31 执行，那时**根本不知道当日收盘**。线上实际用的是
   **昨收 vs 昨日 MA20**（`market_trend.completed_closes`，见 docs）。
   两种口径实测 **12.9% 的交易日结论相反** —— 也就是说此前这个脚本
   验证的是**另一个信号**。
2. `ir = idx.pct_change()` —— 弱势判断用的是**当日收盘涨跌幅**，同样是未来函数。
   线上 `_market_weakness()` 读的是**实时**涨跌幅（≈今开 vs 昨收）。
   这里改用 **今开 / 昨收 - 1** 作代理。

改后两者都只用 **D-1 及之前**（闸门：`c.shift(1)`；弱势：open(D) 相对 close(D-1)），
与 09:31 实盘可执行的信息集一致。

## 大盘代理可换（`--index`）

原实现把 `sh000300` 写死，而线上闸门的指数是**配置项**
（`portfolio_risk.market_trend_gate.index`）。两边不一致会让「线上改配置、
回测不跟着变」。现加 `--index`，默认仍是 `sh000300`（保持原行为）。

用法：
    python scripts/37_risk_rerun_pit.py
    python scripts/37_risk_rerun_pit.py --index sh000001          # 上证
    python scripts/37_risk_rerun_pit.py --index sh000905          # 中证500
    python scripts/37_risk_rerun_pit.py --start 2021-07-01 --topn 12

输出：`results/risk_rerun_pit_<index>.json`（含 index 后缀，三次跑不会互相覆盖）
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))     # noqa: E402

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)

from quant.config import load_config                                # noqa: E402
from quant.data.universe_pit import CSI1000, CSI500, build_mask     # noqa: E402
from quant.realtime.indices import fetch_index_daily                # noqa: E402

REBAL = 5           # 调仓周期（交易日）
COST = 0.0016       # 每期换手成本（与 docs 口径一致）
TRIP, RELEASE = 0.08, 0.04      # 回撤熔断：触发 / 解除
WEAK_THRESHOLD = -0.01          # 弱势定义：大盘当日 ≤ -1%
BRAKE_POS = 0.5                 # 熔断期间仓位上限
WINDOW = 60                     # 回撤窗口（净值点数，与线上 drawdown.py 一致）


# ============================================================
# 载入
# ============================================================
def _pivot(df, col, fdtype="float64"):
    p = df.pivot_table(index="date", columns="symbol", values=col).sort_index()
    p.index = pd.to_datetime(p.index)
    return p.astype(fdtype)


def load_large(cfg, start: str):
    """老 559 池（有幸存者偏差）。"""
    con = sqlite3.connect(str(Path(cfg.resolve("data")) / "large_pool.db"))
    bars = pd.read_sql_query(
        "SELECT symbol,date,close,volume,amount FROM large_daily WHERE date>=?",
        con, params=[start])
    con.close()
    close = _pivot(bars, "close")
    amount = _pivot(bars, "amount")
    return close, amount, None


def load_pit(cfg, start: str):
    """PIT 无偏宇宙：掩码之外置 NaN，后面所有排序自然只在成员内。"""
    con = sqlite3.connect(str(Path(cfg.resolve("data")) / "full_market.db"))
    dates = pd.DatetimeIndex([r[0] for r in con.execute(
        "SELECT DISTINCT date FROM full_daily WHERE date>=? ORDER BY date", (start,))])
    mask = build_mask(con, (CSI500, CSI1000), dates)
    syms = list(mask.columns)
    ph = ",".join("?" * len(syms))
    df = pd.read_sql_query(
        f"SELECT symbol,date,close,amount FROM full_daily "
        f"WHERE date>=? AND symbol IN ({ph})", con, params=[start, *syms])
    con.close()
    close = _pivot(df, "close", "float32")
    amount = _pivot(df, "amount", "float32")
    m = mask.reindex(close.index).fillna(False).astype(bool)
    for p in (close, amount):
        p[~m.reindex(columns=p.columns, fill_value=False)] = np.nan
    return close, amount, None


# ============================================================
# 打分
# ============================================================
def _z(p: pd.DataFrame) -> pd.DataFrame:
    return p.sub(p.mean(axis=1), axis=0).div(p.std(axis=1).replace(0, np.nan), axis=0)


def build_score(close: pd.DataFrame, amount: pd.DataFrame) -> pd.DataFrame:
    """现有 4 因子综合分（与 scripts/33 口径一致，便于跨宇宙对照）。"""
    ret1 = close.pct_change(fill_method=None)
    return (0.25 * _z(close.pct_change(20))
            + 0.15 * _z(close / close.rolling(20).mean() - 1)
            + 0.30 * _z(-ret1.rolling(20).std())
            + 0.30 * _z(close.shift(60) / close - 1))


# ============================================================
# 回测（带风控开关）
# ============================================================
def run(score: pd.DataFrame, close: pd.DataFrame, index_ret: pd.Series,
        index_ma_ok: pd.Series, topn: int, brake: bool, gate: bool, weak: bool):
    """每 REBAL 日调仓；风控在**新买入**与**仓位**上生效。

    index_ret : 大盘当日收益（弱势判断用），index_ma_ok: 大盘是否在 MA20 上方。

    Returns:
        (metrics, per_period) —— per_period 是**每期收益**的 Series（index = 期末日）。
        返回它才能做 B vs C 的**配对**显著性检验：同一批调仓期、只差不加闸门，
        比总收益高低的「看图说话」可靠得多（3 个近 0.96 相关的指数能给出 11pp 的差，
        说明单看总收益是分不清信号和噪声的）。
    """

    dates = score.index
    nav, curve, cur = 1.0, [], []
    navs: list[float] = [1.0]
    tripped = False
    for i in range(0, len(dates) - REBAL, REBAL):
        d, d2 = dates[i], dates[i + REBAL]
        s = score.loc[d].dropna()
        s = s[np.isfinite(s)]
        if len(s) < topn:
            continue
        picks = list(s.nlargest(topn).index)

        # --- 账户回撤熔断（带滞回）---
        # 与线上 `quant/risk/drawdown.py` 口径一致：相对**近 WINDOW 个净值点**的峰值，
        # 不是历史最高（用历史最高会显著低估触发频率 → 高估熔断效果）。
        peak = max(navs[-WINDOW:]) if brake else max(navs)
        dd = nav / peak - 1
        if brake:
            if not tripped and dd <= -TRIP:
                tripped = True
            elif tripped and dd > -RELEASE:
                tripped = False
        # --- 大盘趋势闸门 ---
        # 注意：不能写 `... is False` —— Series.get 返回的是 numpy.bool_，
        # `numpy.bool_(False) is False` 恒为 False，闸门会静默失效（踩过一次）。
        below = not bool(index_ma_ok.get(d, True))

        # --- 风控作用：熔断/闸门 → 停开新仓；弱势 → 仓位腰斩 ---
        if (brake and tripped) or (gate and below):
            picks = [p for p in picks if p in cur]      # 只留已持有，不开新仓
        exposure = BRAKE_POS if (brake and tripped) else 1.0
        if weak and float(index_ret.get(d, 0.0)) <= WEAK_THRESHOLD:
            exposure *= 0.5
        if not picks:
            curve.append((d2, nav))
            navs.append(nav)
            continue

        p0, p1 = close.loc[d, picks], close.loc[d2, picks]
        seg = (p1 / p0 - 1).replace([np.inf, -np.inf], np.nan).dropna()
        if not len(seg):
            continue
        r = float(seg.mean()) * exposure - COST
        nav *= (1 + r)
        cur = picks
        curve.append((d2, nav))
        navs.append(nav)

    if not curve:
        return {}, pd.Series(dtype=float)
    ser = pd.Series(dict(curve))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    tot = ser.iloc[-1] - 1
    per = ser.pct_change().dropna()
    metrics = {"total": round(tot * 100, 1),
               "annual": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else 0.0,
               "maxdd": round(float((ser / ser.cummax() - 1).min()) * 100, 1),
               "sharpe": round(float(per.mean() / per.std() * np.sqrt(252 / REBAL)), 2)
               if per.std() else 0.0}
    return metrics, per


def _paired(pers: dict) -> None:
    """配对检验：C vs B（闸门）、D vs C（弱势减半）。

    同一批调仓日、只差一个风控开关 => 逐期收益之差做单样本 t。
    **比单看总收益可靠得多**：本脚本实测 3 个日收益相关 0.96 的指数，
    能把同一条「闸门」的总收益给出 11pp 的差距 —— 单看总收益分不清信号和噪声。
    """
    for hi, lo, tag in (("C +熔断+趋势闸门", "B +回撤熔断 8%/4%", "闸门 C-B"),
                        ("D +熔断+闸门+弱势减半", "C +熔断+趋势闸门", "弱势 D-C")):
        a, b = pers.get(hi), pers.get(lo)
        if a is None or b is None or a.empty or b.empty:
            continue
        j = pd.concat([a.rename("hi"), b.rename("lo")], axis=1).dropna()
        if len(j) < 3:
            continue
        d = j["hi"] - j["lo"]
        se = float(d.std(ddof=1)) / np.sqrt(len(d))
        t = float(d.mean()) / se if se else 0.0
        print(f"    paired[{tag}] n={len(j)} periods  mean diff {d.mean() * 100:+.3f}%/period"
              f"  t = {t:+.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--topn", type=int, default=12)
    ap.add_argument("--index", default="sh000300",
                    help="大盘代理：sh000300 沪深300 / sh000001 上证 / sh000905 中证500")
    args = ap.parse_args()

    cfg = load_config()
    idf = fetch_index_daily(args.index).set_index("date").sort_index()
    idf.index = pd.to_datetime(idf.index)
    idx_close, idx_open = idf["close"], idf["open"]

    setups = [("A 基准（无风控）", False, False, False),
              ("B +回撤熔断 8%/4%", True, False, False),
              ("C +熔断+趋势闸门", True, True, False),
              ("D +熔断+闸门+弱势减半", True, True, True)]

    out = {}
    for name, loader in (("large", load_large), ("pit", load_pit)):
        close, amount, _ = loader(cfg, args.start)
        dates = close.index
        c = idx_close.reindex(dates)
        ma20 = c.rolling(20).mean()
        # 口径与线上一致：09:31 只能拿到 **D-1 及之前**已收盘的日线。
        # 写成 `c >= ma20` 会变成「今收 vs 今日 MA20」= 未来函数，实测 12.9% 的日子结论相反。
        ma_ok = (c.shift(1) >= ma20.shift(1))
        # 弱势同理：09:31 不知道当日收盘，用「今开 / 昨收 - 1」代理实时涨跌幅。
        ir = (idx_open.reindex(dates) / c.shift(1) - 1).fillna(0.0)
        score = build_score(close, amount)
        print(f"\n===== 宇宙 = {name}（{close.shape[1]} 只，"
              f"{dates[0].date()} ~ {dates[-1].date()}）  大盘代理 = {args.index} =====")
        pers = {}
        for label, b, g, w in setups:
            m, per = run(score, close, ir, ma_ok, args.topn, b, g, w)
            out[f"{name}|{label}"] = m
            pers[label] = per
            if m:
                print(f"  {label:<24} 总收益 {m['total']:>7.1f}%  年化 {m['annual']:>6.2f}%  "
                      f"最大回撤 {m['maxdd']:>6.1f}%  Sharpe {m['sharpe']:>5.2f}")
        _paired(pers)

    p = Path(cfg.resolve("results")) / f"risk_rerun_pit_{args.index}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[37] 结果已写入 {p}")
    print("[37] 口径：闸门与弱势判断均只用 D-1 及之前（09:31 可执行信息集）。")
    print("[37] 判读：看「同一行的两列」——风控在老池上带来的改善，在无偏宇宙上还在不在。")


if __name__ == "__main__":
    main()
