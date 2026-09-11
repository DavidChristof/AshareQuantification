"""第 37 步：把此前的**风控回测**放到 PIT 无偏宇宙上重跑一遍（老池 vs 无偏池对照）。

## 为什么必须重跑

`docs/2026-09-11-risk-control.md`（回撤熔断 / 弱势减仓）与同日的趋势闸门、入场择时闸门，
**全部是在有幸存者偏差的 559 池上测的**。而 `docs/2026-09-11-pit-universe.md` 已证明：
换手率的结论在无偏宇宙上**直接反转**（老池「无改善」→ 无偏池「+61pp」）。
⇒ 风控结论的绝对水平同样可疑，必须重测。

## 重跑什么

同一套组合、同一套阈值，只换宇宙：

    策略   = 现有 4 因子综合分（mom20 / trd / -vol20 / rev60）→ top12 等权
    调仓   = 每 5 个交易日，含换手费 0.16%/期
    A 基准（无风控）
    B + 账户回撤熔断 8%/4%（带滞回；触发则仓位上限降到 50% 且停开新仓）
    C + 大盘趋势闸门（沪深300 < MA20 → 当日不开新仓，保留持仓）
    D + 弱势减半（大盘当日 ≤ -1% → 仓位腰斩）

用法：
    python scripts/37_risk_rerun_pit.py
    python scripts/37_risk_rerun_pit.py --start 2021-07-01 --topn 12
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
        index_ma_ok: pd.Series, topn: int, brake: bool, gate: bool, weak: bool) -> dict:
    """每 REBAL 日调仓；风控在**新买入**与**仓位**上生效。

    index_ret : 大盘当日收益（弱势判断用），index_ma_ok: 大盘是否在 MA20 上方。
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
        return {}
    ser = pd.Series(dict(curve))
    yrs = (ser.index[-1] - ser.index[0]).days / 365.25
    tot = ser.iloc[-1] - 1
    per = ser.pct_change().dropna()
    return {"total": round(tot * 100, 1),
            "annual": round(((1 + tot) ** (1 / yrs) - 1) * 100, 2) if yrs > 0 else 0.0,
            "maxdd": round(float((ser / ser.cummax() - 1).min()) * 100, 1),
            "sharpe": round(float(per.mean() / per.std() * np.sqrt(252 / REBAL)), 2)
            if per.std() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2021-07-01")
    ap.add_argument("--topn", type=int, default=12)
    args = ap.parse_args()

    cfg = load_config()
    idx = fetch_index_daily("sh000300")
    idx = idx.set_index("date")["close"].sort_index()
    idx.index = pd.to_datetime(idx.index)

    setups = [("A 基准（无风控）", False, False, False),
              ("B +回撤熔断 8%/4%", True, False, False),
              ("C +熔断+趋势闸门", True, True, False),
              ("D +熔断+闸门+弱势减半", True, True, True)]

    out = {}
    for name, loader in (("large", load_large), ("pit", load_pit)):
        close, amount, _ = loader(cfg, args.start)
        dates = close.index
        ir = idx.reindex(dates).pct_change().fillna(0.0)
        ma20 = idx.reindex(dates).rolling(20).mean()
        ma_ok = (idx.reindex(dates) >= ma20)
        score = build_score(close, amount)
        print(f"\n===== 宇宙 = {name}（{close.shape[1]} 只，"
              f"{dates[0].date()} ~ {dates[-1].date()}）=====")
        for label, b, g, w in setups:
            m = run(score, close, ir, ma_ok, args.topn, b, g, w)
            out[f"{name}|{label}"] = m
            if m:
                print(f"  {label:<24} 总收益 {m['total']:>7.1f}%  年化 {m['annual']:>6.2f}%  "
                      f"最大回撤 {m['maxdd']:>6.1f}%  Sharpe {m['sharpe']:>5.2f}")

    p = Path(cfg.resolve("results")) / "risk_rerun_pit.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[37] 结果已写入 {p}")
    print("[37] 判读：看「同一行的两列」——风控在老池上带来的改善，在无偏宇宙上还在不在。")


if __name__ == "__main__":
    main()
