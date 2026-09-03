"""绩效指标模块：把净值曲线转换成可量化的风险收益指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252  # A股年交易日数


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """净值序列 → 日收益率序列。"""
    return equity.pct_change().dropna()


def annual_return(equity: pd.Series) -> float:
    """年化收益率。"""
    n = len(equity)
    if n < 2 or equity.iloc[0] <= 0:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0]
    return float(total ** (TRADING_DAYS / n) - 1)


def annual_volatility(equity: pd.Series) -> float:
    """年化波动率。"""
    ret = returns_from_equity(equity)
    return float(ret.std() * np.sqrt(TRADING_DAYS)) if len(ret) else 0.0


def sharpe_ratio(equity: pd.Series, risk_free: float = 0.02) -> float:
    """夏普比率：单位风险的超额收益，越高越好。"""
    ret = returns_from_equity(equity)
    if ret.std() == 0 or len(ret) == 0:
        return 0.0
    excess = ret.mean() * TRADING_DAYS - risk_free
    return float(excess / (ret.std() * np.sqrt(TRADING_DAYS)))


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤：从峰值到谷底的最大跌幅（负数，越小越差）。"""
    if len(equity) == 0:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def win_rate(equity: pd.Series) -> float:
    """交易胜率：上涨日占全部交易日的比例。"""
    ret = returns_from_equity(equity)
    if len(ret) == 0:
        return 0.0
    return float((ret > 0).mean())


def summarize(equity: pd.Series, label: str = "strategy") -> dict:
    """汇总全部指标。"""
    return {
        "label": label,
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1),
        "annual_return": annual_return(equity),
        "annual_volatility": annual_volatility(equity),
        "sharpe": sharpe_ratio(equity),
        "max_drawdown": max_drawdown(equity),
        "win_rate": win_rate(equity),
        "final_equity": float(equity.iloc[-1]),
    }


def compare(*equities: tuple[str, pd.Series]) -> pd.DataFrame:
    """对比多组净值曲线的绩效。"""
    rows = [summarize(eq, label) for label, eq in equities]
    return pd.DataFrame(rows).set_index("label")


def plot_equity(equities: dict[str, pd.Series], title: str = "Equity Curve",
                save_path: str | None = None):
    """绘制净值曲线对比图。"""
    import matplotlib

    # 中文字体设置（失败则回退英文，不影响功能）
    try:
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:  # noqa: BLE001
        pass

    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 5))
    for label, eq in equities.items():
        norm = eq / eq.iloc[0]  # 归一化到 1 便于对比
        plt.plot(norm.index, norm.values, label=label)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("Normalized Equity")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# 风险调整指标（教材 2.4：β / Jensen Alpha / 信息比率 / Treynor / Sortino）
# 全部以「基准（benchmark）」为参照，用于评价主动管理能力。
# benchmark 可以是净值序列（首值≈资金量）或收益序列，自动识别。
# ---------------------------------------------------------------------------

def _align_returns(equity, benchmark):
    """对齐两条净值曲线为收益序列（按共同日期）。"""
    rp = returns_from_equity(equity)
    rb = returns_from_equity(benchmark)
    idx = rp.index.intersection(rb.index)
    return rp.loc[idx], rb.loc[idx]


def _align_returns2(equity, benchmark):
    """对齐净值/收益混合输入为收益序列。

    benchmark 若是净值序列（首值远大于 1），先转收益；若是收益序列直接用。
    返回值：strategy 收益、benchmark 收益（同一日期）。
    """
    def _as_rets(s: pd.Series) -> pd.Series:
        s = s.dropna()
        if len(s) == 0:
            return s
        # 净值序列特征：首值很大（资金量）或值域较宽（归一化净值可到 0.5 以上）；
        # 日收益序列几乎都在 ±0.5 内。满足其一 → 视为净值需转收益。
        if abs(float(s.iloc[0])) > 10 or float(s.abs().max()) > 0.5:
            return s.pct_change().dropna()
        return s

    rp = _as_rets(equity)
    rb = _as_rets(benchmark)
    idx = rp.index.intersection(rb.index)
    return rp.loc[idx], rb.loc[idx]


def _beta_from_returns(rp: pd.Series, rb: pd.Series) -> float:
    """β = Cov(R_p, R_m) / Var(R_m)。"""
    if len(rp) < 2:
        return 0.0
    var_b = float(rb.var())
    if var_b == 0 or np.isnan(var_b):
        return 0.0
    cov = float(np.cov(rp, rb)[0, 1])
    return cov / var_b


def beta(equity, benchmark) -> float:
    """β 系数：策略相对基准的敏感度。β=1 同涨跌，>1 更激进，<1 更稳健。"""
    rp, rb = _align_returns2(equity, benchmark)
    return _beta_from_returns(rp, rb)


def jensen_alpha(equity, benchmark, risk_free: float = 0.02) -> float:
    """Jensen Alpha（年化）：α = R_p - [R_f + β·(R_m - R_f)]。α>0 表示跑赢基准风险补偿。"""
    rp, rb = _align_returns2(equity, benchmark)
    if len(rp) < 2:
        return 0.0
    b = _beta_from_returns(rp, rb)
    rp_ann = float(rp.mean() * TRADING_DAYS)
    rb_ann = float(rb.mean() * TRADING_DAYS)
    return rp_ann - risk_free - b * (rb_ann - risk_free)


def treynor_ratio(equity, benchmark, risk_free: float = 0.02) -> float:
    """Treynor 比率：超额收益 / β，衡量每单位系统性风险的超额回报。"""
    rp, rb = _align_returns2(equity, benchmark)
    if len(rp) < 2:
        return 0.0
    b = _beta_from_returns(rp, rb)
    if b == 0:
        return 0.0
    rp_ann = float(rp.mean() * TRADING_DAYS)
    return (rp_ann - risk_free) / b


def information_ratio(equity, benchmark) -> float:
    """信息比率 IR：年化主动收益 / 跟踪误差（主动收益的标准差）。

    衡量策略相对基准的稳定性，>0.5 良好、>1 优秀。
    """
    rp, rb = _align_returns2(equity, benchmark)
    active = rp - rb
    if len(active) < 2:
        return 0.0
    te = float(active.std() * np.sqrt(TRADING_DAYS))
    if te == 0:
        return 0.0
    return float(active.mean() * TRADING_DAYS) / te


def sortino_ratio(equity, risk_free: float = 0.02) -> float:
    """Sortino 比率：只用下行波动率惩罚风险（夏普的改进版）。"""
    rp = returns_from_equity(equity)
    if len(rp) < 2:
        return 0.0
    downside = rp[rp < 0]
    if len(downside) == 0:
        return 0.0
    dstd = float(downside.std() * np.sqrt(TRADING_DAYS))
    if dstd == 0:
        return 0.0
    return float(rp.mean() * TRADING_DAYS - risk_free) / dstd


def capture_ratio(equity, benchmark) -> tuple[float, float]:
    """上行/下行捕获率：基准上涨日策略平均收益 / 基准下跌日策略平均收益。

    上行捕获 >1 表示涨时涨得更多，下行捕获 <1 表示跌时跌得更少（好）。
    返回 (up_capture, down_capture)。
    """
    rp, rb = _align_returns2(equity, benchmark)
    if len(rp) < 2:
        return 0.0, 0.0
    up = rb > 0
    down = rb < 0
    up_cap = float(rp[up].mean()) / float(rb[up].mean()) if up.any() and rb[up].mean() != 0 else 0.0
    down_cap = float(rp[down].mean()) / float(rb[down].mean()) if down.any() and rb[down].mean() != 0 else 0.0
    return up_cap, down_cap


def risk_adjusted_summary(equity, benchmark, label: str = "strategy",
                          risk_free: float = 0.02) -> dict:
    """汇总全部风险调整指标（含基础夏普/回撤，便于完整对比）。"""
    up_cap, down_cap = capture_ratio(equity, benchmark)
    return {
        "label": label,
        "beta": beta(equity, benchmark),
        "jensen_alpha": jensen_alpha(equity, benchmark, risk_free),
        "treynor": treynor_ratio(equity, benchmark, risk_free),
        "information_ratio": information_ratio(equity, benchmark),
        "sortino": sortino_ratio(equity, risk_free),
        "sharpe": sharpe_ratio(equity, risk_free),
        "up_capture": up_cap,
        "down_capture": down_cap,
        "max_drawdown": max_drawdown(equity),
    }


def compare_risk_adjusted(*equities: tuple[str, pd.Series], benchmark: pd.Series,
                          risk_free: float = 0.02) -> pd.DataFrame:
    """对比多组净值曲线的风险调整指标（以 benchmark 为基准）。"""
    rows = [risk_adjusted_summary(eq, benchmark, label, risk_free)
            for label, eq in equities]
    return pd.DataFrame(rows).set_index("label")
