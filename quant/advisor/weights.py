"""动态选股权重学习器：让基本面打分权重「自适应」。

权重不是拍脑袋固定，而是根据两个信号动态调整：

1. 市场因子有效性（近期哪个因子在赚钱）
    统计每个因子值与「股票近期收益率」的秩相关（Spearman）：
    - 若「高 ROE 的股票近期涨得好」→ ROE 因子近期有效 → 权重上调
    - 若「低 PE 的股票近期涨得好」→ PE 因子近期有效 → 权重上调
    因子方向统一为「越大越好」：PE 用其倒数 1/PE（盈利收益率）。

2. 用户盈利偏好（你的模拟炒股表现）
    你模拟盘里「赚钱的持仓」共同的因子画像 vs 全池平均：
    - 若你赚钱的股票普遍高 ROE → 温和上调 ROE 权重（个性化）
    - 若你赚钱的股票普遍低 PE → 温和上调 PE 权重

公式：
    weight_f = clamp(base_f * (1 + 0.5*effect_f) * (1 + strength*pref_f), min, max)
    最后归一化到权重和为 1。

⚠️ 说明：这是探索性功能。小样本（40 只）下相关性有噪声，
已用温和的调整幅度 + 上下限约束控制过拟合风险。
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from ..config import Config

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {"pe": 0.5, "roe": 0.5}   # 基础权重
MARKET_STRENGTH = 0.5                        # 市场有效性影响强度
USER_STRENGTH = 0.3                          # 用户偏好影响强度（温和）
MIN_W, MAX_W = 0.2, 0.8                      # 权重上下限（防极端）
MIN_SAMPLE = 6                               # 相关性计算最少样本数


def _load_factor_cache(cfg: Config) -> dict:
    """读取选股时的因子缓存 {code: {pe, roe, score}}。"""
    path = cfg.resolve("results/factor_cache.json")
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("因子缓存读取失败，使用默认权重")
    return {}


def _recent_returns(data: dict, window: int = 20) -> dict[str, float]:
    """每只股票近 window 日收益率。"""
    result = {}
    for symbol, df in data.items():
        if len(df) >= window + 1:
            result[symbol] = float(df["close"].iloc[-1] / df["close"].iloc[-window - 1] - 1)
    return result


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（无 scipy 依赖的手写实现）。"""
    def _rank(arr):
        order = np.argsort(arr)
        ranks = np.empty(len(arr))
        ranks[order] = np.arange(1, len(arr) + 1)
        # 处理并列（平均值）
        sorted_arr = np.sort(arr)
        i = 0
        while i < len(arr):
            j = i
            while j + 1 < len(arr) and sorted_arr[j + 1] == sorted_arr[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (ranks[order[i]] + ranks[order[j]]) / 2
            i = j + 1
        return ranks

    rx, ry = _rank(x), _rank(y)
    rx_m, ry_m = rx.mean(), ry.mean()
    cov = np.mean((rx - rx_m) * (ry - ry_m))
    return float(cov / (rx.std() * ry.std())) if rx.std() and ry.std() else 0.0


def _market_effectiveness(factor_cache: dict, returns: dict) -> dict[str, float]:
    """每个因子与近期收益的秩相关（因子方向统一为「越大越好」）。

    - PE 用 1/PE（盈利收益率）：低 PE 股票收益高 → 相关为正 → PE 因子有效
    - ROE 直接用：高 ROE 股票收益高 → 相关为正 → ROE 因子有效
    """
    eff = {}
    for factor, direction in (("pe", "inverse"), ("roe", "direct")):
        vals, rets = [], []
        for code, meta in factor_cache.items():
            v = meta.get(factor)
            if v is None or code not in returns:
                continue
            f = 1.0 / v if direction == "inverse" and v > 0 else v
            if np.isfinite(f):
                vals.append(f)
                rets.append(returns[code])
        if len(vals) >= MIN_SAMPLE:
            eff[factor] = _spearman(np.array(vals), np.array(rets))
        else:
            eff[factor] = 0.0
    return eff


def _user_preference(factor_cache: dict, profit_symbols: list[str],
                     all_symbols: list[str]) -> dict[str, float]:
    """用户盈利持仓的因子画像 vs 全池平均 → 偏好方向（-1~1）。"""
    prof = {c: factor_cache[c] for c in profit_symbols if c in factor_cache}
    if not prof:
        return {"pe": 0.0, "roe": 0.0}

    def _mean(factor):
        vals = [factor_cache[c].get(factor) for c in all_symbols if c in factor_cache]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.mean(vals)) if vals else float("nan")

    pref = {}
    for factor in ("pe", "roe"):
        pv = np.mean([m.get(factor) for m in prof.values()])
        av = _mean(factor)
        if not np.isfinite(pv) or not np.isfinite(av) or av == 0:
            pref[factor] = 0.0
            continue
        # PE 越低越好 → 用户比平均便宜 → 偏好 PE 因子（取负号后为正偏好）
        # ROE 越高越好 → 用户比平均高 → 偏好 ROE 因子
        sign = -1.0 if factor == "pe" else 1.0
        pref[factor] = float(np.tanh(sign * (pv - av) / max(abs(av), 1e-9)))
    return pref


def learn_weights(cfg: Config, data: dict,
                  profit_symbols: list[str]) -> dict:
    """计算动态选股权重。

    Returns:
        {weights, effectiveness, user_pref, reason, meta}
    """
    factor_cache = _load_factor_cache(cfg)
    if not factor_cache:
        return {
            "weights": dict(DEFAULT_WEIGHTS),
            "effectiveness": {"pe": 0, "roe": 0},
            "user_pref": {"pe": 0, "roe": 0},
            "reason": ["暂无因子缓存（请先运行 scripts/07_build_universe.py）"],
            "meta": {"sample": 0},
        }

    returns = _recent_returns(data)
    eff = _market_effectiveness(factor_cache, returns)
    pref = _user_preference(factor_cache, profit_symbols, list(data.keys()))

    # 合成权重
    raw = {}
    for f in ("pe", "roe"):
        base = DEFAULT_WEIGHTS[f]
        market = base * (1 + MARKET_STRENGTH * eff[f])
        personal = 1 + USER_STRENGTH * pref[f]
        raw[f] = max(MIN_W, min(MAX_W, market * personal))
    total = sum(raw.values())
    weights = {f: round(w / total, 4) for f, w in raw.items()}

    # 生成可读依据
    reason = []
    for f, eff_v in eff.items():
        label = {"pe": "PE(低估值)", "roe": "ROE(高盈利)"}[f]
        if eff_v > 0.15:
            reason.append(f"{label}近期有效（相关 {eff_v:+.2f}），权重上调")
        elif eff_v < -0.15:
            reason.append(f"{label}近期失效（相关 {eff_v:+.2f}），权重下调")
        else:
            reason.append(f"{label}近期中性（相关 {eff_v:+.2f}）")
    if pref["pe"] > 0.1 or pref["roe"] > 0.1:
        reason.append("你模拟盘盈利持仓的因子画像与该因子一致，已个性化加权")
    elif pref["pe"] < -0.1 or pref["roe"] < -0.1:
        reason.append("你模拟盘盈利持仓与该因子画像偏离，权重已温和下调")

    return {
        "weights": weights,
        "effectiveness": {k: round(v, 3) for k, v in eff.items()},
        "user_pref": {k: round(v, 3) for k, v in pref.items()},
        "reason": reason,
        "meta": {
            "sample": len(returns),
            "profit_symbols": profit_symbols,
            "factor_cached": len(factor_cache),
        },
    }
