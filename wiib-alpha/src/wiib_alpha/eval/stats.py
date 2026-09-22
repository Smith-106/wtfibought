"""统计原语：Spearman、Aronson 排列检验、BH-FDR、年化 Sharpe、复利。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def spearman(a: np.ndarray, b: np.ndarray, min_points: int = 3) -> float:
    """秩上的 Pearson，并列取平均秩；只用两边都有数的行，点太少或有一边不动就是 NaN"""
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_points:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def permutation_percentile(positions: np.ndarray, returns: np.ndarray, iterations: int, seed: int) -> float:
    """固定真实仓位序列，洗牌 iterations 次打乱时机，数有多少次随机收益 < 真实收益。

    保留下注次数与方向分布，只问"这个时机选得比瞎猜好多少"。seed 固定可复现。
    """
    pos = np.asarray(positions, dtype="float64")
    ret = np.asarray(returns, dtype="float64")
    if len(pos) == 0 or iterations <= 0:
        return 0.0
    real = pos @ ret
    rng = np.random.default_rng(seed)
    less = 0
    for _ in range(iterations):
        if rng.permutation(pos) @ ret < real:
            less += 1
    return less / iterations


def grouped_permutation_percentile(positions: np.ndarray, returns: np.ndarray, groups: np.ndarray,
                                   iterations: int, seed: int) -> float:
    """仓位只在各自的组（币）内洗牌，合并后的收益与真实比；收益不动，币之间的同涨同跌留在零分布里"""
    pos = np.asarray(positions, dtype="float64")
    ret = np.asarray(returns, dtype="float64")
    grp = np.asarray(groups)
    if len(pos) == 0 or iterations <= 0:
        return 0.0
    real = pos @ ret
    segments = [np.flatnonzero(grp == g) for g in np.unique(grp)]
    rng = np.random.default_rng(seed)
    less = 0
    for _ in range(iterations):
        shuffled = pos.copy()
        for idx in segments:
            shuffled[idx] = rng.permutation(pos[idx])
        if shuffled @ ret < real:
            less += 1
    return less / iterations


def bh_reject(pvalues: Sequence[float], q: float) -> list[bool]:
    """Benjamini-Hochberg：按 p 排序，找最大的 k 使 p_(k) ≤ k/m·q，前 k 个算显著"""
    ps = np.asarray(pvalues, dtype="float64")
    m = len(ps)
    if m == 0:
        return []
    order = np.argsort(ps)
    ranked = ps[order]
    passing = np.flatnonzero(ranked <= (np.arange(1, m + 1) / m) * q)
    k = int(passing[-1]) + 1 if len(passing) else 0
    out = np.zeros(m, dtype=bool)
    out[order[:k]] = True
    return out.tolist()


def annualized_sharpe(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype="float64")
    if len(r) < 2:
        return np.nan
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else np.nan


def compound(returns: np.ndarray) -> float:
    """Π(1+r) − 1"""
    r = np.asarray(returns, dtype="float64")
    return float(np.prod(1.0 + r) - 1.0) if len(r) else 0.0
