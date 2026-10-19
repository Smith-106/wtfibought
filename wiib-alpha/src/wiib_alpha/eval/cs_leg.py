"""多空腿：walk-forward 只学方向，样本外每个网格点按缓冲带持仓，等权、美元中性，持有到下一个网格点再调。

方向 = 训练窗里月度截面 IC 均值的符号。答案是对数收益，先 expm1 成算术收益，再减持有期内结算的资金费（多头付、空头收）。
缓冲带：排进头部 leg_enter 才做多、掉出头部 leg_exit 才平，空头对称看尾部，中间的不动；
方向换号、上一点没仓位或被跳过时从头建仓，本点不在样本里的币自动出场。
成本按流动性分档：每次调仓每个名字付 |w_t − w_{t−1}| × 该币当月档的单边成本，名字不换不付；
没仓位的网格点收益是 0，从有仓位变成没仓位那一点付平仓成本。毛收益 = 涨跌 + 资金费，扣交易成本之前；
盈亏平衡成本 = Σ毛收益 / Σ换手量，折成单边 bps。
排列检验：每个 t 内把仓位在币之间打乱、收益不动，看真实毛收益在随机里排第几；同一时刻币数相同的点拼成矩阵一起打乱。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .cs_ic import Points, ranks, trained
from .params import EvalParams, periods_per_year
from .stats import annualized_sharpe, compound
from .walkforward import Window

# 排列检验一块最多这么多个元素（迭代 × 点 × 币），约 32 MB
_CHUNK = 4_000_000


def directions(ic: np.ndarray, months: np.ndarray, wins: list[Window], min_points: int) -> np.ndarray:
    """每个网格点的方向：样本外是所在 test 块训练窗的月度 IC 均值符号，学不出来或不在样本外是 0"""
    out = np.zeros(len(ic))
    for w in wins:
        mean = trained(ic, months, w.train_end, min_points)
        out[w.test_start:w.test_end] = np.sign(mean) if np.isfinite(mean) else 0.0
    return out


def weights(p: Points, direction: np.ndarray, enter: float, exit_: float) -> np.ndarray:
    """每行的仓位：多头每个 +1/多头数，空头每个 −1/空头数；并列按币序号先后排"""
    n = np.bincount(p.g, minlength=p.total)[p.g]
    order = ranks(p.g, p.f, method="first")
    d = direction[p.g]
    # 多头那头从 1 数起的名次：方向为正时因子大的在头部；空头那头是倒过来数
    head = np.where(d > 0, n - order + 1, order)
    tail = n + 1 - head
    trading = d != 0
    enter_long, keep_long = trading & (head / n <= enter), trading & (head / n <= exit_)
    enter_short, keep_short = trading & (tail / n <= enter), trading & (tail / n <= exit_)
    long_, short = enter_long.copy(), enter_short.copy()
    # 上一点每个币是不是在多头 / 空头里，按币序号放
    held_long = np.zeros(int(p.sym.max(initial=-1)) + 1, dtype=bool)
    held_short = held_long.copy()
    starts = np.searchsorted(p.g, np.arange(p.total + 1))
    before = 0.0
    for g in range(p.total):
        a, b = starts[g], starts[g + 1]
        sym = p.sym[a:b]
        sign = np.sign(direction[g]) if a < b else 0.0
        if sign != 0 and sign == before:
            long_[a:b] |= keep_long[a:b] & held_long[sym]
            short[a:b] |= keep_short[a:b] & held_short[sym]
        held_long[:] = False
        held_short[:] = False
        held_long[sym] = long_[a:b]
        held_short[sym] = short[a:b]
        before = sign
    longs = np.bincount(p.g, long_, p.total)[p.g]
    shorts = np.bincount(p.g, short, p.total)[p.g]
    return np.where(long_, 1 / np.maximum(longs, 1), 0.0) - np.where(short, 1 / np.maximum(shorts, 1), 0.0)


def rebalance(p: Points, w: np.ndarray, unit_costs: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """每个网格点的换仓成本与 Σ|Δw|：本点仓位减上一点仓位，按币当月档的单边成本计"""
    held = w != 0
    now = pd.DataFrame({"g": p.g[held], "sym": p.sym[held], "w": w[held], "tier": p.tier[held]})
    before = now.assign(g=now["g"] + 1)
    both = now.merge(before, on=["g", "sym"], how="outer", suffixes=("", "_prev"))
    delta = (both["w"].fillna(0.0) - both["w_prev"].fillna(0.0)).abs().to_numpy()
    # 离场的币这一点不在样本里，用它上一点的档
    tier = both["tier"].fillna(both["tier_prev"]).to_numpy(dtype="int64")
    unit = np.asarray(unit_costs)[tier - 1]
    g = both["g"].to_numpy(dtype="int64")
    keep = g < p.total
    return np.bincount(g[keep], delta[keep] * unit[keep], p.total), np.bincount(g[keep], delta[keep], p.total)


def permutation(p: Points, w: np.ndarray, r: np.ndarray, iterations: int, seed: int) -> float:
    """每个网格点内把仓位在币之间打乱、收益不动，随机毛收益 < 真实毛收益的比例"""
    active = np.bincount(p.g, np.abs(w), p.total)[p.g] > 0
    if not active.any():
        return 0.0
    g, w, r = p.g[active], w[active], r[active]
    real = float(w @ r)
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    sizes = np.diff(np.r_[starts, len(g)])
    rng = np.random.default_rng(seed)
    totals = np.zeros(iterations)
    for n in np.unique(sizes):
        idx = starts[sizes == n][:, None] + np.arange(n)
        wn, rn = w[idx], r[idx]
        step = max(1, _CHUNK // rn.size)
        for a in range(0, iterations, step):
            b = min(iterations, a + step)
            shuffled = rng.permuted(np.broadcast_to(rn, (b - a, *rn.shape)), axis=2)
            totals[a:b] += np.einsum("ktn,tn->k", shuffled, wn)
    return float(np.mean(totals < real))


def simulate(p: Points, ic: np.ndarray, wins: list[Window], bars: int, params: EvalParams,
             iterations: int) -> dict[str, Any]:
    """样本外每个网格点一期；iterations = 0 不做排列检验"""
    oos = np.zeros(p.total, dtype=bool)
    for win in wins:
        oos[win.test_start:win.test_end] = True
    if not oos.any():
        return {"periods": 0}
    direction = directions(ic, p.months, wins, params.min_points)
    w = weights(p, direction, params.leg_enter, params.leg_exit)
    # 一个币持有一期的收益：涨跌减资金费，空头仓位 w < 0 正好变成收费
    r = np.expm1(p.y) - p.fund
    gross = np.bincount(p.g, w * r, p.total)
    funding = -np.bincount(p.g, w * p.fund, p.total)
    cost, traded = rebalance(p, w, params.costs)
    net = (gross - cost)[oos]
    active = (np.bincount(p.g, np.abs(w), p.total) > 0)[oos]
    volume = traded[oos].sum()
    return {
        "periods": int(oos.sum()),
        "netReturn": compound(net),
        "grossReturn": compound(gross[oos]),
        # 资金费每期平均多少 bps，正数是收、负数是付；已经算在毛收益里
        "fundingBps": float(funding[oos].mean() * 1e4),
        "sharpe": annualized_sharpe(net, periods_per_year(bars)),
        "hitRate": float(np.mean(net > 0)),
        "exposure": float(np.mean(active)),
        # Σ|Δw| / 4：两条腿整个换掉是 1，从空仓建仓是 0.5
        "turnover": float(np.mean(traded[oos]) / 4),
        # 每单位换手量赚的毛收益：单边成本到这个数，净收益正好是 0
        "breakEvenBps": float(gross[oos].sum() / volume * 1e4) if volume > 0 else np.nan,
        "permPct": permutation(p, w, r, iterations, params.seed) if iterations else None,
    }
