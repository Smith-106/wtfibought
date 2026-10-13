"""方向腿：训练窗学因子的方向与尺度，样本外按 |z| > 1 开仓，与 Java DirectionLegSimulator 同口径。

非重叠采样是固定步长网格：从第一个 OOS 点起每 h 根 bar 取一个点，平仓信号也占格子。
每个非零信号当一笔独立往返交易扣一次成本；排列检验用不扣成本的收益，成本对洗牌不变。
答案列是对数收益，进方向腿前换成算术收益，复利和成本都按算术口径，与 Java longReturn 一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .params import EvalParams, periods_per_year
from .stats import annualized_sharpe, compound, grouped_permutation_percentile, permutation_percentile
from .walkforward import TrainStats, Window


@dataclass(frozen=True)
class Leg:
    rows: np.ndarray        # 网格点行号
    positions: np.ndarray   # -1 / 0 / 1
    targets: np.ndarray     # 网格点的算术收益 exp(fwd_ret) − 1
    net: np.ndarray         # 方向 × 答案 − 成本
    exposure: float         # 重叠口径：非零信号占比
    turnover: float         # 重叠口径：方向变化次数 / (点数 − 1)


def simulate(factor: np.ndarray, target: np.ndarray, wins: list[Window], stats: TrainStats,
             horizon_bars: int, cost: float) -> Leg | None:
    n = len(factor)
    dirs = np.zeros(n, dtype="int64")
    valid = np.zeros(n, dtype=bool)
    for w in wins:
        seg = slice(w.test_start, w.test_end)
        valid[seg] = True
        mean, std, corr = stats.at(w.train_end)
        if not np.isfinite(corr) or not np.isfinite(std):
            continue
        with np.errstate(invalid="ignore"):
            z = (factor[seg] - mean) / std
        d = np.where(z > 1, 1, np.where(z < -1, -1, 0))
        d[~np.isfinite(z)] = 0
        dirs[seg] = int(np.sign(corr)) * d
    rows_all = np.flatnonzero(valid & np.isfinite(target))
    if len(rows_all) == 0:
        return None
    d_all = dirs[rows_all]
    exposure = float(np.mean(d_all != 0))
    turnover = float(np.mean(d_all[1:] != d_all[:-1])) if len(d_all) > 1 else 0.0
    grid = np.arange(rows_all[0], rows_all[-1] + 1, horizon_bars)
    grid = grid[np.isfinite(target[grid])]
    d = dirs[grid]
    y = np.expm1(target[grid])
    net = d * y - cost * (d != 0)
    return Leg(grid, d, y, net, exposure, turnover)


def summarize(leg: Leg, horizon_bars: int, params: EvalParams) -> dict[str, Any]:
    active = leg.positions != 0
    return {
        "periods": int(len(leg.net)),
        "netReturn": compound(leg.net),
        "sharpe": annualized_sharpe(leg.net, periods_per_year(horizon_bars)),
        "hitRate": float(np.mean(leg.net > 0)) if len(leg.net) else 0.0,
        "activeHitRate": float(np.mean(leg.net[active] > 0)) if active.any() else 0.0,
        "exposure": leg.exposure,
        "turnover": leg.turnover,
        "permPct": permutation_percentile(leg.positions, leg.targets, params.iterations, params.seed),
    }


def pool(legs: dict[str, Leg], horizon_bars: int, params: EvalParams) -> dict[str, Any]:
    """全部币的网格点串成一条算命中；排列检验只在币内洗仓位；净收益按每币各自复利再等权平均，Sharpe 各币取中位数"""
    if not legs:
        return {"periods": 0}
    counts = np.array([len(l.net) for l in legs.values()], dtype="float64")
    merged = Leg(
        rows=np.concatenate([l.rows for l in legs.values()]),
        positions=np.concatenate([l.positions for l in legs.values()]),
        targets=np.concatenate([l.targets for l in legs.values()]),
        net=np.concatenate([l.net for l in legs.values()]),
        exposure=float(np.average([l.exposure for l in legs.values()], weights=counts)) if counts.sum() else 0.0,
        turnover=float(np.average([l.turnover for l in legs.values()], weights=counts)) if counts.sum() else 0.0,
    )
    out = summarize(merged, horizon_bars, params)
    groups = np.concatenate([np.full(len(l.net), i) for i, l in enumerate(legs.values())])
    out["permPct"] = grouped_permutation_percentile(merged.positions, merged.targets, groups,
                                                    params.iterations, params.seed)
    out["netReturn"] = float(np.mean([compound(l.net) for l in legs.values()]))
    sharpes = [annualized_sharpe(l.net, periods_per_year(horizon_bars)) for l in legs.values()]
    out["sharpe"] = float(np.nanmedian(sharpes)) if np.isfinite(sharpes).any() else np.nan
    return out
