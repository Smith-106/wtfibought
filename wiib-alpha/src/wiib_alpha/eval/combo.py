"""因子组合：库里的因子合成一条截面信号。

每条因子在每个时刻跨币先转秩、再对秩做 z 分数，按扩张训练窗里的月度截面 IC 均值加权求和，再除以权重绝对值之和；
切窗与单因子同一套（h 档网格点），信号只在 test 块有值，IC 和多空腿也与单因子一样算。
候选的增量 = 把它加进组合后，主档组合多空腿扣成本和资金费的年化净 Sharpe、三档样本外 IC 的变化；G9 看主档的这两个。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..data.pool import on_grid
from ..store import now_ms
from . import cs_ic, cs_leg
from .horizon import grid_windows, horizon, oos
from .params import EvalParams
from .recheck import after
from .walkforward import Window


@dataclass(frozen=True)
class Combo:
    """一组因子在骨架上的值：rows 是 pool.search（搜索段）或 pool.rows（全量），values (行数, N)；N 可以是 0（库是空的）"""
    rows: pd.DataFrame
    symbols: tuple[str, ...]
    names: tuple[str, ...]
    values: np.ndarray

    @classmethod
    def empty(cls, rows: pd.DataFrame, symbols: tuple[str, ...]) -> "Combo":
        return cls(rows, symbols, (), np.empty((len(rows), 0)))

    def plus(self, name: str, f: np.ndarray) -> "Combo":
        """f 与 rows 对齐"""
        return Combo(self.rows, self.symbols, (*self.names, name), np.column_stack([self.values, f]))


def zscores(combo: Combo) -> np.ndarray:
    """每个时刻跨币先转秩、再对秩做 z 分数，(行数, N)：因子原值的极端值很多，直接做 z 分数会让一两个币带偏组合"""
    t = combo.rows["t"].to_numpy()
    ranked = pd.DataFrame(combo.values).groupby(t).rank(pct=True)
    g = ranked.groupby(t)
    with np.errstate(all="ignore"):
        return ((ranked - g.transform("mean")) / g.transform("std")).to_numpy()


def _combine(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Σ w·z / Σ|w|，w 是一行权重或每行各一组；缺值的因子当 0（截面中间）算，没权重的不算，一个有值的都没有是 NaN"""
    terms = z * w
    got = np.isfinite(terms) & (w != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = np.where(got, terms, 0.0).sum(axis=1) / np.abs(w).sum(axis=-1)
    total[~got.any(axis=1)] = np.nan
    return total


def _weights(ics: list[np.ndarray], months: np.ndarray, end: int, min_points: int) -> np.ndarray:
    """每条因子在 [0, end) 的月度 IC 均值，学不出来的当 0"""
    w = np.array([cs_ic.trained(ic, months, end, min_points) for ic in ics])
    return np.where(np.isfinite(w), w, 0.0)


def _weighted(combo: Combo, z: np.ndarray, bars: int, params: EvalParams,
              blocks: Callable[[cs_ic.Points], list[Window]]) -> np.ndarray:
    """每个块用它训练窗学到的权重出信号，与 rows 对齐，块外是 NaN"""
    ics = []
    for j in range(len(combo.names)):
        p = cs_ic.points(combo.rows, combo.values[:, j], bars, params.min_names)
        ics.append(cs_ic.series(p))
    t = combo.rows["t"].to_numpy(dtype="int64")
    out = np.full(len(t), np.nan)
    wins = blocks(p)
    if not wins:
        return out
    on = on_grid(t, bars)
    g = np.full(len(t), -1)
    g[on] = cs_ic.grid_index(t[on], bars, int(p.times[0]))
    # 块是按时间排好、互不重叠的：每行落在哪一块，就用那一块的权重
    starts = np.array([b.test_start for b in wins])
    ends = np.array([b.test_end for b in wins])
    weights = np.array([_weights(ics, p.months, b.train_end, params.min_points) for b in wins])
    k = np.searchsorted(starts, g, side="right") - 1
    inside = (g >= 0) & (k >= 0) & (g < ends[np.maximum(k, 0)])
    out[inside] = _combine(z[inside], weights[k[inside]])
    return out


def signal(combo: Combo, z: np.ndarray, bars: int, params: EvalParams) -> np.ndarray:
    """walk-forward 组合信号，只在 test 块有值"""
    return _weighted(combo, z, bars, params, lambda p: grid_windows(p.total, bars, params))


def ics(combo: Combo, params: EvalParams) -> dict[int, dict[str, Any]]:
    """三档样本外 IC，口径与单因子一样；主档再带组合多空腿扣成本和资金费后的年化净 Sharpe（sharpe）。库空时全是 NaN"""
    z = zscores(combo) if combo.names else None
    out = {}
    for bars in HORIZON_BARS.values():
        s = signal(combo, z, bars, params) if combo.names else np.full(len(combo.rows), np.nan)
        p, ic, wins, lo, hi = oos(combo.rows, s, bars, params)
        out[bars] = cs_ic.summary(p, ic, lo, hi, params)
        if bars == params.primary:
            out[bars]["sharpe"] = cs_leg.simulate(p, ic, wins, bars, params, 0).get("sharpe", np.nan)
    return out


def evaluate(combo: Combo, params: EvalParams) -> dict[int, dict[str, Any]]:
    """三档样本外 IC 加多空腿，与单因子走同一个 horizon"""
    z = zscores(combo)
    return {bars: horizon(combo.rows, signal(combo, z, bars, params), bars, params, params.iterations)
            for bars in HORIZON_BARS.values()}


def gain(combo: Combo, name: str, f: np.ndarray, params: EvalParams) -> dict[str, Any]:
    """把候选加进组合后：三档样本外 IC 的变化、主档组合多空腿年化净 Sharpe 的变化，主档再数几个月 IC 比原来好；
    库空时原来都当 0"""
    base = ics(combo, params) if combo.names else None
    plus = ics(combo.plus(name, f), params)
    horizons = {}
    for bars, after_ in plus.items():
        before = base[bars]["mean"] if base else 0.0
        horizons[bars] = {"base": before, "with": after_["mean"], "gain": after_["mean"] - before}
    main = params.primary
    # 组合一期仓位都没有时收益全是 0，Sharpe 算不出来，按 0
    sharpe_before = float(np.nan_to_num(base[main]["sharpe"])) if base else 0.0
    sharpe_with = float(np.nan_to_num(plus[main]["sharpe"]))
    months = plus[main]["monthly"]
    before_months = base[main]["monthly"] if base else {}
    return {"horizons": horizons, "months": len(months),
            "monthsBetter": sum(1 for m, v in months.items() if v > before_months.get(m, 0.0)),
            "sharpe": {"base": sharpe_before, "with": sharpe_with, "gain": sharpe_with - sharpe_before}}


def holdout(combo: Combo, cut_ms: int, params: EvalParams) -> dict[str, Any]:
    """combo 是全量骨架：搜索段整段当训练窗定权重，只在 cut 之后出信号算 IC；形状与 recheck 一样"""
    z = zscores(combo)
    out: dict[str, Any] = {"since": cut_ms, "at": now_ms(), "horizons": {}}
    for bars in HORIZON_BARS.values():
        def block(p: cs_ic.Points) -> list[Window]:
            start = int(np.searchsorted(p.times, cut_ms, side="right"))
            # 训练窗最后一个点的答案要在 cut 之前收完：隔 purge 一个点再加 embargo
            return [Window(start - 1 - params.grid(bars)[1], start, p.total)]
        out["horizons"][bars] = after(combo.rows, _weighted(combo, z, bars, params, block), bars, cut_ms, params)
    return out


def latest(combo: Combo, params: EvalParams) -> dict[str, Any]:
    """combo 是全量骨架：全部数据当训练窗定主档权重，出最后一个网格时刻每个在池币的组合信号"""
    ics_ = []
    for j in range(len(combo.names)):
        p = cs_ic.points(combo.rows, combo.values[:, j], params.primary, params.min_names)
        ics_.append(cs_ic.series(p))
    w = _weights(ics_, p.months, p.total, params.min_points)
    t = combo.rows["t"].to_numpy()
    last = t == t.max()
    values = _combine(zscores(combo)[last], w)
    syms = combo.rows["sym"].to_numpy()[last]
    return {"asOf": int(t.max()),
            "weights": [{"name": n, "weight": float(v)} for n, v in zip(combo.names, w)],
            "latest": {combo.symbols[s]: float(v) for s, v in zip(syms, values)}}
