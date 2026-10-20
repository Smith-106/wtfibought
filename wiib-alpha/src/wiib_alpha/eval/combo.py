"""因子组合：库里的因子合成一条截面信号。

每条因子在每个时刻跨币先转秩、再对秩做 z 分数，按扩张训练窗里的月度截面 IC 均值加权求和，再除以权重绝对值之和；
切窗与单因子同一套（h 档网格点），信号只在 test 块有值，IC 和多空腿也与单因子一样算。
候选对交易方案的增量（G9）在 trade/scheme.py。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..data.pool import on_grid
from ..store import now_ms
from . import cs_ic
from .horizon import grid_windows, horizon
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


def holdout_signal(combo: Combo, z: np.ndarray, bars: int, cut_ms: int, params: EvalParams) -> np.ndarray:
    """combo 是全量骨架：搜索段整段当训练窗定权重，只在 cut 之后出信号"""
    def block(p: cs_ic.Points) -> list[Window]:
        start = int(np.searchsorted(p.times, cut_ms, side="right"))
        # 训练窗最后一个点的答案要在 cut 之前收完：隔 purge 一个点再加 embargo
        return [Window(start - 1 - params.grid(bars)[1], start, p.total)]
    return _weighted(combo, z, bars, params, block)


def learned(combo: Combo, bars: int, params: EvalParams) -> np.ndarray:
    """全部数据当训练窗学到的每条权重（月度截面 IC 均值）；combo 至少一条"""
    ics = []
    for j in range(len(combo.names)):
        p = cs_ic.points(combo.rows, combo.values[:, j], bars, params.min_names)
        ics.append(cs_ic.series(p))
    return _weights(ics, p.months, p.total, params.min_points)


def score(combo: Combo, w: np.ndarray) -> np.ndarray:
    """每行按一组权重合成的组合信号：截面秩 z 分数的加权和"""
    return _combine(zscores(combo), w)


def evaluate(combo: Combo, vol: np.ndarray, params: EvalParams) -> dict[int, dict[str, Any]]:
    """三档样本外 IC 加多空腿，与单因子走同一个 horizon；vol 与 combo.rows 对齐"""
    z = zscores(combo)
    return {bars: horizon(combo.rows, signal(combo, z, bars, params), vol, bars, params, params.iterations)
            for bars in HORIZON_BARS.values()}


def holdout(combo: Combo, cut_ms: int, params: EvalParams) -> dict[str, Any]:
    """combo 是全量骨架：搜索段整段当训练窗定权重，只在 cut 之后出信号算 IC；形状与 recheck 一样"""
    z = zscores(combo)
    out: dict[str, Any] = {"since": cut_ms, "at": now_ms(), "horizons": {}}
    for bars in HORIZON_BARS.values():
        out["horizons"][bars] = after(combo.rows, holdout_signal(combo, z, bars, cut_ms, params), bars, cut_ms, params)
    return out


def latest(combo: Combo, params: EvalParams) -> dict[str, Any]:
    """combo 是全量骨架：全部数据当训练窗定主档权重，出最后一个网格时刻每个在池币的组合信号"""
    w = learned(combo, params.primary, params)
    t = combo.rows["t"].to_numpy()
    last = t == t.max()
    values = score(combo, w)[last]
    syms = combo.rows["sym"].to_numpy()[last]
    return {"asOf": int(t.max()),
            "weights": [{"name": n, "weight": float(v)} for n, v in zip(combo.names, w)],
            "latest": {combo.symbols[s]: float(v) for s, v in zip(syms, values)}}
