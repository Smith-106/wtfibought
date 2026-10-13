"""因子组合：库里的因子合成一条信号，口径与 Java ContinuousFactorForecaster 同。

每条因子在训练窗里做 z-score、tanh 压住尾巴，按训练窗里与答案的 Pearson 相关加权求和，再除以权重绝对值之和。
权重跨币取平均：一条因子对所有币用同一个权重，与 G2 "多币同号" 一个口径。
信号只在 walk-forward 的 test 块有值；方向腿拿信号自己过去样本外的均值方差定 |z| > 1，与单因子同一套 leg。
候选的增量 = 把它加进组合后三档 OOS IC 的变化，主档的增量是 G9。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS, aligned
from ..dsl import Expr
from ..store import now_ms
from . import ic
from .horizon import horizon
from .params import EvalParams
from .walkforward import TrainStats, Window, windows


@dataclass(frozen=True)
class Combo:
    """一组因子在对齐后面板上的值：币 -> (n, N)；N 可以是 0（库是空的）。
    因子在各币完整面板 source 上算，再只取全部币都有的时刻"""
    source: dict[str, pd.DataFrame]
    panels: dict[str, pd.DataFrame]
    names: tuple[str, ...]
    values: dict[str, np.ndarray]

    @classmethod
    def build(cls, factors: Sequence[tuple[str, Expr]], panels: dict[str, pd.DataFrame]) -> "Combo":
        same = aligned(panels)
        combo = cls(panels, same, (), {s: np.empty((len(p), 0)) for s, p in same.items()})
        for name, expr in factors:
            combo = combo.plus(name, expr)
        return combo

    def plus(self, name: str, expr: Expr) -> "Combo":
        values = {s: np.column_stack([self.values[s], expr.evaluate(self.source[s]).loc[p.index].to_numpy(dtype="float64")])
                  for s, p in self.panels.items()}
        return Combo(self.source, self.panels, (*self.names, name), values)

    @property
    def rows(self) -> int:
        return len(next(iter(self.panels.values())))

    def targets(self, bars: int) -> dict[str, np.ndarray]:
        return {s: p[f"fwd_ret_{bars}"].to_numpy(dtype="float64") for s, p in self.panels.items()}


def score(values: dict[str, np.ndarray], targets: dict[str, np.ndarray], wins: list[Window]) -> dict[str, np.ndarray]:
    """walk-forward 组合信号，只在 test 块有值"""
    stats = {s: [TrainStats(v[:, i], targets[s]) for i in range(v.shape[1])] for s, v in values.items()}
    out = {s: np.full(len(t), np.nan) for s, t in targets.items()}
    for w in wins:
        means, stds, weights = _fit(stats, w.train_end)
        for s, v in values.items():
            out[s][w.test_start:w.test_end] = _combine(v[w.test_start:w.test_end], means[s], stds[s], weights)
    return out


def _fit(stats: dict[str, list[TrainStats]], train_end: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    """每币的 mean/std 向量，和跨币平均的权重向量；某币某因子在训练窗里算不出相关就不参与平均"""
    fits = {s: [st.at(train_end) for st in sts] for s, sts in stats.items()}
    count = len(next(iter(stats.values())))
    weights = np.zeros(count)
    for i in range(count):
        corrs = [fits[s][i][2] for s in fits if np.isfinite(fits[s][i][2])]
        weights[i] = float(np.mean(corrs)) if corrs else 0.0
    means = {s: np.array([f[0] for f in fs]) for s, fs in fits.items()}
    stds = {s: np.array([f[1] for f in fs]) for s, fs in fits.items()}
    return means, stds, weights


def _combine(block: np.ndarray, means: np.ndarray, stds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Σ w·tanh(z) / Σ|w|；缺值的和没权重的因子不算，一个都没有是 NaN"""
    denom = float(np.abs(weights).sum())
    if denom == 0.0:
        return np.full(len(block), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = np.tanh((block - means) / stds) * weights
    got = np.isfinite(terms) & (weights != 0)
    total = np.where(got, terms, 0.0).sum(axis=1) / denom
    total[~got.any(axis=1)] = np.nan
    return total


def _signals(combo: Combo, params: EvalParams) -> Iterator[tuple[int, list[Window], dict[str, np.ndarray], dict[str, np.ndarray]]]:
    """三档各一份 (bars, wins, targets, signal)"""
    for bars in HORIZON_BARS.values():
        targets = combo.targets(bars)
        wins = windows(combo.rows, params.test_size, bars, params.embargo, params.min_train)
        yield bars, wins, targets, score(combo.values, targets, wins)


def _points(combo: Combo, signal: dict[str, np.ndarray], targets: dict[str, np.ndarray], wins: list[Window],
            bars: int) -> dict[str, pd.DataFrame]:
    return {s: ic.oos_points(p.index.to_numpy(dtype="int64"), signal[s], targets[s], wins, bars)
            for s, p in combo.panels.items()}


def ics(combo: Combo, params: EvalParams) -> dict[int, dict[str, Any]]:
    """三档 OOS IC，口径与单因子一样"""
    return {bars: ic.summary(_points(combo, signal, targets, wins, bars), params.segments, params.min_points)
            for bars, wins, targets, signal in _signals(combo, params)}


def evaluate(combo: Combo, params: EvalParams) -> dict[int, dict[str, Any]]:
    """三档 OOS IC 加方向腿，与单因子走同一个 horizon"""
    return {bars: horizon(bars, combo.panels, signal, params) for bars, _, _, signal in _signals(combo, params)}


def gain(combo: Combo, name: str, expr: Expr, params: EvalParams) -> dict[str, Any]:
    """把候选加进组合后三档 OOS IC 的变化，主档再数有几个月比原来好；库空时原来就是 0"""
    base = ics(combo, params) if combo.names else None
    plus = ics(combo.plus(name, expr), params)
    horizons = {}
    for bars, after in plus.items():
        before = base[bars]["mean"] if base else 0.0
        horizons[bars] = {"base": before, "with": after["mean"], "gain": after["mean"] - before}
    months = plus[params.primary]["monthly"]
    before_months = base[params.primary]["monthly"] if base else {}
    return {"horizons": horizons, "months": len(months),
            "monthsBetter": sum(1 for m, v in months.items() if v > before_months.get(m, 0.0))}


def holdout(combo: Combo, cut_ms: int, params: EvalParams) -> dict[str, Any]:
    """搜索段整段当训练窗定权重，只在 cut 之后出信号算 IC；形状与 recheck 一样"""
    index = next(iter(combo.panels.values())).index.to_numpy(dtype="int64")
    start = int(np.searchsorted(index, cut_ms, side="right"))
    out: dict[str, Any] = {"since": cut_ms, "at": now_ms(), "horizons": {}}
    for bars in HORIZON_BARS.values():
        targets = combo.targets(bars)
        wins = [Window(start - bars - params.embargo, start, combo.rows)]
        signal = score(combo.values, targets, wins)
        out["horizons"][bars] = ic.compact(ic.summary(_points(combo, signal, targets, wins, bars),
                                                      params.segments, params.min_points))
    return out


def latest(combo: Combo, params: EvalParams) -> dict[str, Any]:
    """全部数据当训练窗：主档的权重，和每个币最后一根（全部币都有的最后一个时刻）的信号"""
    targets = combo.targets(params.primary)
    stats = {s: [TrainStats(v[:, i], targets[s]) for i in range(v.shape[1])] for s, v in combo.values.items()}
    means, stds, weights = _fit(stats, combo.rows)
    return {"asOf": int(next(iter(combo.panels.values())).index[-1]),
            "weights": [{"name": n, "weight": w} for n, w in zip(combo.names, weights.tolist())],
            "latest": {s: float(_combine(v[-1:], means[s], stds[s], weights)[0]) for s, v in combo.values.items()}}
