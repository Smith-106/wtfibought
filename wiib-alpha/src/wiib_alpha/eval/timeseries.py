"""时序报告：7 个产品币各自按时间看因子与之后收益的关系（v1 口径：按币月度 IC 再跨币平均、|z| > 1 的方向腿）。

只给评审和页面看，不做门（截面才做门）。含截面算子的表达式用池子里算好的截面值，币不在池的时刻是 NaN。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..dsl import Cross, Expr
from . import ic, leg
from .params import EvalParams
from .walkforward import TrainStats, windows


def horizon(bars: int, panels: dict[str, pd.DataFrame], factors: dict[str, Any], params: EvalParams) -> dict[str, Any]:
    """factors 一币一条，与面板等长；每币各自切窗"""
    points: dict[str, pd.DataFrame] = {}
    legs: dict[str, leg.Leg] = {}
    for s, panel in panels.items():
        f = np.asarray(factors[s], dtype="float64")
        y = panel[f"fwd_ret_{bars}"].to_numpy(dtype="float64")
        wins = windows(len(panel), params.test_size, bars, params.embargo, params.min_train)
        if not wins:
            continue
        points[s] = ic.oos_points(panel.index.to_numpy(dtype="int64"), f, y, wins, bars)
        simulated = leg.simulate(f, y, wins, TrainStats(f, y), bars, params.ts_cost)
        if simulated is not None:
            legs[s] = simulated
    return {"ic": ic.summary(points, params.segments, params.min_points), "leg": leg.pool(legs, bars, params)}


def report(expr: Expr, panels: dict[str, pd.DataFrame], params: EvalParams,
           cross: Callable[[str], Cross]) -> dict[str, Any]:
    """三档的 IC 与方向腿净收益；cross 给每个币的截面节点值"""
    factors = {s: expr.evaluate(p, cross(s)) for s, p in panels.items()}
    out: dict[str, Any] = {"symbols": list(panels), "horizons": {}}
    for bars in HORIZON_BARS.values():
        h = horizon(bars, panels, factors, params)
        out["horizons"][bars] = {
            "mean": h["ic"]["mean"], "icir": h["ic"]["icir"], "months": h["ic"]["months"],
            "sameSignSymbols": h["ic"]["sameSignSymbols"], "bySymbol": h["ic"]["bySymbol"],
            "netReturn": h["leg"].get("netReturn"), "permPct": h["leg"].get("permPct"),
        }
    return out
