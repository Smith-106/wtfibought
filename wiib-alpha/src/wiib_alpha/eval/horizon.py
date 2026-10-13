"""一档 h 的样本外 IC 与方向腿，单因子和组合共用。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import ic, leg
from .params import EvalParams
from .walkforward import TrainStats, windows


def horizon(bars: int, panels: dict[str, pd.DataFrame], factors: dict[str, Any], params: EvalParams) -> dict[str, Any]:
    """factors 一币一条，与面板等长；每币各自切窗"""
    points: dict[str, pd.DataFrame] = {}
    legs: dict[str, leg.Leg] = {}
    leg_rows: dict[str, dict[str, Any]] = {}
    for s, panel in panels.items():
        f = np.asarray(factors[s], dtype="float64")
        y = panel[f"fwd_ret_{bars}"].to_numpy(dtype="float64")
        wins = windows(len(panel), params.test_size, bars, params.embargo, params.min_train)
        if not wins:
            continue
        points[s] = ic.oos_points(panel.index.to_numpy(dtype="int64"), f, y, wins, bars)
        simulated = leg.simulate(f, y, wins, TrainStats(f, y), bars, params.cost)
        if simulated is not None:
            legs[s] = simulated
            leg_rows[s] = leg.summarize(simulated, bars, params)
    return {
        "bars": bars,
        "ic": ic.summary(points, params.segments, params.min_points),
        "leg": {"pooled": leg.pool(legs, bars, params), "bySymbol": leg_rows},
    }
