"""一档 h 的截面 IC 与多空腿，单因子和组合共用。

切窗在 h 档网格点上切：test、min_train、embargo 按 bar 数换成点数，purge 是 h 根正好一个点。
IC 只在样本外（第一个 test 块到最后一个）算，和多空腿同一段。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from . import cs_ic, cs_leg
from .params import EvalParams
from .walkforward import Window, windows


def grid_windows(total: int, bars: int, params: EvalParams) -> list[Window]:
    test, embargo, min_train = params.grid(bars)
    return windows(total, test, 1, embargo, min_train)


def oos(rows: pd.DataFrame, f: np.ndarray, bars: int,
        params: EvalParams) -> tuple[cs_ic.Points, np.ndarray, list[Window], int, int]:
    """有效样本、每个网格点的 IC、切窗、样本外区间 [lo, hi)"""
    p = cs_ic.points(rows, f, bars, params.min_names)
    wins = grid_windows(p.total, bars, params)
    lo, hi = (wins[0].test_start, wins[-1].test_end) if wins else (0, 0)
    return p, cs_ic.series(p), wins, lo, hi


def horizon(rows: pd.DataFrame, f: np.ndarray, vol: np.ndarray, bars: int, params: EvalParams,
            iterations: int) -> dict[str, Any]:
    """rows 是骨架，f、vol（按波动分仓用的 Std($ret, 672)）与它对齐；iterations 是排列次数，0 = 不做。
    IC 用 f 有数的全部行，多空腿只用波动也有数的"""
    p, ic, wins, lo, hi = oos(rows, f, bars, params)
    leg_p, leg_vol = cs_leg.sized(rows, f, vol, bars, params.min_names)
    return {"bars": bars, "ic": cs_ic.summary(p, ic, lo, hi, params),
            "leg": cs_leg.simulate(leg_p, leg_vol, ic, wins, bars, params, iterations)}
