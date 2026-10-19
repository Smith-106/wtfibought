"""只用某个时刻之后的网格点算截面 IC：holdout（cut 之后）和入库后复查（入库之后）都是这一套。

因子用全量数据算（滚动窗口需要之前的历史），骨架是全量的 pool.rows；没有训练窗、没有门，只出 IC 数字。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..store import now_ms
from . import cs_ic
from .params import EvalParams


def after(rows: pd.DataFrame, f: np.ndarray, bars: int, since_ms: int, params: EvalParams) -> dict[str, Any]:
    """一档：since 之后的网格点"""
    p = cs_ic.points(rows, f, bars, params.min_names)
    lo = int(np.searchsorted(p.times, since_ms, side="right"))
    return cs_ic.compact(cs_ic.summary(p, cs_ic.series(p), lo, p.total, params))


def recheck(rows: pd.DataFrame, f: np.ndarray, since_ms: int, params: EvalParams) -> dict[str, Any]:
    return {"since": since_ms, "at": now_ms(),
            "horizons": {bars: after(rows, f, bars, since_ms, params) for bars in HORIZON_BARS.values()}}
