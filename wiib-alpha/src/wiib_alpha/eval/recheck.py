"""只用某个时刻之后的数据算 IC：holdout（cut 之后）和入库后复查（入库之后）都是这一套。

因子用全量数据算（滚动窗口需要之前的历史），再截到时刻之后；没有训练窗、没有门，只出 IC 数字。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..dsl import Expr
from ..store import now_ms
from . import ic
from .params import EvalParams


def recheck(expr: Expr, panels: dict[str, pd.DataFrame], since_ms: int, params: EvalParams) -> dict[str, Any]:
    out: dict[str, Any] = {"since": since_ms, "at": now_ms(), "horizons": {}}
    factors = {s: expr.evaluate(p) for s, p in panels.items()}
    for bars in HORIZON_BARS.values():
        points: dict[str, pd.DataFrame] = {}
        for s, p in panels.items():
            after = p.index.to_numpy(dtype="int64") > since_ms
            if after.sum() <= bars:
                continue
            f = factors[s].to_numpy(dtype="float64")[after]
            y = p[f"fwd_ret_{bars}"].to_numpy(dtype="float64")[after]
            times = p.index.to_numpy(dtype="int64")[after]
            rows = np.arange(0, len(f), bars)
            ok = np.isfinite(f[rows]) & np.isfinite(y[rows])
            points[s] = pd.DataFrame({"t": times[rows][ok], "f": f[rows][ok], "y": y[rows][ok]})
        out["horizons"][bars] = ic.compact(ic.summary(points, params.segments, params.min_points))
    return out
