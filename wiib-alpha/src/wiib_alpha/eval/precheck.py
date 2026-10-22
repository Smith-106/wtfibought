"""评估前预检：只用因子值和信号时刻已知的列（当期资金费），不碰答案，算 24h 秩自相关（换手代理）、覆盖、因子值高低两头的当期资金费。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..data.pool import on_grid
from . import cs_ic
from .params import EvalParams


def precheck(rows: pd.DataFrame, f: np.ndarray, funding: np.ndarray, params: EvalParams) -> dict[str, Any]:
    """rows 是搜索段骨架（pool.search），f、funding（$funding_rate 的值）与它对齐，只看 24h 网格上的点。
    rankAutocorr：相邻两点同一批币截面分位的相关，跨点平均；coverage：每点因子有数的币占在池币的比例，跨点中位数；
    points：有数的币够 min_names 的点数；fundingHigh / fundingLow：因子值头部、尾部各 leg_enter 那么多币的当期资金费均值，
    跨点平均，bps。有数的币不够 min_names 的点都不算"""
    bars = HORIZON_BARS[24]
    t = rows["t"].to_numpy(dtype="int64")
    on = on_grid(t, bars)
    g = cs_ic.grid_index(t[on], bars, int(t[on].min()))
    total = int(g.max()) + 1
    sym = rows["sym"].to_numpy()[on]
    x = np.asarray(f, dtype="float64")[on]
    fund = np.asarray(funding, dtype="float64")[on]
    names = np.bincount(g[np.isfinite(x)], minlength=total)
    # 骨架上每点的在池币数；数据断档的点没有行
    pool = np.bincount(g, minlength=total)
    present = pool > 0

    # 每点内的截面分位 (0,1]；dense 第 g + 1 行放第 g 点，第 g 行就是上一点
    with np.errstate(invalid="ignore", divide="ignore"):
        q = cs_ic.ranks(g, x) / names[g]
    dense = np.full((total + 1, int(sym.max()) + 1), np.nan)
    dense[g + 1, sym] = q
    prev = dense[g, sym]
    pair = np.isfinite(q) & np.isfinite(prev)
    corr, n = cs_ic.corr_by(g[pair], q[pair], prev[pair], total)
    corr = corr[(n >= params.min_names) & np.isfinite(corr)]

    # 两头按因子值名次挑，和多空腿进场一样：并列按币序号先后排
    both = np.isfinite(x) & np.isfinite(fund)
    gb, fb = g[both], fund[both]
    m = np.bincount(gb, minlength=total)
    order = cs_ic.ranks(gb, x[both], method="first")
    enough = m >= params.min_names

    def side(pick: np.ndarray) -> float:
        """每点挑出来的币的资金费均值，跨点平均，bps"""
        mean = np.bincount(gb[pick], fb[pick], total)[enough] / np.bincount(gb[pick], minlength=total)[enough]
        return float(mean.mean() * 1e4) if len(mean) else np.nan

    return {
        "rankAutocorr": float(corr.mean()) if len(corr) else np.nan,
        "coverage": float(np.median(names[present] / pool[present])),
        "points": int((names >= params.min_names).sum()),
        "fundingHigh": side((m[gb] - order + 1) / m[gb] <= params.leg_enter),
        "fundingLow": side(order / m[gb] <= params.leg_enter),
    }
