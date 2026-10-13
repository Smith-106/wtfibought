"""一条表达式的完整评估：前视自检 → 每档 h 的 IC 与方向腿 → 主档 IC 的替身检验 → 冗余 → 复杂度 → 对组合的增量。

输出是可直接落库/下发的 dict，判门在 gates.py。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..dsl import Expr
from . import combo, surrogate
from .horizon import horizon
from .params import EvalParams
from .stats import spearman


def evaluate(expr: Expr, panels: dict[str, pd.DataFrame], params: EvalParams = EvalParams(),
             references: dict[str, pd.DataFrame] | None = None, library: combo.Combo | None = None) -> dict[str, Any]:
    """panels 一币一张；references 一币一张，列是参照因子名；library 是库里因子的组合，不给就不算增量"""
    factors = {s: expr.evaluate(p) for s, p in panels.items()}
    out: dict[str, Any] = {
        "expr": str(expr),
        "canonical": expr.canonical(),
        "columns": sorted(expr.columns()),
        "complexity": expr.complexity(),
        "lookaheadFree": _lookahead_free(expr, panels, factors, params),
        "horizons": {},
    }
    for bars in HORIZON_BARS.values():
        out["horizons"][bars] = horizon(bars, panels, factors, params)
    main = out["horizons"][params.primary]["ic"]
    nulls = surrogate.null_ics(expr, panels, params, params.primary, params.surrogates, params.seed)
    valid = [v for v in nulls if np.isfinite(v)]
    main["surrogate"] = {
        "k": len(valid),
        "p": surrogate.pvalue(main["mean"], nulls),
        "nullMean": float(np.mean(valid)) if valid else np.nan,
        "nullStd": float(np.std(valid, ddof=1)) if len(valid) > 1 else np.nan,
        "ics": nulls,
    }
    out["redundancy"] = redundancy(factors, references, params.redundancy_stride)
    out["gain"] = combo.gain(library, str(expr), expr, params) if library is not None else None
    return out


def _lookahead_free(expr: Expr, panels: dict[str, pd.DataFrame], factors: dict[str, pd.Series],
                    params: EvalParams) -> bool:
    """截断点轮流落在每个币上：截到第 c 行重算，前 c 行必须与全量算出来的一样"""
    rng = np.random.default_rng(params.seed)
    symbols = list(panels)
    for i in range(params.lookahead_cuts):
        s = symbols[i % len(symbols)]
        n = len(panels[s])
        lo = max(10, n // 10)
        c = n if n <= lo + 1 else int(rng.integers(lo, n))
        part = expr.evaluate(panels[s].iloc[:c]).to_numpy(dtype="float64")
        full = factors[s].to_numpy(dtype="float64")[:c]
        if not np.allclose(part, full, rtol=1e-9, atol=1e-12, equal_nan=True):
            return False
    return True


def redundancy(factors: dict[str, pd.Series], references: dict[str, pd.DataFrame] | None,
               stride: int) -> dict[str, Any]:
    """与每个参照因子按币算 Spearman 取绝对值，跨币平均，取最大的那个"""
    if not references:
        return {"maxAbs": 0.0, "with": None}
    per_ref: dict[str, list[float]] = {}
    for s, f in factors.items():
        ref = references.get(s)
        if ref is None or ref.empty:
            continue
        cand = f.to_numpy(dtype="float64")[::stride]
        for name in ref.columns:
            value = spearman(ref[name].to_numpy(dtype="float64")[::stride], cand)
            if np.isfinite(value):
                per_ref.setdefault(str(name), []).append(abs(value))
    if not per_ref:
        return {"maxAbs": 0.0, "with": None}
    means = {name: float(np.mean(v)) for name, v in per_ref.items()}
    best = max(means, key=means.get)
    return {"maxAbs": means[best], "with": best}
