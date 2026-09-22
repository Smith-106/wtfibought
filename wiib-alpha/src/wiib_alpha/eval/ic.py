"""IC = Spearman(因子_t, 答案_{t+h})，只在样本外非重叠点上算。

先按币算再跨币等权平均，不把不同币混在一起排名：量纲不同（Amihud、成交额）会让排名变成"哪个币"而不是"哪个时刻"。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .stats import spearman
from .walkforward import Window

DAY_MS = 86_400_000


def oos_points(times: np.ndarray, factor: np.ndarray, target: np.ndarray,
               wins: list[Window], horizon_bars: int) -> pd.DataFrame:
    """样本外从第一个 test 起每 h 根取一个点，两边都有数的留下；列 t / f / y"""
    if not wins:
        return pd.DataFrame({"t": np.array([], dtype="int64"), "f": np.array([]), "y": np.array([])})
    rows = np.arange(wins[0].test_start, wins[-1].test_end, horizon_bars)
    f, y = factor[rows], target[rows]
    ok = np.isfinite(f) & np.isfinite(y)
    return pd.DataFrame({"t": np.asarray(times)[rows][ok], "f": f[ok], "y": y[ok]})


def monthly(points: dict[str, pd.DataFrame], min_points: int) -> pd.DataFrame:
    """每币每月一个 IC；列 symbol / month / ic"""
    records = []
    for symbol, p in points.items():
        if p.empty:
            continue
        month = pd.to_datetime(p["t"], unit="ms", utc=True).dt.strftime("%Y-%m")
        for key, g in p.groupby(month):
            if len(g) >= min_points:
                records.append((symbol, key, spearman(g["f"].to_numpy(), g["y"].to_numpy())))
    return pd.DataFrame(records, columns=["symbol", "month", "ic"])


def pooled(table: pd.DataFrame) -> pd.Series:
    """跨币等权平均后的月度 IC 序列"""
    if table.empty:
        return pd.Series(dtype="float64")
    return table.dropna().groupby("month")["ic"].mean().sort_index()


def mean_monthly(points: dict[str, pd.DataFrame], min_points: int) -> float:
    """跨币等权平均后的月度 IC 均值；替身检验只要这一个数"""
    series = pooled(monthly(points, min_points))
    return float(series.mean()) if len(series) else np.nan


def by_symbol(points: dict[str, pd.DataFrame]) -> dict[str, float]:
    return {s: spearman(p["f"].to_numpy(), p["y"].to_numpy()) for s, p in points.items() if not p.empty}


def segments(points: dict[str, pd.DataFrame], k: int, min_points: int) -> list[float]:
    """OOS 时间范围等分 k 段，每段先按币算 IC 再跨币平均"""
    frames = [p for p in points.values() if not p.empty]
    if not frames:
        return [np.nan] * k
    lo = min(int(p["t"].min()) for p in frames)
    hi = max(int(p["t"].max()) for p in frames) + 1
    edges = np.linspace(lo, hi, k + 1)
    out = []
    for i in range(k):
        values = []
        for p in frames:
            g = p[(p["t"] >= edges[i]) & (p["t"] < edges[i + 1])]
            if len(g) >= min_points:
                values.append(spearman(g["f"].to_numpy(), g["y"].to_numpy()))
        out.append(float(np.nanmean(values)) if values and np.isfinite(values).any() else np.nan)
    return out


def summary(points: dict[str, pd.DataFrame], k: int, min_points: int) -> dict[str, Any]:
    table = monthly(points, min_points)
    series = pooled(table)
    mean = float(series.mean()) if len(series) else np.nan
    std = float(series.std(ddof=1)) if len(series) > 1 else np.nan
    per_symbol = by_symbol(points)
    same_sign = sum(1 for v in per_symbol.values() if np.isfinite(v) and np.isfinite(mean) and np.sign(v) == np.sign(mean))
    return {
        "mean": mean,
        "icir": float(mean / std) if np.isfinite(std) and std > 0 else np.nan,
        "months": int(len(series)),
        "monthly": {m: float(v) for m, v in series.items()},
        "bySymbol": per_symbol,
        "sameSignSymbols": same_sign,
        "segments": segments(points, k, min_points),
        "samples": int(sum(len(p) for p in points.values())),
    }
