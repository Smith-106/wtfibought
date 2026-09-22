"""替身检验：bar 按块打乱重排、价格链重新拼，同一条公式在"没有可预测性"的行情上算 IC 的分布。

月内 IC 对持久因子有系统偏差：随机游走上价格水平与未来收益的窗内相关天然约 −0.2，
长窗口均线比、区间位置之类都沾。替身保留公式的持久结构，只砍掉可预测性；观测 IC 得比替身分布极端才算数。

一个替身只抽一套块顺序，套到全部币上：7 个币同涨同跌的横截面相关必须留在替身里，
每币各打乱各的会把它毁掉，跨币平均后的替身分布比真零分布窄一倍多，假因子就能过门。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import HORIZON_BARS
from ..dsl import Expr
from . import ic
from .params import EvalParams
from .walkforward import windows

# 以价格计价的列跟着新价格链缩放
PRICE_SCALED = ("open", "high", "low", "quote_volume", "taker_buy_quote", "liq_long", "liq_short")


def block_order(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """bar 按 block 根一块换位置，第一块不动（首根的 NaN 收益留在原位）"""
    starts = np.arange(block, n, block)
    rng.shuffle(starts)
    return np.concatenate([np.arange(0, min(block, n))] + [np.arange(s, min(s + block, n)) for s in starts])


def surrogate_panel(panel: pd.DataFrame, order: np.ndarray) -> pd.DataFrame:
    """时间轴不动，行按 order 换位置；close 与 btc_close 按换位后的收益重新累出来"""
    df = panel.iloc[order].copy()
    df.index = panel.index
    scale = _rechain(df, "close", "ret", float(panel["close"].iloc[0]))
    for c in PRICE_SCALED:
        if c in df:
            df[c] = df[c].to_numpy(dtype="float64") * scale
    if "btc_close" in df and "btc_ret" in df and panel["btc_close"].notna().any():
        _rechain(df, "btc_close", "btc_ret", float(panel["btc_close"].dropna().iloc[0]))
    for bars in HORIZON_BARS.values():
        df[f"fwd_ret_{bars}"] = np.log(df["close"].shift(-bars) / df["close"])
    return df


def _rechain(df: pd.DataFrame, price_col: str, ret_col: str, start: float) -> np.ndarray:
    """按 ret 列累出新价格链，返回新旧价格之比"""
    ret = df[ret_col].to_numpy(dtype="float64")
    steps = np.where(np.isfinite(ret), ret, 0.0)
    new_price = start * np.exp(np.cumsum(steps))
    old_price = df[price_col].to_numpy(dtype="float64")
    df[price_col] = new_price
    df[ret_col] = np.log(new_price / np.concatenate([[np.nan], new_price[:-1]]))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(old_price > 0, new_price / old_price, 1.0)


def aligned(panels: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """只留全部币都有的时刻，同一套顺序才能套到每个币上"""
    common = None
    for p in panels.values():
        common = p.index if common is None else common.intersection(p.index)
    common = common.sort_values()
    return {s: p.loc[common] for s, p in panels.items()}


def null_ics(expr: Expr, panels: dict[str, pd.DataFrame], params: EvalParams, bars: int,
             k: int, seed: int) -> list[float]:
    """k 个替身各算一次主档的月度 IC 均值（跨币等权），口径与 evaluate 的 IC 完全一样"""
    if k <= 0 or not panels:
        return []
    rng = np.random.default_rng(seed)
    same = aligned(panels)
    n = len(next(iter(same.values())))
    wins = windows(n, params.test_size, bars, params.embargo, params.min_train)
    if not wins:
        return []
    out = []
    for _ in range(k):
        order = block_order(n, params.surrogate_block, rng)
        points = {}
        for s, p in same.items():
            sp = surrogate_panel(p, order)
            f = expr.evaluate(sp).to_numpy(dtype="float64")
            y = sp[f"fwd_ret_{bars}"].to_numpy(dtype="float64")
            points[s] = ic.oos_points(sp.index.to_numpy(dtype="int64"), f, y, wins, bars)
        out.append(ic.mean_monthly(points, params.min_points))
    return out


def pvalue(observed: float, nulls: list[float]) -> float:
    """双侧：|替身 IC| ≥ |观测| 的个数加一，除以替身数加一；分辨率是 1/(k+1)"""
    valid = [v for v in nulls if np.isfinite(v)]
    if not np.isfinite(observed) or not valid:
        return np.nan
    count = sum(1 for v in valid if abs(v) >= abs(observed))
    return (1 + count) / (len(valid) + 1)
