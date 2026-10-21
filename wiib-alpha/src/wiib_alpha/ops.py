"""DSL 算子的 pandas 实现。

时序算子全部只看 [t−w, t]（含当前 bar），窗口不满或窗内有 NaN 就是 NaN；输入一律是对齐好的 float Series，输出同 index。
EMA 是例外：从序列起点递推、记着全部历史，之前的 NaN 跳过去不断链，只有当前输入是 NaN 才出 NaN。
截面算子只看同一时刻：输入是全部在池币拼起来的长 Series 和每行的时刻，按时刻分组算。
Std / Var / Cov 用样本口径（ddof=1），与 Java FactorMath 同。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


@dataclass(frozen=True)
class Op:
    # unary(x) binary(x,y) ternary(c,a,b) rolling(x,w) quantile(x,w,q) pair(x,y,w) cross(x)
    kind: str
    fn: Callable[..., pd.Series]


# 分块算窗口时临时数组的上限；(n, w) 一次实体化在 7 万行 × 2016 窗上要 1 GB 多
_CHUNK_BYTES = 16 << 20


def _windows(x: pd.Series, w: int) -> np.ndarray:
    """(n, w) 滑动窗口视图，第 i 行是 [i−w+1, i]，前面不够的用 NaN 垫；只能拿去做 @ 和 reduce，别实体化"""
    values = np.concatenate([np.full(w - 1, np.nan), x.to_numpy(dtype="float64")])
    return sliding_window_view(values, w)


def _chunked(x: pd.Series, w: int, fn) -> pd.Series:
    """fn 吃 (m, w) 出 (m,)；按块喂，实体化的临时数组不超过 _CHUNK_BYTES"""
    values = np.concatenate([np.full(w - 1, np.nan), x.to_numpy(dtype="float64")])
    n = len(x)
    step = max(256, _CHUNK_BYTES // (w * 8))
    out = np.empty(n, dtype="float64")
    for start in range(0, n, step):
        stop = min(n, start + step)
        out[start:stop] = fn(sliding_window_view(values[start:stop + w - 1], w))
    return _wrap(out, x)


def _centered_t(w: int) -> np.ndarray:
    """0..w−1 减去均值，Σt = 0，回归斜率就是 Σ t·x / Σ t²"""
    return np.arange(w, dtype="float64") - (w - 1) / 2


def _wrap(values: np.ndarray, like: pd.Series) -> pd.Series:
    return pd.Series(values, index=like.index, dtype="float64")


# ========== 一元 ==========


def _abs(x: pd.Series) -> pd.Series:
    return x.abs()


def _sign(x: pd.Series) -> pd.Series:
    return _wrap(np.sign(x.to_numpy(dtype="float64")), x)


def _log(x: pd.Series) -> pd.Series:
    """非正数是 NaN"""
    return _wrap(np.log(x.where(x > 0).to_numpy(dtype="float64")), x)


# ========== 二元 / 三元 ==========


def _pow(x: pd.Series, y: pd.Series) -> pd.Series:
    with np.errstate(all="ignore"):
        return _wrap(np.power(x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")), x)


def _greater(x: pd.Series, y: pd.Series) -> pd.Series:
    return _wrap(np.maximum(x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")), x)


def _less(x: pd.Series, y: pd.Series) -> pd.Series:
    return _wrap(np.minimum(x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")), x)


def _gt(x: pd.Series, y: pd.Series) -> pd.Series:
    """1/0，任一边 NaN 就 NaN"""
    return (x > y).astype("float64").where(x.notna() & y.notna())


def _lt(x: pd.Series, y: pd.Series) -> pd.Series:
    return (x < y).astype("float64").where(x.notna() & y.notna())


def _if(c: pd.Series, a: pd.Series, b: pd.Series) -> pd.Series:
    """c 非零取 a 否则取 b；c 是 NaN 就 NaN"""
    return a.where(c != 0, b).where(c.notna())


# ========== 滚动 ==========


def _ref(x: pd.Series, n: int) -> pd.Series:
    return x.shift(n)


def _delta(x: pd.Series, n: int) -> pd.Series:
    return x - x.shift(n)


def _mean(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).mean()


def _sum(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).sum()


def _std(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).std()


def _var(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).var()


def _max(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).max()


def _min(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).min()


def _med(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).median()


def _mad(x: pd.Series, w: int) -> pd.Series:
    """对窗口均值的平均绝对偏差"""
    return _chunked(x, w, lambda win: np.abs(win - win.mean(axis=1, keepdims=True)).mean(axis=1))


def _rank(x: pd.Series, w: int) -> pd.Series:
    """当前值在窗内的平均秩 / w，取值 (0, 1]，并列取平均秩"""
    return x.rolling(w).rank(pct=True)


def _skew(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).skew()


def _kurt(x: pd.Series, w: int) -> pd.Series:
    """超额峰度"""
    return x.rolling(w).kurt()


def _ema(x: pd.Series, w: int) -> pd.Series:
    """span = w 的指数均线，从序列起点递推；pandas 在 NaN 处会把上一个值带过去，这里按算子约定出 NaN"""
    return x.ewm(span=w, adjust=False, min_periods=w).mean().where(x.notna())


def _wma(x: pd.Series, w: int) -> pd.Series:
    """线性加权，最新一根权重 w，最旧一根权重 1"""
    weights = np.arange(1, w + 1, dtype="float64")
    return _wrap(_windows(x, w) @ weights / weights.sum(), x)


def _quantile(x: pd.Series, w: int, q: float) -> pd.Series:
    return x.rolling(w).quantile(q)


def _idxmax(x: pd.Series, w: int) -> pd.Series:
    """窗内最大值离现在几根 bar：0 = 就是当前这根；并列取最旧的那根"""
    def distance(win: np.ndarray) -> np.ndarray:
        dist = (w - 1 - np.argmax(win, axis=1)).astype("float64")
        dist[np.isnan(win).any(axis=1)] = np.nan
        return dist
    return _chunked(x, w, distance)


def _idxmin(x: pd.Series, w: int) -> pd.Series:
    def distance(win: np.ndarray) -> np.ndarray:
        dist = (w - 1 - np.argmin(win, axis=1)).astype("float64")
        dist[np.isnan(win).any(axis=1)] = np.nan
        return dist
    return _chunked(x, w, distance)


def _slope(x: pd.Series, w: int) -> pd.Series:
    """窗内对 bar 序号做最小二乘的斜率"""
    t = _centered_t(w)
    return _wrap(_windows(x, w) @ t / (t @ t), x)


def _rsquare(x: pd.Series, w: int) -> pd.Series:
    """窗内对 bar 序号回归的 R² = (Σ x·t_c)² / (Σ x_c² · Σ t_c²)；t 居中后 Σ x·t_c 就是 Σ x_c·t_c，窗内没波动是 NaN"""
    t = _centered_t(w)
    cov = _windows(x, w) @ t
    ss = x.rolling(w).var().to_numpy(dtype="float64") * (w - 1)
    with np.errstate(all="ignore"):
        return _wrap(cov ** 2 / (ss * (t @ t)), x)


def _resi(x: pd.Series, w: int) -> pd.Series:
    """当前点相对窗内回归线的残差"""
    win = _windows(x, w)
    t = _centered_t(w)
    slope = win @ t / (t @ t)
    fitted = win.mean(axis=1) + slope * ((w - 1) / 2)
    return _wrap(win[:, -1] - fitted, x)


# ========== 成对滚动 ==========


def _corr(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).corr(y)


def _cov(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    return x.rolling(w).cov(y)


# ========== 截面：x 是全部在池币的长 Series，t 是每行的时刻 ==========


def _cs_rank(x: pd.Series, t: np.ndarray) -> pd.Series:
    """同一时刻有数的币里的平均秩 / 币数，取值 (0, 1]"""
    return x.groupby(t).rank(pct=True)


def _cs_demean(x: pd.Series, t: np.ndarray) -> pd.Series:
    return x - x.groupby(t).transform("mean")


def _cs_z(x: pd.Series, t: np.ndarray) -> pd.Series:
    """减截面均值除截面样本标准差；那一刻只有一个币或全一样是 NaN"""
    g = x.groupby(t)
    with np.errstate(all="ignore"):
        return (x - g.transform("mean")) / g.transform("std")


OPERATORS: dict[str, Op] = {
    "Abs": Op("unary", _abs),
    "Sign": Op("unary", _sign),
    "Log": Op("unary", _log),
    "Pow": Op("binary", _pow),
    "Greater": Op("binary", _greater),
    "Less": Op("binary", _less),
    "Gt": Op("binary", _gt),
    "Lt": Op("binary", _lt),
    "If": Op("ternary", _if),
    "Ref": Op("rolling", _ref),
    "Delta": Op("rolling", _delta),
    "Mean": Op("rolling", _mean),
    "Sum": Op("rolling", _sum),
    "Std": Op("rolling", _std),
    "Var": Op("rolling", _var),
    "Max": Op("rolling", _max),
    "Min": Op("rolling", _min),
    "Med": Op("rolling", _med),
    "Mad": Op("rolling", _mad),
    "Rank": Op("rolling", _rank),
    "Skew": Op("rolling", _skew),
    "Kurt": Op("rolling", _kurt),
    "EMA": Op("rolling", _ema),
    "WMA": Op("rolling", _wma),
    "IdxMax": Op("rolling", _idxmax),
    "IdxMin": Op("rolling", _idxmin),
    "Slope": Op("rolling", _slope),
    "Rsquare": Op("rolling", _rsquare),
    "Resi": Op("rolling", _resi),
    "Quantile": Op("quantile", _quantile),
    "Corr": Op("pair", _corr),
    "Cov": Op("pair", _cov),
    "CsRank": Op("cross", _cs_rank),
    "CsDemean": Op("cross", _cs_demean),
    "CsZ": Op("cross", _cs_z),
}
