"""walk-forward 切窗与训练窗统计。

切窗与 Java WalkForwardEvaluator 同定义：训练窗从 0 起扩张，test 块连续前进互不重叠，
train 与 test 之间隔 purge(horizon_bars) + embargo，最后一个训练样本的标准答案够不到 test。
训练窗统计走前缀和，任意 train_end 都是 O(1)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Window:
    train_end: int
    test_start: int
    test_end: int


def windows(total: int, test_size: int, horizon_bars: int, embargo: int, min_train: int) -> list[Window]:
    out: list[Window] = []
    if test_size <= 0 or total <= 0:
        return out
    gap = horizon_bars + embargo
    test_start = min_train + gap
    while test_start + test_size <= total:
        train_end = test_start - gap
        if train_end >= min_train:
            out.append(Window(train_end, test_start, test_start + test_size))
        test_start += test_size
    return out


class TrainStats:
    """因子的均值/样本标准差，与因子-答案 Pearson 相关的符号；只用 [0, train_end) 里有数的行"""

    def __init__(self, factor: np.ndarray, target: np.ndarray):
        f = np.asarray(factor, dtype="float64")
        y = np.asarray(target, dtype="float64")
        ok_f = np.isfinite(f)
        ok = ok_f & np.isfinite(y)
        # 减掉第一个有效值再累加，避免大数相减丢精度；均值最后加回去，方差和相关不受影响
        self._base = float(f[ok_f][0]) if ok_f.any() else 0.0
        x = np.where(ok_f, f - self._base, 0.0)
        self._n = np.cumsum(ok_f)
        self._s = np.cumsum(x)
        self._s2 = np.cumsum(x * x)
        xp = np.where(ok, f - self._base, 0.0)
        yp = np.where(ok, y, 0.0)
        self._m = np.cumsum(ok)
        self._sx = np.cumsum(xp)
        self._sy = np.cumsum(yp)
        self._sxy = np.cumsum(xp * yp)
        self._sxx = np.cumsum(xp * xp)
        self._syy = np.cumsum(yp * yp)

    def at(self, train_end: int) -> tuple[float, float, int]:
        """(mean, std, sign)；样本不够或没波动时 std=NaN、sign=0"""
        if train_end <= 0:
            return np.nan, np.nan, 0
        i = min(train_end, len(self._n)) - 1
        n = int(self._n[i])
        if n < 2:
            return np.nan, np.nan, 0
        mean = self._s[i] / n + self._base
        var = (self._s2[i] - self._s[i] ** 2 / n) / (n - 1)
        std = float(np.sqrt(var)) if var > 0 else np.nan
        m = int(self._m[i])
        sign = 0
        if m >= 2:
            cov = self._sxy[i] - self._sx[i] * self._sy[i] / m
            vx = self._sxx[i] - self._sx[i] ** 2 / m
            vy = self._syy[i] - self._sy[i] ** 2 / m
            if vx > 0 and vy > 0:
                sign = int(np.sign(cov))
        return float(mean), std, sign
