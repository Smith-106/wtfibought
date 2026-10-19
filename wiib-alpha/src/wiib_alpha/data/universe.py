"""币池：每个月初按过去 30 天日均成交额选前 N 个，只用月初之前的数据，点时点选池。

符号表来自归档目录，含已下架的币，所以回测里有当年在池、后来下架的币，没有幸存者偏差。
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .binance import DAY_MS, ms, next_month, utc

WINDOW_DAYS = 30
# 上市不满这么多天不进池：刚上市那阵成交额虚高，滚动窗口也没历史可算
MIN_LISTED_DAYS = 90
# 30 天里至少要有这么多天的日线，不然是下架了或断了
MIN_WINDOW_DAYS = 28
COLUMNS = ["month", "symbol", "rank", "avg_quote_volume"]


def excluded(symbol: str, info: Mapping[str, Any] | None, exclude: Collection[str]) -> bool:
    """不是 USDT 结尾、在名单里、或 exchangeInfo 说它不是普通币（指数、代币化股票）"""
    if not symbol.endswith("USDT") or symbol in exclude:
        return True
    return info is not None and info.get("underlyingType", "COIN") != "COIN"


def month_starts(from_ms: int, to_ms: int) -> list[int]:
    """[from, to] 里的 UTC 月初"""
    cur = utc(from_ms).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if ms(cur) < from_ms:
        cur = next_month(cur)
    out = []
    while ms(cur) <= to_ms:
        out.append(ms(cur))
        cur = next_month(cur)
    return out


def rank(daily: Mapping[str, pd.DataFrame], months: Sequence[int], size: int, min_volume: float) -> pd.DataFrame:
    """每个月初一张榜：列 month（YYYY-MM）、symbol、rank、avg_quote_volume"""
    series = {s: (df["open_time"].to_numpy(dtype="int64"), df["quote_volume"].to_numpy(dtype="float64"))
              for s, df in daily.items() if not df.empty}
    records = []
    for m in months:
        rows = []
        for s, (t, v) in series.items():
            if t[0] > m - MIN_LISTED_DAYS * DAY_MS:
                continue
            a, b = np.searchsorted(t, m - WINDOW_DAYS * DAY_MS), np.searchsorted(t, m)
            if b - a < MIN_WINDOW_DAYS:
                continue
            avg = float(np.mean(v[a:b]))
            if avg >= min_volume:
                rows.append((s, avg))
        rows.sort(key=lambda r: -r[1])
        label = utc(m).strftime("%Y-%m")
        records += [(label, s, i + 1, avg) for i, (s, avg) in enumerate(rows[:size])]
    return pd.DataFrame(records, columns=COLUMNS)


def months_by_symbol(table: pd.DataFrame) -> dict[str, frozenset[str]]:
    return {s: frozenset(g["month"]) for s, g in table.groupby("symbol")}
