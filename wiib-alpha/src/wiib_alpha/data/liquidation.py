"""PG force_order 按 15m 桶聚合：多空被强平的名义额与笔数。

side=SELL 是多头被强平，其余是空头被强平，与前端 ForceOrders 页同口径。
trade_time 是不带时区的本地时间（Java 按容器 TZ 落库），换 UTC 靠 tz 参数。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import psycopg

from .binance import BAR_MS

COLUMNS = ["symbol", "close_time", "liq_long", "liq_short", "liq_count"]
# 全部币合起来连续这么多个 15m 桶一条强平都没有，只能是采集断了（2 小时）
OUTAGE_RUN = 8

# 桶的 key 是 bar 收盘时刻 = 桶序号 × 15m + 15m − 1ms，与 Binance kline 的 closeTime 同口径
_SQL = """
SELECT symbol,
       ((floor(extract(epoch FROM (trade_time AT TIME ZONE %(tz)s)) * 1000 / %(bar)s))::bigint + 1) * %(bar)s - 1
           AS close_time,
       sum(CASE WHEN side = 'SELL' THEN amount ELSE 0 END) AS liq_long,
       sum(CASE WHEN side <> 'SELL' THEN amount ELSE 0 END) AS liq_short,
       count(*) AS liq_count
FROM force_order
WHERE symbol = ANY(%(symbols)s)
GROUP BY symbol, close_time
ORDER BY symbol, close_time
"""


def load(conninfo: dict[str, Any], symbols: Iterable[str], tz: str) -> pd.DataFrame:
    with psycopg.connect(**conninfo) as conn, conn.cursor() as cur:
        cur.execute(_SQL, {"tz": tz, "bar": BAR_MS, "symbols": list(symbols)})
        rows = cur.fetchall()
    return frame(rows)


def frame(rows: list[tuple[Any, ...]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.astype({"close_time": "int64", "liq_long": "float64", "liq_short": "float64", "liq_count": "int64"})


def outage_buckets(liq: pd.DataFrame, min_run: int = OUTAGE_RUN) -> frozenset[int]:
    """全市场流每几秒必有强平；全部币合起来连续 min_run 个桶一条都没有算断档，返回断掉的桶 key（close_time）"""
    if liq.empty:
        return frozenset()
    present = np.unique(liq["close_time"].to_numpy(dtype="int64"))
    grid = np.arange(present.min(), present.max() + 1, BAR_MS)
    empty = ~np.isin(grid, present)
    out: list[int] = []
    start = None
    for i, is_empty in enumerate(empty):
        if is_empty and start is None:
            start = i
        if start is not None and (not is_empty or i == len(empty) - 1):
            end = i + 1 if is_empty else i
            if end - start >= min_run:
                out.extend(grid[start:end].tolist())
            start = None
    return frozenset(out)
