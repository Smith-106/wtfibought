from datetime import datetime, timezone

import numpy as np
import pandas as pd

from wiib_alpha.data import universe
from wiib_alpha.data.binance import DAY_MS


def ms(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp() * 1000)


def daily(first: int, last: int, volume: float) -> pd.DataFrame:
    """[first, last] 每天一根日线"""
    t = np.arange(first, last + 1, DAY_MS, dtype="int64")
    return pd.DataFrame({"open_time": t, "quote_volume": np.full(len(t), float(volume))})


def test_excluded_rules():
    assert universe.excluded("BTCUSDC", None, ())
    assert universe.excluded("USDCUSDT", None, ("USDCUSDT",))
    assert universe.excluded("TSLAUSDT", {"underlyingType": "STOCK"}, ())
    assert not universe.excluded("BTCUSDT", {"underlyingType": "COIN"}, ())
    # 下架的没有 exchangeInfo，只看名单
    assert not universe.excluded("DEADUSDT", None, ())


def test_month_starts():
    assert universe.month_starts(ms(2026, 1, 1), ms(2026, 3, 15)) == [ms(2026, 1, 1), ms(2026, 2, 1), ms(2026, 3, 1)]
    assert universe.month_starts(ms(2026, 1, 2), ms(2026, 2, 1)) == [ms(2026, 2, 1)]


def test_rank_is_point_in_time():
    months = universe.month_starts(ms(2026, 1, 1), ms(2026, 6, 1))
    data = {
        "BTCUSDT": daily(ms(2025, 1, 1), ms(2026, 6, 1), 5e9),
        "AAAUSDT": daily(ms(2025, 1, 1), ms(2026, 6, 1), 2e8),
        # 3 月 15 日下架：4 月起窗口里不够 28 天
        "DEADUSDT": daily(ms(2025, 1, 1), ms(2026, 3, 15), 3e8),
        # 3 月 1 日上市：6 月 1 日才满 90 天
        "NEWBUSDT": daily(ms(2026, 3, 1), ms(2026, 6, 1), 4e8),
        # 成交额不到底线
        "TINYUSDT": daily(ms(2025, 1, 1), ms(2026, 6, 1), 1e7),
    }
    table = universe.rank(data, months, size=2, min_volume=1e8)
    assert list(table.columns) == universe.COLUMNS
    by_month = {m: g.sort_values("rank")["symbol"].tolist() for m, g in table.groupby("month")}
    assert by_month["2026-01"] == ["BTCUSDT", "DEADUSDT"] and by_month["2026-03"] == ["BTCUSDT", "DEADUSDT"]
    assert by_month["2026-04"] == ["BTCUSDT", "AAAUSDT"] and by_month["2026-05"] == ["BTCUSDT", "AAAUSDT"]
    assert by_month["2026-06"] == ["BTCUSDT", "NEWBUSDT"]
    assert "TINYUSDT" not in set(table["symbol"])
    assert table[(table["month"] == "2026-01") & (table["symbol"] == "BTCUSDT")]["avg_quote_volume"].iloc[0] == 5e9
    assert universe.months_by_symbol(table)["DEADUSDT"] == frozenset({"2026-01", "2026-02", "2026-03"})
