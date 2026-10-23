import os
from decimal import Decimal

import pandas as pd
import pytest

from wiib_alpha.data import liquidation as liq


def test_frame_types():
    df = liq.frame([("BTCUSDT", Decimal("899999"), Decimal("12.5"), Decimal("0"), 3)])
    assert df["close_time"].dtype == "int64"
    assert df["liq_long"].iloc[0] == 12.5 and df["liq_count"].dtype == "int64"


def bucket(k: int) -> int:
    return (k + 1) * liq.BAR_MS - 1


def test_outage_buckets_need_a_long_enough_empty_run():
    # 桶 0..2 有记录，3..5 空（3 个，不算断），6 有，7..14 空（8 个，算断），15 有
    present = [0, 1, 2, 6, 15]
    df = pd.DataFrame({"symbol": ["A", "B", "A", "B", "A"], "close_time": [bucket(k) for k in present],
                       "liq_long": 1.0, "liq_short": 0.0, "liq_count": 1})
    assert liq.outage_buckets(df) == frozenset(bucket(k) for k in range(7, 15))
    assert liq.outage_buckets(df, min_run=3) == frozenset(bucket(k) for k in [3, 4, 5, *range(7, 15)])
    assert liq.outage_buckets(df.iloc[:0]) == frozenset()


def test_outage_run_at_the_end_is_not_counted():
    # 最后一条记录之后没有桶可判，尾部不算断
    df = pd.DataFrame({"symbol": ["A", "A"], "close_time": [bucket(0), bucket(20)],
                       "liq_long": 1.0, "liq_short": 0.0, "liq_count": 1})
    assert liq.outage_buckets(df) == frozenset(bucket(k) for k in range(1, 20))


@pytest.mark.skipif(os.getenv("WIIB_REAL_RUN") != "1", reason="要连真 PG")
def test_real_load():
    from wiib_alpha.config import config
    df = liq.load(config().pg_conninfo, ("BTCUSDT",), config().force_order_tz)
    assert not df.empty
    # 桶 key 落在 15m 边界减 1ms 上
    assert ((df["close_time"] + 1) % liq.BAR_MS == 0).all()
