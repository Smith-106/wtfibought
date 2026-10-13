import numpy as np
import pandas as pd

from wiib_alpha.data import panel
from wiib_alpha.data.binance import BAR_MS, KLINE_COLUMNS

DAY = panel.DAY_MS
N = 3 * 96  # 三天


def klines(n: int, base: float) -> pd.DataFrame:
    rows = [[k * BAR_MS, base, base + 1, base - 1, base + k, 10.0, (k + 1) * BAR_MS - 1, 1000.0, 7, 4.0, 400.0]
            for k in range(n)]
    return pd.DataFrame(rows, columns=KLINE_COLUMNS)


def close_time(k: int) -> int:
    return (k + 1) * BAR_MS - 1


def build(with_liq: bool = True) -> pd.DataFrame:
    alt, btc = klines(N, 100.0), klines(N, 1000.0)
    funding = pd.DataFrame({"funding_time": [close_time(2), close_time(50)], "funding_rate": [0.0001, -0.0002]})
    if not with_liq:
        return panel.assemble(alt, funding, btc)
    # 第 0 天有强平、第 1 天整天的桶都判成断档、第 2 天又有
    liq = pd.DataFrame({"symbol": ["X"] * 3, "close_time": [close_time(3), close_time(5), close_time(2 * 96 + 1)],
                        "liq_long": [5.0, 0.0, 1.0], "liq_short": [0.0, 2.0, 0.0], "liq_count": [1, 1, 1]})
    outage = frozenset(close_time(k) for k in range(96, 192))
    return panel.assemble(alt, funding, btc, liq, liq_start=close_time(3), outage=outage)


def test_index_and_returns():
    df = build(with_liq=False)
    assert df.index.name == "close_time" and df.index[0] == close_time(0)
    assert np.isnan(df["ret"].iloc[0])
    assert df["ret"].iloc[1] == np.log(101.0 / 100.0)
    # bar 开始时刻的小时：00:45~01:00 那根还是 0 点，23:45~24:00 是 23 点
    assert df["hour"].iloc[0] == 0 and df["hour"].iloc[3] == 0 and df["hour"].iloc[4] == 1 and df["hour"].iloc[95] == 23
    assert panel.describe(df)["gaps"] == 0


def test_funding_is_last_settled_at_close():
    df = build(with_liq=False)
    assert np.isnan(df["funding_rate"].iloc[1])
    assert df["funding_rate"].iloc[2] == 0.0001 and df["funding_rate"].iloc[49] == 0.0001
    assert df["funding_rate"].iloc[50] == -0.0002
    # 离上次结算超过 12 小时还没新的，是数据断了，不再硬填
    assert df["funding_rate"].iloc[98] == -0.0002 and np.isnan(df["funding_rate"].iloc[99])


def test_btc_aligned_by_close_time():
    df = build(with_liq=False)
    assert df["btc_close"].iloc[4] == 1004.0
    assert df["btc_ret"].iloc[4] == np.log(1004.0 / 1003.0)


def test_targets_look_forward_only():
    df = build(with_liq=False)
    assert df["fwd_ret_24"].iloc[0] == np.log(124.0 / 100.0)
    assert df["fwd_ret_96"].iloc[N - 97] == np.log((100.0 + N - 1) / (100.0 + N - 97))
    assert df["fwd_ret_96"].iloc[N - 96:].isna().all()


def test_liquidation_zero_vs_nan():
    df = build()
    assert np.isnan(df["liq_long"].iloc[2])          # 数据开始前
    assert df["liq_long"].iloc[3] == 5.0 and df["liq_short"].iloc[5] == 2.0
    assert df["liq_long"].iloc[4] == 0.0 and df["liq_count"].iloc[4] == 0.0   # 有数据的日子没强平
    assert df["liq_count"].iloc[96:192].isna().all()  # 断掉的那天
    assert df["liq_long"].iloc[2 * 96 + 1] == 1.0 and df["liq_long"].iloc[2 * 96 + 2] == 0.0


def test_no_liq_source_is_all_nan():
    df = build(with_liq=False)
    assert df[panel.LIQ_COLUMNS].isna().all().all()


def test_describe_counts_index_gaps():
    df = build(with_liq=False)
    holed = df.drop(df.index[[10, 11, 50]])
    assert panel.describe(holed)["gaps"] == 2


def test_holdout_cut_and_truncate():
    df = build(with_liq=False)
    cut = panel.holdout_cut({"X": df, "Y": df.iloc[:-5]}, 1)
    assert cut == close_time(N - 1) - DAY
    search = panel.truncate(df, cut)
    assert search.index.max() <= cut and len(search) == 2 * 96
    # 搜索段末尾的答案在截断处重算：最后 96 根的 24h 答案看不到 holdout 的价格
    assert search["fwd_ret_96"].iloc[-96:].isna().all() and df["fwd_ret_96"].iloc[len(search) - 1] > 0
    assert search["fwd_ret_24"].iloc[0] == df["fwd_ret_24"].iloc[0]
    assert set(search.columns) == set(df.columns)


def test_feature_columns_exist_and_targets_excluded():
    df = build()
    assert set(panel.FEATURE_COLUMNS) <= set(df.columns)
    assert not set(panel.TARGET_COLUMNS) & set(panel.FEATURE_COLUMNS)


class StubBinance:
    def klines(self, symbol, start, end):
        return klines(N, 100.0)

    def funding(self, symbol, start, end):
        return pd.DataFrame({"funding_time": [0], "funding_rate": [0.0001]})


def test_benchmark_panel_has_no_btc_columns(tmp_path):
    store = panel.PanelStore(tmp_path)
    store.refresh(StubBinance(), ("BTCUSDT", "ETHUSDT"), history_days=3, now_ms=N * BAR_MS)
    assert store.load("BTCUSDT")[["btc_close", "btc_ret"]].isna().all().all()
    assert store.load("ETHUSDT")["btc_close"].notna().sum() == N


def test_store_grows_incrementally(tmp_path):
    calls = []

    class FakeBinance:
        def klines(self, symbol, start, end):
            calls.append((symbol, start))
            k0 = start // BAR_MS
            return klines(N, 100.0).iloc[k0:] if k0 < N else klines(0, 100.0)

        def funding(self, symbol, start, end):
            return pd.DataFrame({"funding_time": [0], "funding_rate": [0.0001]}) if start == 0 else \
                pd.DataFrame({"funding_time": pd.Series(dtype="int64"), "funding_rate": pd.Series(dtype="float64")})

    store = panel.PanelStore(tmp_path)
    now = N * BAR_MS
    status = store.refresh(FakeBinance(), ("ETHUSDT",), history_days=3, now_ms=now)
    assert status["ETHUSDT"]["rows"] == N and status["ETHUSDT"]["liqFrom"] is None
    assert calls == [("ETHUSDT", 0), ("BTCUSDT", 0)]

    store.refresh(FakeBinance(), ("ETHUSDT",), history_days=3, now_ms=now)
    # 第二次从最后一根之后接着拉
    assert calls[2] == ("ETHUSDT", close_time(N - 1) + 1)
    assert len(store.load("ETHUSDT")) == N
    assert store.status()["ETHUSDT"]["to"] == close_time(N - 1)
