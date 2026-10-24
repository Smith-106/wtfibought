import numpy as np
import pandas as pd
import pytest

from wiib_alpha.data.binance import BAR_MS
from wiib_alpha.data.panel import PanelStore
from wiib_alpha.data.pool import Pool, forward_funding, month_label, month_of, on_grid, tier_of
from wiib_alpha.dsl import parse

# 2026-01-30 00:00 UTC：四天跨 1 月和 2 月
BASE = 1_769_731_200_000
N = 4 * 96
COLS = ["volume", "close"]
HOUR = 3_600_000
# A 的结算记录：8 点那次晚了 2 ms（真数据常这样），16 点那次正点
FUNDING_A = pd.DataFrame({"funding_time": [BASE + 8 * HOUR + 2, BASE + 16 * HOUR], "funding_rate": [0.001, 0.002]})


def close_time(k: int) -> int:
    return BASE + (k + 1) * BAR_MS - 1


def panel(seed: int, in_pool: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, N)))
    df = pd.DataFrame({"volume": rng.uniform(1, 10, N), "close": close, "in_pool": in_pool},
                      index=pd.Index([close_time(k) for k in range(N)], name="close_time"))
    for h in (24, 48, 96):
        df[f"fwd_ret_{h}"] = np.log(df["close"].shift(-h) / df["close"])
    return df


UNIVERSE = pd.DataFrame([("2026-01", "A", 1, 9.0), ("2026-01", "B", 30, 8.0),
                         ("2026-02", "A", 1, 9.0), ("2026-02", "B", 30, 8.0), ("2026-02", "C", 60, 7.0),
                         ("2026-02", "GONE", 2, 9.0)], columns=["month", "symbol", "rank", "avg_quote_volume"])


@pytest.fixture
def pool(tmp_path):
    """A 全程在池排第 1；B 全程在池排第 30；C 只有 2 月在池排第 60。按信号那根收盘成交，晚一根的见最后一条"""
    store = PanelStore(tmp_path)
    feb = np.array([pd.Timestamp(close_time(k), unit="ms").month == 2 for k in range(N)], dtype="float64")
    for i, (s, mask) in enumerate((("A", np.ones(N)), ("B", np.ones(N)), ("C", feb))):
        store.save_panel(s, panel(i, mask))
    store.save_raw("funding", "A", FUNDING_A)
    return Pool.load(store, UNIVERSE, cut_ms=close_time(3 * 96 - 1), delay=0)


def test_grid_month_and_tier_helpers():
    t = np.array([close_time(k) for k in range(N)])
    # 6h 网格每 24 根一个点，12h、24h 是它的子集；边界是 UTC 0 点起数
    assert np.flatnonzero(on_grid(t, 24)).tolist()[:3] == [23, 47, 71]
    assert set(np.flatnonzero(on_grid(t, 96))) <= set(np.flatnonzero(on_grid(t, 48))) <= set(np.flatnonzero(on_grid(t, 24)))
    # 1 月最后一根的收盘时刻还在 1 月，与面板 in_pool 同口径
    assert month_label(month_of(np.array([close_time(2 * 96 - 1)]))[0]) == "2026-01"
    assert month_label(month_of(np.array([close_time(2 * 96)]))[0]) == "2026-02"
    assert tier_of(np.array([1, 20, 21, 50, 51, 80])).tolist() == [1, 1, 2, 2, 3, 3]


def test_listed_is_the_first_bar_except_for_coins_present_at_the_data_start(tmp_path):
    store = PanelStore(tmp_path)
    store.save_panel("A", panel(0, np.ones(N)))
    store.save_panel("B", panel(1, np.ones(N)))
    # D 比别的币晚三天才有数据（数据起点后两天内有数的算起点就在）
    store.save_panel("D", panel(2, np.ones(N)).iloc[3 * 96:])
    universe = pd.DataFrame([(m, s, i + 1, 9.0) for m in ("2026-01", "2026-02") for i, s in enumerate(("A", "B", "D"))],
                            columns=UNIVERSE.columns)
    p = Pool.load(store, universe, cut_ms=close_time(N - 1), delay=0)
    assert list(p.listed) == [0, 0, close_time(3 * 96)]
    age = p.age_days(p.rows)
    d = p.rows["sym"].to_numpy() == 2
    assert age[~d].min() > 10_000
    assert np.allclose(age[d], (p.rows["t"].to_numpy()[d] - close_time(3 * 96)) / 86_400_000)


def test_rows_are_in_pool_grid_points_with_tiers(pool):
    # 面板不在的币不进；sym 是 symbols 里的序号
    assert pool.symbols == ("A", "B", "C")
    rows = pool.rows
    counts = rows.groupby("sym").size().to_dict()
    assert counts == {0: 16, 1: 16, 2: 8}
    assert on_grid(rows["t"].to_numpy(), 24).all()
    assert set(rows.loc[rows["sym"] == 1, "tier"]) == {2} and set(rows.loc[rows["sym"] == 2, "tier"]) == {3}
    assert set(rows.loc[rows["sym"] == 0, "tier"]) == {1}
    a = pool.store.load("A")
    first = rows[rows["sym"] == 0].iloc[0]
    assert first["t"] == close_time(23) and first["y48"] == pytest.approx(a["fwd_ret_48"].loc[close_time(23)])
    assert pool.describe() == {"symbols": 3, "months": 2, "gridPoints": 16, "rows": 40}


def test_forward_funding_counts_settlements_inside_the_holding_window():
    # 8 点前最后一根收盘建仓：6h 碰上 8 点那次，12h 两次都碰上；8 点那根收盘才建仓的碰不上 8 点那次
    t = np.array([close_time(31), close_time(32)])
    assert forward_funding(t, FUNDING_A, 24).tolist() == pytest.approx([0.001, 0.0])
    assert forward_funding(t, FUNDING_A, 48).tolist() == pytest.approx([0.003, 0.002])
    assert forward_funding(t, None, 48).tolist() == [0.0, 0.0]


def test_rows_carry_forward_funding(pool):
    rows = pool.rows
    first = rows[rows["sym"] == 0].iloc[0]
    # 6 点前最后一根收盘：6h 窗口 (6, 12] 点碰上 8 点那次，24h 窗口两次都碰上；B 没有结算记录是 0（够到数据末尾之后的是 NaN）
    assert first["t"] == close_time(23)
    assert (first["fund24"], first["fund48"], first["fund96"]) == pytest.approx((0.001, 0.003, 0.003))
    b = rows.loc[rows["sym"] == 1, ["fund24", "fund48", "fund96"]]
    assert ((b == 0) | b.isna()).all().all() and (b.iloc[0] == 0).all()


def test_search_drops_holdout_and_masks_answers_that_reach_it(pool):
    search = pool.search
    assert (search["t"] <= pool.cut_ms).all() and len(search) == int(pool.in_search.sum())
    # cut 前 6h 那个网格点：6h 的答案正好收在 cut，12h、24h 的够到 cut 之后
    t = close_time(3 * 96 - 25)
    assert t + 24 * BAR_MS == pool.cut_ms
    row = search[search["t"] == t]
    assert row["y24"].notna().all() and row["y48"].isna().all() and row["y96"].isna().all()
    assert row["fund24"].notna().all() and row["fund48"].isna().all() and row["fund96"].isna().all()
    # 全量骨架里同一行的答案是有的
    full = pool.rows[pool.in_search]
    assert full.loc[full["t"] == t, "y48"].notna().all()


def test_answers_settle_at_the_last_close_when_a_coin_stops(tmp_path):
    # D 在 [200, 260) 断档、300 之后下架；A 照常到最后
    store = PanelStore(tmp_path)
    a, d = panel(0, np.ones(N)), panel(3, np.ones(N)).iloc[np.r_[0:200, 260:301]]
    store.save_panel("A", a)
    store.save_panel("D", d)
    universe = pd.DataFrame([(m, s, r, 9.0) for m in ("2026-01", "2026-02") for s, r in (("A", 1), ("D", 2))],
                            columns=["month", "symbol", "rank", "avg_quote_volume"])
    rows = Pool.load(store, universe, cut_ms=close_time(N - 1), delay=0).rows
    ys = ["y24", "y48", "y96"]
    got = rows[rows["sym"] == 1].set_index("t")[ys]
    log = lambda frame, k, at: np.log(frame["close"].loc[close_time(k)] / frame["close"].loc[close_time(at)])
    # 持有期碰上断档：6h、12h 按断之前最后一根平；24h 的第 t+h 根在，就是真实涨跌
    assert got.loc[close_time(191)].tolist() == pytest.approx([log(d, 199, 191), log(d, 199, 191), log(d, 287, 191)])
    # 持有期里下架：三档都按下架前最后一根结算
    assert got.loc[close_time(287)].tolist() == pytest.approx([log(d, 300, 287)] * 3)
    # 够到全部数据最后一根之后的还没发生，照样是 NaN
    last = rows[rows["sym"] == 0].set_index("t").loc[close_time(359)]
    assert last["y24"] == pytest.approx(log(a, N - 1, 359)) and last[["y48", "y96", "fund48"]].isna().all()


def test_default_is_one_bar_late_moving_answers_and_funding(pool):
    late = Pool.load(pool.store, UNIVERSE, pool.cut_ms)
    # 同币同行，只换答案和资金费
    assert late.symbols == pool.symbols and late.rows[["t", "sym", "tier"]].equals(pool.rows[["t", "sym", "tier"]])
    close = pool.store.load("A")["close"].to_numpy()
    first = late.rows[late.rows["sym"] == 0].iloc[0]
    # 6 点前那根出信号、6:15 那根收盘建仓：持有 24 根到第 48 根
    assert first["t"] == close_time(23) and first["y24"] == pytest.approx(np.log(close[48] / close[24]))
    # 资金费从建仓那刻起：(6:15, 12:15] 碰上 8 点那次，晚一根不改变这里的结果
    assert (first["fund24"], first["fund48"]) == pytest.approx((0.001, 0.003))
    # 持有期往后挪了一根：不晚的时候刚好收在数据末尾的那一期，晚一根就够不到了
    t = close_time(N - 25)
    assert pool.rows.loc[(pool.rows["sym"] == 0) & (pool.rows["t"] == t), "y24"].notna().all()
    assert late.rows.loc[(late.rows["sym"] == 0) & (late.rows["t"] == t), ["y24", "fund24"]].isna().all().all()
    # 搜索段同样按晚一根的持有期判有没有够到切点之后
    t = close_time(3 * 96 - 25)
    assert pool.search.loc[pool.search["t"] == t, "y24"].notna().all() and late.search.loc[late.search["t"] == t, "y24"].isna().all()


def test_values_are_aligned_with_rows(pool):
    f = pool.values(parse("Mean($volume, 4)", COLS)).f
    assert len(f) == len(pool.rows)
    b = pool.store.load("B")
    expect = b["volume"].rolling(4).mean().loc[pool.rows.loc[pool.rows["sym"] == 1, "t"]].to_numpy()
    np.testing.assert_allclose(f[pool.rows["sym"].to_numpy() == 1], expect)


def test_cross_node_uses_only_coins_in_pool_at_that_moment(pool):
    expr = parse("CsRank($volume)", COLS)
    values = pool.values(expr)
    node = expr.cross_nodes()[0]
    vol = {s: pool.store.load(s)["volume"] for s in pool.symbols}
    jan, feb = close_time(10), close_time(2 * 96 + 10)
    # 1 月只有 A、B 两个币在池，2 月三个
    assert sorted([values.cross[node]["A"].loc[jan], values.cross[node]["B"].loc[jan]]) == [0.5, 1.0]
    assert jan not in values.cross[node]["C"].index
    ranked = pd.Series({s: vol[s].loc[feb] for s in pool.symbols}).rank(pct=True)
    assert {s: values.cross[node][s].loc[feb] for s in pool.symbols} == pytest.approx(ranked.to_dict())
    # 外层按骨架对齐；没进过池的币拿到的是空的
    rows = pool.rows
    at = (rows["sym"] == 2).to_numpy() & (rows["t"] == close_time(2 * 96 + 23)).to_numpy()
    assert values.f[at][0] == pytest.approx(values.cross[node]["C"].loc[close_time(2 * 96 + 23)])
    assert values.at("NOPE")[node].empty
