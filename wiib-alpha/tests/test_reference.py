import numpy as np
import pandas as pd
import pytest

from wiib_alpha import reference
from wiib_alpha.data.panel import PanelStore, add_targets
from wiib_alpha.data.pool import Pool, on_grid
from wiib_alpha.dsl import parse

COLS = ["close", "volume", "ret", "btc_ret", "open", "high", "low"]


def frame(**cols) -> pd.DataFrame:
    df = pd.DataFrame(cols)
    if "ret" not in df and "close" in df:
        df["ret"] = np.log(df["close"] / df["close"].shift(1))
    return df


def last(text: str, df: pd.DataFrame, window_override: dict[str, int] | None = None) -> float:
    return float(parse(text, COLS).evaluate(df).iloc[-1])


# ========== 与 Java ContinuousFactorSignalsTest 同一组输入 ==========


def test_risk_adjusted_momentum_matches_java():
    df = frame(close=[100.0, 110.0, 105.0, 130.0], volume=[1.0] * 4)
    returns = np.log([110 / 100, 105 / 110, 130 / 105])
    expected = np.log(130 / 100) / returns.std(ddof=1)
    assert last("Sum($ret, 3) / Std($ret, 3)", df) == pytest.approx(expected, abs=1e-12)
    assert last("-(Sum($ret, 3) / Std($ret, 3))", df) == pytest.approx(-expected, abs=1e-12)


def test_volume_zscore_matches_java():
    df = frame(close=[1.0, 1.0, 1.0], volume=[100.0, 200.0, 400.0])
    window = np.array([100.0, 200.0, 400.0])
    expected = (400.0 - window.mean()) / window.std(ddof=1)
    assert last("($volume - Mean($volume, 3)) / Std($volume, 3)", df) == pytest.approx(expected, abs=1e-12)


def test_amihud_matches_java():
    df = frame(close=[100.0, 110.0, 99.0], volume=[10.0, 20.0, 30.0])
    expected = (abs(np.log(110 / 100)) / (110 * 20) + abs(np.log(99 / 110)) / (99 * 30)) / 2
    assert last("Mean(Abs($ret) / ($close * $volume), 2)", df) == pytest.approx(expected, rel=1e-12)


def test_residual_momentum_matches_java():
    # 收益序列直接给：alt = 2·btc 前三点，第四点 9 − 2·4 = 1
    df = pd.DataFrame({"ret": [2.0, 4.0, 6.0, 9.0], "btc_ret": [1.0, 2.0, 3.0, 4.0],
                       "close": [1.0] * 4, "volume": [1.0] * 4})
    text = "Sum($ret - Ref(Cov($ret, $btc_ret, 3) / Var($btc_ret, 3), 1) * $btc_ret, 1)"
    assert last(text, df) == pytest.approx(1.0, abs=1e-12)


def test_baselines_use_the_java_windows():
    assert reference.BASELINES["risk_adj_momentum"][0] == "Sum($ret, 48) / Std($ret, 48)"
    assert "192" in reference.BASELINES["residual_momentum"][0] and ", 48)" in reference.BASELINES["residual_momentum"][0]


# ========== 参照集整体 ==========


def synthetic_panel(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)), "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.lognormal(3, 0.5, n), "quote_volume": rng.lognormal(10, 0.5, n), "trades": rng.integers(1, 100, n),
        "taker_buy_base": rng.lognormal(2, 0.5, n), "taker_buy_quote": rng.lognormal(9, 0.5, n),
        "funding_rate": rng.normal(0, 1e-4, n), "btc_close": close, "liq_long": rng.uniform(0, 1, n),
        "liq_short": rng.uniform(0, 1, n), "liq_count": rng.integers(0, 5, n), "hour": np.arange(n) % 24,
        "premium": rng.normal(0, 1e-4, n), "oi_value": rng.lognormal(15, 0.3, n),
    }, index=np.arange(n) * 900_000)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["btc_ret"] = df["ret"]
    return df


def test_reference_set_size_and_names():
    exprs = reference.expressions()
    assert len(exprs) == 5 + 10 + 9 + 29 * 5
    assert list(exprs)[:15] == [*reference.BASELINES, *reference.KNOWN_EFFECTS]
    assert "MA48" in exprs and "KMID" in exprs and "VSUMD192" in exprs
    # 老效应的名字不以数字结尾：按窗口筛 Alpha158 的地方不会把它们当成某个窗口
    assert not any(name[-1].isdigit() for name in reference.KNOWN_EFFECTS)


def test_every_reference_parses_and_produces_numbers():
    # 老效应最长 3 周窗口（2016 根）
    df = synthetic_panel(n=2100)
    bad = []
    for name, expr in reference.parsed().items():
        out = expr.evaluate(df)
        if not np.isfinite(out.to_numpy()).any():
            bad.append(name)
    assert bad == []


def test_grid_ranks_in_pool_rows_per_moment_and_aligns_to_the_pool(tmp_path):
    store = PanelStore(tmp_path)
    for i, s in enumerate("ABC"):
        df = synthetic_panel(seed=20 + i)
        # C 前一半不在池
        df["in_pool"] = 1.0 if s != "C" else (np.arange(len(df)) >= 200).astype("float64")
        add_targets(df)
        store.save_panel(s, df)
    exprs = {"mom": parse("Sum($ret, 4)", COLS), "vol": parse("$volume", COLS)}
    progress = {"done": 0}
    table = reference.grid(store, ["A", "B", "C"], progress, exprs)
    assert list(table.columns) == ["t", "symbol", "mom", "vol"] and progress == {"done": 3, "symbol": None}
    assert on_grid(table["t"].to_numpy(), reference.GRID_BARS).all() and table["vol"].dtype == "float32"
    # 每个时刻在池的币里排秩：C 进池前那些时刻只有两个币
    early = table[table["t"] == table["t"].min()]
    assert sorted(early["symbol"]) == ["A", "B"] and sorted(early["vol"]) == [0.5, 1.0]
    late = table[table["t"] == table["t"].max()]
    raw = {s: store.load(s)["volume"].loc[late["t"].iloc[0]] for s in "ABC"}
    assert dict(zip(late["symbol"], late["vol"])) == pytest.approx(pd.Series(raw).rank(pct=True).to_dict())

    universe = pd.DataFrame([("1970-01", s, i + 1, 1.0) for i, s in enumerate("ABC")],
                            columns=["month", "symbol", "rank", "avg_quote_volume"])
    pool = Pool.load(store, universe, cut_ms=int(table["t"].max()) - 1)
    grid = reference.align(table, pool)
    main = pool.search[on_grid(pool.search["t"].to_numpy(), reference.GRID_BARS)]
    assert grid.names == ("mom", "vol") and grid.ranks.shape == (len(main), 2)
    first = main.iloc[0]
    expect = table[(table["t"] == first["t"]) & (table["symbol"] == pool.symbols[int(first["sym"])])]["vol"].iloc[0]
    assert grid.ranks[0, 1] == expect
    assert reference.Grid.empty().names == ()


def test_saved_grid_counts_as_missing_once_the_reference_set_changes(tmp_path, monkeypatch):
    store = PanelStore(tmp_path)
    store.write_status({})
    table = pd.DataFrame({"t": [1], "symbol": ["A"], "mom": np.float32([0.5])})
    store.save_reference_grid(table)
    monkeypatch.setattr(reference, "parsed", lambda: {"mom": parse("Sum($ret, 4)", COLS)})
    pd.testing.assert_frame_equal(reference.load_grid(store), table)
    # 参照集加了一条：旧网格少一列，得重算
    monkeypatch.setattr(reference, "parsed", lambda: {"mom": parse("Sum($ret, 4)", COLS), "vol": parse("$volume", COLS)})
    assert reference.load_grid(store) is None
