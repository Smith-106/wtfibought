import numpy as np
import pandas as pd
import pytest

from wiib_alpha import reference
from wiib_alpha.data.panel import FEATURE_COLUMNS
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
    }, index=np.arange(n) * 900_000)
    df["ret"] = np.log(df["close"] / df["close"].shift(1))
    df["btc_ret"] = df["ret"]
    return df


def test_reference_set_size_and_names():
    exprs = reference.expressions()
    assert len(exprs) == 5 + 9 + 29 * 5
    assert list(exprs)[:5] == list(reference.BASELINES)
    assert "MA48" in exprs and "KMID" in exprs and "VSUMD192" in exprs


def test_every_reference_parses_and_produces_numbers():
    df = synthetic_panel()
    bad = []
    for name, expr in reference.parsed().items():
        out = expr.evaluate(df)
        if not np.isfinite(out.to_numpy()).any():
            bad.append(name)
    assert bad == []


def test_values_builds_one_frame_per_symbol():
    panels = {"A": synthetic_panel(), "B": synthetic_panel(seed=12)}
    got = reference.values(panels)
    assert set(got) == {"A", "B"}
    assert list(got["A"].columns) == list(reference.expressions())
    assert got["A"].index.equals(panels["A"].index)
    assert set(FEATURE_COLUMNS) >= {"close", "volume", "ret", "btc_ret"}
