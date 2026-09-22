import numpy as np
import pandas as pd
import pytest

from wiib_alpha.eval import ic, leg
from wiib_alpha.eval.params import EvalParams
from wiib_alpha.eval.walkforward import TrainStats, windows

N = 60
H = 2
WINS = windows(N, 10, H, 1, 20)  # (20,23,33) (30,33,43) (40,43,53)
PARAMS = EvalParams(iterations=300, seed=1)


def target() -> np.ndarray:
    return np.random.default_rng(5).normal(0, 0.01, N)


def test_perfect_factor_trades_only_when_confident_and_always_wins():
    y = target()
    result = leg.simulate(y, y, WINS, TrainStats(y, y), H, cost=0.001)
    assert result.rows.tolist() == list(range(23, 53, 2))
    active = result.positions != 0
    assert active.any() and not active.all()
    assert (result.positions[active] == np.sign(y[result.rows][active])).all()
    assert (result.net[active] > 0).all() and (result.net[~active] == 0).all()
    # 答案是对数收益，方向腿按算术收益算
    assert result.net[active] == pytest.approx(np.abs(np.expm1(y[result.rows][active])) - 0.001)
    summary = leg.summarize(result, H, PARAMS)
    assert summary["periods"] == 15 and summary["activeHitRate"] == 1.0
    assert summary["permPct"] >= 0.9 and summary["netReturn"] > 0
    assert 0 < summary["exposure"] < 1 and 0 < summary["turnover"] <= 1


def test_sign_is_learned_so_inverted_factor_trades_the_same():
    y = target()
    a = leg.simulate(y, y, WINS, TrainStats(y, y), H, cost=0.0)
    b = leg.simulate(-y, y, WINS, TrainStats(-y, y), H, cost=0.0)
    assert a.positions.tolist() == b.positions.tolist()


def test_nan_factor_is_flat_and_nan_target_leaves_grid():
    y = target()
    f = y.copy()
    f[25] = np.nan
    y2 = y.copy()
    y2[27] = np.nan
    result = leg.simulate(f, y2, WINS, TrainStats(f, y2), H, cost=0.0)
    assert 27 not in result.rows and result.positions[result.rows.tolist().index(25)] == 0


def test_no_windows_or_no_targets_gives_none():
    y = target()
    assert leg.simulate(y, y, [], TrainStats(y, y), H, 0.0) is None
    assert leg.simulate(y, np.full(N, np.nan), WINS, TrainStats(y, y), H, 0.0) is None


def test_pool_concatenates_and_weights():
    y = target()
    one = leg.simulate(y, y, WINS, TrainStats(y, y), H, cost=0.0)
    pooled = leg.pool({"A": one, "B": one}, H, PARAMS)
    assert pooled["periods"] == 30
    assert pooled["exposure"] == pytest.approx(one.exposure)
    assert leg.pool({}, H, PARAMS) == {"periods": 0}


# ========== IC ==========


def month_ms(year: int, month: int, day: int) -> int:
    return int(pd.Timestamp(year, month, day, tz="UTC").timestamp() * 1000)


def points(sign: float, seed: int = 9) -> pd.DataFrame:
    """1 月里 f = y，2 月里 f = sign·y"""
    rng = np.random.default_rng(seed)
    jan = month_ms(2025, 1, 1) + np.arange(20) * 43_200_000
    feb = month_ms(2025, 2, 1) + np.arange(20) * 43_200_000
    y = rng.normal(0, 0.01, 40)
    f = np.concatenate([y[:20], sign * y[20:]])
    return pd.DataFrame({"t": np.concatenate([jan, feb]), "f": f, "y": y})


def test_oos_points_grid_and_filter():
    y = target()
    f = y.copy()
    f[25] = np.nan
    p = ic.oos_points(np.arange(N) * 15, f, y, WINS, H)
    assert p["t"].tolist() == [r * 15 for r in range(23, 53, 2) if r != 25]
    assert ic.oos_points(np.arange(N) * 15, f, y, [], H).empty


def test_monthly_pooled_and_segments():
    table = ic.monthly({"A": points(-1.0), "B": points(1.0)}, min_points=8)
    assert sorted(table["month"].unique()) == ["2025-01", "2025-02"]
    got = {(r.symbol, r.month): r.ic for r in table.itertuples()}
    assert got[("A", "2025-01")] == pytest.approx(1.0) and got[("A", "2025-02")] == pytest.approx(-1.0)
    series = ic.pooled(table)
    assert series["2025-01"] == pytest.approx(1.0) and series["2025-02"] == pytest.approx(0.0)
    segs = ic.segments({"A": points(-1.0)}, k=2, min_points=8)
    assert segs == pytest.approx([1.0, -1.0])


def test_summary_counts_same_sign_symbols():
    against = points(1.0)
    against["f"] = -against["f"]
    s = ic.summary({"A": points(1.0), "B": points(1.0), "C": against}, k=4, min_points=8)
    assert s["months"] == 2 and s["mean"] > 0
    assert s["sameSignSymbols"] == 2
    assert s["samples"] == 120
    assert set(s["bySymbol"]) == {"A", "B", "C"}


def test_summary_empty():
    s = ic.summary({}, k=4, min_points=8)
    assert np.isnan(s["mean"]) and s["months"] == 0 and s["samples"] == 0 and s["sameSignSymbols"] == 0
