import numpy as np
import pandas as pd
import pytest

from wiib_alpha.dsl import parse
from wiib_alpha.eval import combo
from wiib_alpha.eval.params import EvalParams

COLS = ["close", "ret", "volume", "trades"]
PARAMS = EvalParams(test_size=96, embargo=4, min_train=500, iterations=100, seed=7, holdout_days=10)


def search_values(pool, text: str) -> np.ndarray:
    return pool.values(parse(text, COLS)).f[pool.in_search]


def test_zscores_are_per_moment_on_ranks():
    rows = pd.DataFrame({"t": [1, 1, 1, 2, 2, 2]})
    c = combo.Combo(rows, ("A", "B", "C"), ("x",), np.array([[1.0], [2.0], [3.0], [5.0], [5.0], [np.nan]]))
    z = combo.zscores(c)[:, 0]
    assert z[:3] == pytest.approx([-1.0, 0.0, 1.0]) and np.isnan(z[3:]).all()
    # 先转秩：一个币的值大到离谱，z 分数和它只是排第一一样
    outlier = combo.Combo(rows, ("A", "B", "C"), ("x",), np.array([[1.0], [2.0], [1e6], [5.0], [5.0], [np.nan]]))
    assert combo.zscores(outlier)[:3, 0] == pytest.approx([-1.0, 0.0, 1.0])


def test_combine_ignores_weightless_and_missing_factors():
    z = np.array([[1.0, 0.5], [np.nan, 0.5], [np.nan, np.nan]])
    out = combo._combine(z, np.array([2.0, 0.0]))
    # 第二条权重 0：它有值不算有值，第一条缺了整行就是 NaN
    assert out[0] == pytest.approx(1.0) and np.isnan(out[1:]).all()
    assert combo._combine(z[:1], np.array([1.0, -1.0]))[0] == pytest.approx(0.25)


def test_empty_combo(planted_pool):
    empty = combo.Combo.empty(planted_pool.search, planted_pool.symbols)
    assert empty.names == () and empty.values.shape == (len(planted_pool.search), 0)


def test_signal_learns_the_sign_of_an_inverted_factor(planted_pool):
    empty = combo.Combo.empty(planted_pool.search, planted_pool.symbols)
    inverted = empty.plus("inv", -search_values(planted_pool, "$volume"))
    s = combo.signal(inverted, combo.zscores(inverted), 48, PARAMS)
    planted = search_values(planted_pool, "$volume")
    ok = np.isfinite(s)
    # 方向学成负的：组合信号和原值同号排序
    assert ok.any() and np.corrcoef(s[ok], planted[ok])[0, 1] > 0.5


def test_evaluate_holdout_and_latest(planted_pool, planted_vol):
    expr = parse("Mean($volume, 4)", COLS)
    full_values = planted_pool.values(expr).f
    base = combo.Combo.empty(planted_pool.search, planted_pool.symbols).plus("planted", full_values[planted_pool.in_search])
    ev = combo.evaluate(base, planted_vol[planted_pool.in_search], PARAMS)
    assert set(ev) == {24, 48, 96} and ev[48]["ic"]["mean"] > 0.3
    assert ev[48]["leg"]["netReturn"] > 0 and ev[48]["leg"]["permPct"] >= 0.95

    full = combo.Combo.empty(planted_pool.rows, planted_pool.symbols).plus("planted", full_values)
    held = combo.holdout(full, planted_pool.cut_ms, PARAMS)
    assert held["since"] == planted_pool.cut_ms and held["horizons"][48]["mean"] > 0.3
    assert held["horizons"][48]["samples"] > 0
    now = combo.latest(full, PARAMS)
    assert now["asOf"] == int(planted_pool.rows["t"].max())
    assert [w["name"] for w in now["weights"]] == ["planted"] and now["weights"][0]["weight"] > 0
    assert set(now["latest"]) == set(planted_pool.symbols)
    # 一条因子：组合信号就是它的截面 z 分数
    assert np.mean(list(now["latest"].values())) == pytest.approx(0.0, abs=1e-9)
