import numpy as np
import pytest

from wiib_alpha.data.panel import holdout_cut, truncate
from wiib_alpha.dsl import parse
from wiib_alpha.eval import combo
from wiib_alpha.eval.params import EvalParams
from wiib_alpha.eval.walkforward import Window

COLS = ["close", "ret", "volume", "trades"]
PARAMS = EvalParams(test_size=96, embargo=4, min_train=500, iterations=100, seed=7, surrogates=0, holdout_days=10)
PLANTED = parse("Mean($volume, 4)", COLS)


def test_score_learns_sign_and_weights_noise_down():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.normal(0, 1, n)
    # 三条因子：正的、反的、纯噪音
    values = {"A": np.column_stack([y, -y, rng.normal(0, 1, n)])}
    wins = [Window(200, 210, 300), Window(290, 300, 400)]
    out = combo.score(values, {"A": y}, wins)["A"]
    assert np.isnan(out[:210]).all() and np.isfinite(out[210:]).all()
    # 反号的那条被学成正的，噪音权重接近 0，信号就是 tanh(z)
    assert np.corrcoef(out[210:300], y[210:300])[0, 1] > 0.85
    assert np.all(np.abs(out[210:]) < 1)


def test_empty_combo_has_no_signal(planted_panels):
    empty = combo.Combo.build([], planted_panels)
    assert empty.names == () and empty.values["AAA"].shape == (6000, 0)
    assert all(np.isnan(h["mean"]) for h in combo.ics(empty, PARAMS).values())


def test_combine_ignores_weightless_factors():
    # 第二条权重 0：它有值不算有值，第一条缺了整行就是 NaN
    block = np.array([[1.0, 0.5], [np.nan, 0.5]])
    out = combo._combine(block, np.zeros(2), np.ones(2), np.array([1.0, 0.0]))
    assert out[0] == pytest.approx(np.tanh(1.0)) and np.isnan(out[1])


def test_factor_values_come_from_each_coin_full_panel(planted_panels):
    # BBB 缺一段，AAA 的因子值不受影响：在完整面板上算，再只取共同时刻
    gapped = dict(planted_panels)
    gapped["BBB"] = planted_panels["BBB"].drop(planted_panels["BBB"].index[1000:1008])
    built = combo.Combo.build([("planted", PLANTED)], gapped)
    expect = PLANTED.evaluate(planted_panels["AAA"]).loc[built.panels["AAA"].index].to_numpy()
    np.testing.assert_array_equal(built.values["AAA"][:, 0], expect)
    assert len(built.panels["AAA"]) == 6000 - 8


def test_gain_of_planted_and_its_twin(planted_panels):
    cut = holdout_cut(planted_panels, PARAMS.holdout_days)
    search = {s: truncate(p, cut) for s, p in planted_panels.items()}
    empty = combo.Combo.build([], search)
    first = combo.gain(empty, "planted", PLANTED, PARAMS)
    assert first["horizons"][48]["base"] == 0.0 and first["horizons"][48]["gain"] > 0.3
    assert first["monthsBetter"] == first["months"] > 0
    lib = empty.plus("planted", PLANTED)
    twin = combo.gain(lib, "twin", parse("Mean($volume, 8)", COLS), PARAMS)
    assert twin["horizons"][48]["base"] == pytest.approx(first["horizons"][48]["with"])
    # 换个窗口的翻版加不了多少；噪音权重接近 0，组合几乎不动
    assert abs(twin["horizons"][48]["gain"]) < 0.05
    assert abs(combo.gain(lib, "noise", parse("$trades", COLS), PARAMS)["horizons"][48]["gain"]) < 0.05


def test_evaluate_holdout_and_latest(planted_panels):
    cut = holdout_cut(planted_panels, PARAMS.holdout_days)
    search = {s: truncate(p, cut) for s, p in planted_panels.items()}
    ev = combo.evaluate(combo.Combo.build([("planted", PLANTED)], search), PARAMS)
    assert set(ev) == {24, 48, 96} and ev[48]["ic"]["mean"] > 0.3
    assert ev[48]["leg"]["pooled"]["netReturn"] > 0 and set(ev[48]["leg"]["bySymbol"]) == {"AAA", "BBB", "CCC"}

    full = combo.Combo.build([("planted", PLANTED)], planted_panels)
    held = combo.holdout(full, cut, PARAMS)
    assert held["since"] == cut and held["horizons"][48]["mean"] > 0 and held["horizons"][48]["samples"] > 0
    now = combo.latest(full, PARAMS)
    assert now["asOf"] == int(planted_panels["AAA"].index[-1])
    assert [w["name"] for w in now["weights"]] == ["planted"] and now["weights"][0]["weight"] > 0
    assert set(now["latest"]) == {"AAA", "BBB", "CCC"} and all(abs(v) < 1 for v in now["latest"].values())
