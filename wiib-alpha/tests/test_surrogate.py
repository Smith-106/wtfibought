import numpy as np
import pytest

from wiib_alpha import calibrate
from wiib_alpha.data.panel import FEATURE_COLUMNS
from wiib_alpha.dsl import parse
from wiib_alpha.eval import surrogate
from wiib_alpha.eval.params import EvalParams


def test_surrogate_keeps_bar_shapes_and_rebuilds_the_chain():
    panel = calibrate.synthetic_panels(("AAA",), days=2, seed=4)["AAA"]
    panel["volume"] = np.arange(len(panel), dtype="float64")  # 用来认出原来的行号
    order = surrogate.block_order(len(panel), 8, np.random.default_rng(1))
    sp = surrogate.surrogate_panel(panel, order)
    assert sp.index.equals(panel.index)
    ids = sp["volume"].to_numpy().astype(int)
    assert sorted(ids) == list(range(len(panel))) and ids.tolist() != list(range(len(panel)))
    # 8 根一块整体挪，块内顺序不变，第一块不动
    assert ids[:8].tolist() == list(range(8))
    assert all(ids[i + 1] == ids[i] + 1 for i in range(len(ids) - 1) if (i + 1) % 8 != 0)
    # 每根 bar 的高低相对 close 的形状不变
    assert np.allclose(np.sort(sp["high"] / sp["close"]), np.sort(panel["high"] / panel["close"]))
    # 收益只是换了位置
    assert np.allclose(np.sort(sp["ret"].dropna()), np.sort(panel["ret"].dropna()))
    assert np.allclose(np.sort(sp["btc_ret"].dropna()), np.sort(panel["btc_ret"].dropna()))
    # 价格链和答案按新顺序重算
    assert sp["close"].iloc[0] == pytest.approx(panel["close"].iloc[0])
    assert np.allclose(sp["ret"].iloc[1:], np.log(sp["close"] / sp["close"].shift(1)).iloc[1:])
    assert np.allclose(sp["fwd_ret_24"].dropna(), np.log(sp["close"].shift(-24) / sp["close"]).dropna())
    assert np.isnan(sp["fwd_ret_96"].iloc[-1])


def test_one_order_is_shared_by_every_symbol():
    """同一套块顺序套到全部币：币之间的同涨同跌留在替身里"""
    panels = calibrate.synthetic_panels(("AAA", "BBB"), days=3, seed=6)
    for p in panels.values():
        p["volume"] = np.arange(len(p), dtype="float64")
    seen = []

    original = surrogate.surrogate_panel

    def spy(panel, order):
        seen.append(panel["volume"].to_numpy()[order].astype(int).tolist())
        return original(panel, order)

    surrogate.surrogate_panel = spy
    try:
        surrogate.null_ics(parse("$ret", FEATURE_COLUMNS), panels, EvalParams(test_size=24, embargo=4, min_train=48),
                           bars=24, k=2, seed=1)
    finally:
        surrogate.surrogate_panel = original
    assert len(seen) == 4
    assert seen[0] == seen[1] and seen[2] == seen[3] and seen[0] != seen[2]


def test_aligned_keeps_only_common_timestamps():
    panels = calibrate.synthetic_panels(("AAA", "BBB"), days=2, seed=2)
    panels["BBB"] = panels["BBB"].iloc[10:]
    same = surrogate.aligned(panels)
    assert len(same["AAA"]) == len(same["BBB"]) == len(panels["AAA"]) - 10
    assert same["AAA"].index.equals(same["BBB"].index)


def test_null_reproduces_the_level_bias_so_a_price_level_is_not_significant():
    panels = calibrate.synthetic_panels(("AAA", "BBB", "CCC"), days=200, seed=5)
    params = EvalParams(test_size=96, embargo=4, min_train=500)
    nulls = surrogate.null_ics(parse("$low", FEATURE_COLUMNS), panels, params, bars=48, k=20, seed=3)
    assert len(nulls) == 20 and np.isfinite(nulls).all()
    assert -0.3 < np.mean(nulls) < -0.08
    assert surrogate.pvalue(-0.2, nulls) > 0.05
    assert surrogate.pvalue(0.6, nulls) == pytest.approx(1 / 21)


def test_pvalue_edges():
    assert surrogate.pvalue(0.1, [0.05, -0.2, 0.0]) == pytest.approx(2 / 4)
    assert np.isnan(surrogate.pvalue(np.nan, [0.1]))
    assert np.isnan(surrogate.pvalue(0.1, [np.nan]))
    assert surrogate.pvalue(0.1, [np.nan, 0.05]) == pytest.approx(1 / 2)
    assert surrogate.null_ics(parse("$ret", FEATURE_COLUMNS), {}, EvalParams(), 48, 0, 1) == []
