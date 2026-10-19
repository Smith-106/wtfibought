import numpy as np

from wiib_alpha import calibrate
from wiib_alpha.data.panel import BENCHMARK, FEATURE_COLUMNS, TARGET_COLUMNS
from wiib_alpha.dsl import parse
from wiib_alpha.eval.evaluate import evaluate
from wiib_alpha.eval.params import EvalParams


def test_synthetic_store_has_the_real_layout_and_a_planted_carry(tmp_path):
    store, universe = calibrate.synthetic_store(tmp_path, coins=25, days=40, seed=1, beta=2e-4)
    df = store.load("C001USDT")
    assert len(df) == 40 * 96 and df.index.name == "close_time" and (df["in_pool"] == 1.0).all()
    assert set(FEATURE_COLUMNS) <= set(df.columns) and set(TARGET_COLUMNS) <= set(df.columns)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all() and df["liq_count"].isna().all()
    # 第一个币当 BTC 基准：它自己的 btc 两列是空的，别的币拿它的收益
    btc = store.load(BENCHMARK)
    assert btc["btc_ret"].isna().all() and np.allclose(df["btc_ret"].dropna(), btc["ret"].dropna())
    # 池子表每个月 25 个币，排名按成交额
    assert universe.groupby("month").size().tolist() == [25, 25] and universe["rank"].max() == 25
    # 植入的信号：费率截面排名高的币之后涨得多
    pool = calibrate.load_pool(store, universe, holdout_days=5)
    params = EvalParams(test_size=96, embargo=4, min_train=500, iterations=50, holdout_days=5)
    ev = evaluate(parse("$funding_rate", FEATURE_COLUMNS), pool, params)
    assert ev["horizons"][48]["ic"]["mean"] > 0.1


def test_random_expression_is_valid_and_not_degenerate(tmp_path):
    store, _ = calibrate.synthetic_store(tmp_path, coins=3, days=10, seed=2, beta=0.0)
    panel = store.load("C001USDT")
    rng = np.random.default_rng(3)
    for _ in range(5):
        expr = calibrate.random_expression(rng, panel)
        out = expr.evaluate(panel).to_numpy()
        assert np.isfinite(out).mean() > 0.5 and np.nanstd(out) > 0 and "funding_rate" not in expr.columns()
