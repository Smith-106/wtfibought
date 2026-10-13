import numpy as np

from wiib_alpha import calibrate
from wiib_alpha.data.panel import FEATURE_COLUMNS, TARGET_COLUMNS


def test_synthetic_panels_have_the_real_column_layout():
    panels = calibrate.synthetic_panels(("AAA", "BBB"), days=3, seed=1)
    df = panels["BBB"]
    assert len(df) == 3 * 96 and df.index.name == "close_time"
    assert set(FEATURE_COLUMNS) <= set(df.columns) and set(TARGET_COLUMNS) <= set(df.columns)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert df["funding_rate"].notna().sum() > 0 and df["liq_count"].isna().all()
    # 第一个币当 BTC 基准：它自己的 btc 两列是空的，别的币拿它的收益
    assert panels["AAA"]["btc_ret"].isna().all()
    assert np.allclose(panels["BBB"]["btc_ret"].dropna(), panels["AAA"]["ret"].dropna())


def test_random_expression_is_valid_and_not_degenerate():
    panels = calibrate.synthetic_panels(("AAA",), days=10, seed=2)
    rng = np.random.default_rng(3)
    for _ in range(5):
        expr = calibrate.random_expression(rng, panels["AAA"])
        out = expr.evaluate(panels["AAA"]).to_numpy()
        assert np.isfinite(out).mean() > 0.5 and np.nanstd(out) > 0
