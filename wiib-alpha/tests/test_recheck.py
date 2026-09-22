import numpy as np

from wiib_alpha.dsl import parse
from wiib_alpha.eval.params import EvalParams
from wiib_alpha.eval.recheck import recheck

COLS = ["close", "ret", "volume", "trades"]
PARAMS = EvalParams(min_points=8)


def test_recheck_uses_only_data_after_since(planted_panels):
    first = next(iter(planted_panels.values()))
    since = int(first.index[3000])
    out = recheck(parse("$volume", COLS), planted_panels, since, PARAMS)
    assert out["since"] == since and set(out["horizons"]) == {24, 48, 96}
    h = out["horizons"][48]
    assert h["mean"] > 0.3 and h["samples"] > 0 and h["sameSignSymbols"] == 3
    # 24h 档：3000 行 / 96 ≈ 31 点一币
    assert 80 <= out["horizons"][96]["samples"] <= 96


def test_recheck_with_no_future_data_is_empty(planted_panels):
    first = next(iter(planted_panels.values()))
    out = recheck(parse("$volume", COLS), planted_panels, int(first.index[-1]), PARAMS)
    assert out["horizons"][48]["samples"] == 0 and np.isnan(out["horizons"][48]["mean"])
