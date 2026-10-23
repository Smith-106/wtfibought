import numpy as np

from wiib_alpha.dsl import parse
from wiib_alpha.eval.params import EvalParams
from wiib_alpha.eval.recheck import recheck

COLS = ["close", "ret", "volume", "trades"]
PARAMS = EvalParams(min_points=8)


def test_recheck_uses_only_grid_points_after_since(planted_pool):
    rows = planted_pool.rows
    f = planted_pool.values(parse("$volume", COLS)).f
    since = int(np.sort(rows["t"].unique())[125])
    out = recheck(rows, f, since, PARAMS)
    assert out["since"] == since and set(out["horizons"]) == {24, 48, 96}
    h = out["horizons"][48]
    assert h["mean"] > 0.3 and h["monotonic"] > 0.9
    # 6h 网格 250 个点，since 之后 124 个，最后一个点答案够不到；24h 网格是每 4 个取一个
    assert out["horizons"][24]["samples"] == 124 - 1 and 29 <= out["horizons"][96]["samples"] <= 31


def test_recheck_with_no_future_data_is_empty(planted_pool):
    rows = planted_pool.rows
    f = planted_pool.values(parse("$volume", COLS)).f
    out = recheck(rows, f, int(rows["t"].max()), PARAMS)
    assert out["horizons"][48]["samples"] == 0 and np.isnan(out["horizons"][48]["mean"])
