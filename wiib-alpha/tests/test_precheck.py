import numpy as np
import pytest

from wiib_alpha.dsl import parse
from wiib_alpha.eval.params import EvalParams
from wiib_alpha.eval.precheck import precheck

COLS = ["close", "volume", "trades", "funding_rate"]
PARAMS = EvalParams(test_size=96, embargo=4, min_train=500, iterations=200, seed=7, holdout_days=10)


@pytest.fixture(scope="module")
def check(planted_pool):
    pick = planted_pool.in_search
    funding = planted_pool.values(parse("$funding_rate", COLS)).f[pick]

    def run(text: str) -> dict:
        return precheck(planted_pool.search, planted_pool.values(parse(text, COLS)).f[pick], funding, PARAMS)
    return run


def test_slow_factors_keep_their_ranks_and_noise_does_not(check):
    smooth, raw, noise = (check(text) for text in ("Mean($volume, 672)", "$volume", "$trades"))
    # 埋的 volume 是 0.98 的 AR(1)：隔一天（96 根）还剩约 0.98^96 ≈ 0.14 的相关，7 天均线变得更慢；trades 是白噪音
    assert smooth["rankAutocorr"] > raw["rankAutocorr"] + 0.3
    assert raw["rankAutocorr"] > noise["rankAutocorr"] + 0.05 and abs(noise["rankAutocorr"]) < 0.1


def test_coverage_and_points(check):
    close, late = check("$close"), check("Mean($volume, 2016)")
    assert close["coverage"] == pytest.approx(1.0) and close["points"] > 40
    # 前 2016 根（21 天）算不出来：那些点不算
    assert late["points"] < close["points"] - 15


def test_funding_at_both_ends(check):
    noise = check("$trades")
    assert np.isfinite(noise["fundingHigh"]) and np.isfinite(noise["fundingLow"])
    # 按费率自己排：高的一头费率是正的、低的一头是负的
    carry = check("$funding_rate")
    assert carry["fundingHigh"] > 0 > carry["fundingLow"]
