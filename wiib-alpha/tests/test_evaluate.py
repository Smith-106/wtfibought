import copy

import pandas as pd
import pytest

from wiib_alpha.dsl import parse
from wiib_alpha.eval.evaluate import evaluate, redundancy
from wiib_alpha.eval.gates import apply_fdr, judge, passed
from wiib_alpha.eval.params import EvalParams, Thresholds

COLS = ["close", "ret", "volume", "trades"]
PARAMS = EvalParams(test_size=96, embargo=4, min_train=500, iterations=200, seed=7, surrogates=20)
GATES = Thresholds(min_samples=100, min_symbols_same_sign=3)


@pytest.fixture(scope="module")
def panels(planted_panels):
    return planted_panels


def test_planted_signal_passes_every_gate(panels):
    ev = evaluate(parse("$volume", COLS), panels, PARAMS)
    assert ev["lookaheadFree"]
    main = ev["horizons"][48]["ic"]
    assert main["mean"] > 0.3 and main["sameSignSymbols"] == 3 and main["months"] >= 2
    assert main["surrogate"]["k"] == 20 and main["surrogate"]["p"] <= 0.05
    # 块内 8 根的顺序没打乱，埋进去的强信号有一小部分留在替身里，但离观测值远
    assert abs(main["surrogate"]["nullMean"]) < 0.5 * main["mean"]
    pooled = ev["horizons"][48]["leg"]["pooled"]
    assert pooled["netReturn"] > 0 and pooled["permPct"] >= 0.95
    gates = judge(ev, GATES, primary=48)
    assert passed(gates), [g for g in gates if not g["passed"]]
    assert [g["id"] for g in gates] == [f"G{i}" for i in range(9)]


def test_noise_factor_fails(panels):
    ev = evaluate(parse("$trades", COLS), panels, PARAMS)
    gates = judge(ev, GATES, primary=48)
    assert not passed(gates)
    assert abs(ev["horizons"][48]["ic"]["mean"]) < 0.1
    assert ev["horizons"][48]["ic"]["surrogate"]["p"] > 0.05


def test_high_surrogate_mean_fails_g2_even_with_small_p(panels):
    """裸价格这类持久量：真值比 20 个替身都极端一点点，p 停在 1/21，但替身自己的 IC 就很高"""
    ev = evaluate(parse("$volume", COLS), panels, PARAMS)
    assert judge(ev, GATES, primary=48)[2]["passed"]
    ev["horizons"][48]["ic"]["surrogate"]["nullMean"] = 0.9 * ev["horizons"][48]["ic"]["mean"]
    g2 = judge(ev, GATES, primary=48)[2]
    assert not g2["passed"] and g2["threshold"]["nullRatio"] == 0.5


def test_no_surrogates_means_g2_cannot_pass(panels):
    ev = evaluate(parse("$volume", COLS), panels, EvalParams(test_size=96, embargo=4, min_train=500,
                                                              iterations=50, seed=7, surrogates=0))
    assert ev["horizons"][48]["ic"]["surrogate"]["k"] == 0
    assert not judge(ev, GATES, primary=48)[2]["passed"]


def test_evaluate_output_is_plain_data(panels):
    ev = evaluate(parse("Mean($volume, 4) / 2", COLS), panels, PARAMS)
    assert ev["expr"] == "(Mean($volume, 4) / 2)" and ev["canonical"] == "(Mean($volume, w) / c)"
    assert ev["columns"] == ["volume"] and ev["complexity"]["nodes"] == 4
    assert set(ev["horizons"]) == {24, 48, 96}
    assert set(ev["horizons"][24]["leg"]["bySymbol"]) == {"AAA", "BBB", "CCC"}
    assert ev["redundancy"] == {"maxAbs": 0.0, "with": None}


def test_redundancy_finds_the_twin(panels):
    factors = {s: p["volume"] for s, p in panels.items()}
    refs = {s: pd.DataFrame({"twin": p["volume"] * 3 + 1, "noise": p["trades"]}) for s, p in panels.items()}
    red = redundancy(factors, refs, stride=4)
    assert red["with"] == "twin" and red["maxAbs"] == pytest.approx(1.0)
    assert redundancy(factors, {}, 4) == {"maxAbs": 0.0, "with": None}


def test_gates_structural_checks(panels):
    ev = evaluate(parse("$volume", COLS), panels, PARAMS)
    assert not judge(ev, GATES, 48, known_canonicals={"$volume"})[8]["passed"]
    fat = copy.deepcopy(ev)
    fat["complexity"] = {"nodes": 41, "windows": 1, "constants": 0}
    assert not judge(fat, GATES, 48)[7]["passed"]
    leaky = copy.deepcopy(ev)
    leaky["lookaheadFree"] = False
    g0 = judge(leaky, GATES, 48)[0]
    assert not g0["passed"] and g0["value"] is False and g0["threshold"] is True
    dup = copy.deepcopy(ev)
    dup["redundancy"] = {"maxAbs": 0.9, "with": "x"}
    assert not judge(dup, GATES, 48)[6]["passed"]


def test_apply_fdr_flips_the_marginal_one(panels):
    ev = evaluate(parse("$volume", COLS), panels, PARAMS)
    ev["gates"] = judge(ev, GATES, 48)
    batch = []
    for pct, sp in zip((0.999, 0.955, 0.5, 0.4, 0.3), (0.01, 0.045, 0.5, 0.6, 0.7)):
        e = copy.deepcopy(ev)
        g5 = next(g for g in e["gates"] if g["id"] == "G5")
        g5["value"]["permPct"] = pct
        g5["passed"] = pct >= 0.95
        g2 = next(g for g in e["gates"] if g["id"] == "G2")
        g2["value"]["surrogateP"] = sp
        g2["passed"] = sp <= 0.05
        batch.append(e)
    apply_fdr(batch, q=0.10, iterations=1000)
    g5_flags = [next(g for g in e["gates"] if g["id"] == "G5")["passed"] for e in batch]
    g2_flags = [next(g for g in e["gates"] if g["id"] == "G2")["passed"] for e in batch]
    assert g5_flags == [True, False, False, False, False]
    # 替身 p 是离散的，不做 BH：G2 保持单条判的结果
    assert g2_flags == [True, True, False, False, False]
    assert batch[0]["passed"] and not batch[1]["passed"]
    assert "BH" in next(g for g in batch[1]["gates"] if g["id"] == "G5")["note"]


def test_a_real_factor_survives_batch_fdr_at_production_k(panels):
    """默认 K=3：一条真信号加两条噪音一起做 BH，真信号必须还能过"""
    good = evaluate(parse("$volume", COLS), panels, PARAMS)
    good["gates"] = judge(good, GATES, 48)
    batch = [good]
    for seed in (1, 2):
        noise = evaluate(parse("$trades", COLS), panels, EvalParams(test_size=96, embargo=4, min_train=500,
                                                                     iterations=200, seed=seed, surrogates=20))
        noise["gates"] = judge(noise, GATES, 48)
        batch.append(noise)
    apply_fdr(batch, q=0.10, iterations=200)
    assert batch[0]["passed"], [g for g in batch[0]["gates"] if not g["passed"]]
