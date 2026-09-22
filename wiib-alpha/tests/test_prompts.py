from wiib_alpha import ops, prompts
from wiib_alpha.data.panel import FEATURE_COLUMNS

FACTS = {"symbols": ["BTCUSDT", "ETHUSDT"], "from": "2024-09-21", "to": "2026-09-20", "rows": 70000,
         "liqFrom": "2026-04-27"}


def test_parse_object_and_list_tolerate_fences_and_chatter():
    assert prompts.parse_object('好的：\n```json\n{"expr": "$ret", "note": "x"}\n```') == {"expr": "$ret", "note": "x"}
    assert prompts.parse_object("没有 json") is None
    assert prompts.parse_object('{"a": 1} 然后 {"b": 2}') == {"a": 1}
    assert prompts.parse_list('结果：[{"name": "a"}, {"name": "b"}] 完') == [{"name": "a"}, {"name": "b"}]
    assert prompts.parse_list('[1, 2') is None
    assert prompts.parse_object('[1, 2]') is None


def test_system_prompt_lists_every_column_and_operator():
    text = prompts.system_gen(FACTS, "zh")
    for c in FEATURE_COLUMNS:
        assert f"${c}" in text
    for name in ops.OPERATORS:
        assert f"- {name}：" in text
    assert "2026-04-27" in text and "中文" in text
    assert "英文" in prompts.system_gen({**FACTS, "liqFrom": None}, "en")


def test_propose_and_implement_messages():
    text = prompts.propose_message(3, [{"name": "mom", "expr": "$ret", "hypothesis": "h"}],
                                   [{"outcome": "rejected_gates", "hypothesis": "old", "expr": "$x", "reason": "G2"}],
                                   "只看资金费率", "zh")
    assert "mom：$ret" in text and "[rejected_gates] old" in text and "只看资金费率" in text and "提出 3 条" in text
    cand = {"name": "carry", "hypothesis": "费率高做空"}
    assert "上一版的问题" not in prompts.implement_message(cand, None)
    assert "表达式不合法：x" in prompts.implement_message(cand, "表达式不合法：x")


def test_eval_brief_rounds_and_maps_nan():
    ev = {
        "horizons": {48: {"ic": {"mean": 0.123456, "icir": float("nan"), "months": 3, "sameSignSymbols": 2,
                                 "segments": [0.1, float("nan")],
                                 "surrogate": {"p": 0.0476, "nullMean": -0.01, "k": 20}},
                          "leg": {"pooled": {"netReturn": 0.05, "permPct": 0.99, "hitRate": 0.6, "exposure": 0.3}}}},
        "redundancy": {"maxAbs": 0.2, "with": "MA48"},
        "complexity": {"nodes": 3, "windows": 1, "constants": 0},
    }
    gates = [{"id": "G0", "passed": True, "note": "ok"}]
    brief = prompts.eval_brief(ev, gates, 48)
    assert brief["horizons"]["12h"]["meanIc"] == 0.1235 and brief["horizons"]["12h"]["icir"] is None
    assert brief["horizons"]["12h"]["segments"] == [0.1, None]
    assert brief["surrogate"] == {"p": 0.0476, "nullMean": -0.01, "k": 20}
    assert brief["gates"] == [{"id": "G0", "passed": True, "note": "ok"}]
    assert "MA48" in prompts.review_message({"name": "x", "hypothesis": "h", "expr": "$ret"}, brief)
