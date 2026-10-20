import copy

from wiib_alpha import ops, prompts
from wiib_alpha.data.panel import FEATURE_COLUMNS

FACTS = {"symbols": ["BTCUSDT", "ETHUSDT"], "from": "2024-09-21", "to": "2026-09-20", "rows": 70000,
         "liqFrom": "2026-04-27", "holdoutFrom": "2026-06-22", "costTiers": [5.0, 8.0, 12.0], "poolSize": 80,
         "poolSymbols": 481}


def test_parse_object_and_list_tolerate_fences_and_chatter():
    assert prompts.parse_object('好的：\n```json\n{"expr": "$ret", "note": "x"}\n```') == {"expr": "$ret", "note": "x"}
    assert prompts.parse_object("没有 json") is None
    assert prompts.parse_object('{"a": 1} 然后 {"b": 2}') == {"a": 1}
    assert prompts.parse_list('结果：[{"name": "a"}, {"name": "b"}] 完') == [{"name": "a"}, {"name": "b"}]
    assert prompts.parse_list('[1, 2') is None
    assert prompts.parse_object('[1, 2]') is None


def test_system_prompt_lists_every_column_and_operator():
    # 生成、评审、复盘都要读公式，三份系统提示带同一份列和算子说明
    for text in (prompts.system_gen(FACTS, "zh"), prompts.system_review("zh"), prompts.system_reflect("zh")):
        for c in FEATURE_COLUMNS:
            assert f"${c}" in text
        for name in ops.OPERATORS:
            assert f"- {name}：{prompts.OPERATOR_DOCS[name]}" in text
    text = prompts.system_gen(FACTS, "zh")
    assert "2026-04-27" in text and "中文" in text
    assert "强平三列只有产品币有数" in text and "截面因子用不了" in text and "单独排序没有意义" in text
    assert "2026-06-22 起是锁定的 holdout" in text and "单边 5 / 8 / 12 bps" in text
    assert "前 80 个币" in text and "共 481 个" in text and "BTCUSDT, ETHUSDT" in text
    # 已知效应带公式列出来；去老效应是标签不是门
    assert "- turnover_oi：Sum($quote_volume, 672) / $oi_value（换手" in text and "它不是门，但评审和人都会看" in text
    assert "t（ICIR × √月数）≥ 3" in text and "最近一段必须同号" in text
    # 门对齐交易方案：主档 24h、按波动分仓，单条净收益只报，G9 看加进交易方案后的净 Sharpe
    assert "主档 24 小时" in text and "按 7 天波动的倒数分仓" in text and "只报不判" in text
    assert "G9 加进实盘的交易方案后净 Sharpe 至少升 0.1" in text
    assert "单边 4 / 6.5 / 10 bps" in prompts.system_gen({**FACTS, "costTiers": [4.0, 6.5, 10.0]}, "zh")
    assert "英文" in prompts.system_gen({**FACTS, "liqFrom": None}, "en")


def test_propose_and_implement_messages():
    text = prompts.propose_message(3, [{"name": "mom", "expr": "$ret", "hypothesis": "h"}],
                                   [{"outcome": "rejected_gates", "hypothesis": "old", "expr": "$x", "reason": "G2"}],
                                   [{"kind": "try", "text": "新的"}, {"kind": "avoid", "text": "旧的"}],
                                   "只看资金费率", "zh")
    assert "mom：$ret" in text and "[rejected_gates] old" in text and "只看资金费率" in text and "提出 3 条" in text
    assert "- [try] 新的\n- [avoid] 旧的" in text
    # 可选方向里是截面上做得了的：没有强平（只有产品币有数）和时段（同一时刻所有币一样）
    assert "溢价 / 基差" in text and "强平" not in text and "时段" not in text
    assert "经验（最新在前）：\n（还没有）" in prompts.propose_message(1, [], [], [], "", "zh")
    cand = {"name": "carry", "hypothesis": "费率高做空"}
    plain = prompts.implement_message(cand, None)
    assert "上一版" not in plain and "上一次输出的问题" not in plain and '"direction": 1 或 -1' in plain
    # 第一版就要想清楚：写之前的自查清单
    assert "想清楚再写" in plain and "资金费：排在做多那头和做空那头的币" in plain
    # 复评老候选补方向：只给假设和表达式，没有数字
    ask = prompts.direction_message({**cand, "expr": "-$funding_rate", "note": None})
    assert "费率高做空" in ask and "-$funding_rate" in ask and "IC" not in ask
    # 上一版的意见一直带着，输出本身的毛病接在后面
    text = prompts.implement_message(cand, "上一版表达式：$ret\n评审要求修改：换个窗口", "表达式不合法：x")
    assert text.index("评审要求修改：换个窗口") < text.index("上一次输出的问题：表达式不合法：x")
    assert prompts.revise_feedback({"expr": "$ret"}, {"suggestion": "", "comment": "太毛"}) == "上一版表达式：$ret\n评审要求修改：太毛"


def evaluated(gates: list[dict]) -> dict:
    """落过库的老结果：档的键是字符串，NaN 成了 None，没记主档（是 12h）"""
    return {"expr": "Mean($ret, 4)", "direction": -1, "gates": gates, "eval": {
        "horizons": {"48": {"ic": {"mean": 0.0123, "icir": 0.25,
                                   "monotonic": {"rho": 0.5, "spread": 0.00031, "spreadT": 2.5, "steps": [2.4, 0.3]}},
                            "leg": {"turnover": 0.214, "fundingBps": -1.26, "breakEvenBps": 3.24}}},
        "redundancy": {"maxAbs": 0.42, "with": "MA48", "ready": True},
        "neutral": {"mean": 0.0041, "icir": 0.5, "months": 36, "segments": [], "against": 10}}}


def test_neutral_brief_reports_share_left_and_t():
    ev = evaluated([])["eval"]
    assert prompts.neutral_brief(ev) == {"meanIc": 0.0041, "icir": 0.5, "t": 3.0, "retained": 0.3333,
                                         "against": 10}
    # 网格没算好是 None；落过库的 NaN 成了 None 也能算
    assert prompts.neutral_brief({**ev, "neutral": None}) is None
    assert prompts.neutral_brief({**ev, "neutral": {**ev["neutral"], "icir": None}})["t"] is None


def test_gate_feedback_lists_failed_gates_and_the_numbers():
    gates = [{"id": "G1", "name": "最小样本", "passed": True, "note": "够"},
             {"id": "G2", "name": "IC", "passed": False, "note": "ICIR 太低"},
             {"id": "G5", "name": "可交易", "passed": False, "note": "净收益为负"}]
    text = prompts.gate_feedback(evaluated(gates))
    assert text.startswith("上一版表达式：Mean($ret, 4)（声明的方向 -1，因子值越大越弱）\n"
                           "没过的门：\n- G2 IC：ICIR 太低\n- G5 可交易：净收益为负\n")
    assert "G1" not in text and "主档（12h）" in text
    for part in ("IC 均值 0.0123", "ICIR 0.25", "截尾后顶组减底组每期 3.1 bps、t 2.5",
                 "三组形状：只有弱的一头有效（中、强两组分不开）", "换手 0.21", "每期资金费 -1.3 bps",
                 "盈亏平衡单边成本 3.2 bps", "与 MA48 的截面 |ρ| 0.42", "这些数字说明了什么由你判断",
                 "回归掉 10 条老效应后 IC 0.0041（剩原来的 0.33），t 3.0"):
        assert part in text, part
    ev = evaluated(gates)["eval"]
    ev["redundancy"] = {"maxAbs": None, "with": None, "ready": False}
    ev["horizons"]["48"]["leg"]["breakEvenBps"] = None
    assert "参照网格还没算好" in prompts.key_numbers(ev) and "盈亏平衡单边成本 NaN bps" in prompts.key_numbers(ev)
    # 现在的结果记着主档 24h，数字取那一档
    ev["horizons"]["96"] = copy.deepcopy(ev["horizons"].pop("48"))
    ev["horizons"]["96"]["ic"]["mean"] = 0.0456
    assert "IC 均值 0.0456" in prompts.key_numbers({**ev, "primary": 96})


def test_reflect_message_shows_outcomes_numbers_and_old_lessons():
    good = {**evaluated([]), "seq": 1, "name": "carry", "hypothesis": "费率高的币之后弱", "status": "passed",
            "reason": None, "evaluations": 2}
    bad = {"seq": 2, "name": "bad", "hypothesis": "写不出来", "status": "failed", "reason": "连试三次都没写出合法表达式"}
    text = prompts.reflect_message([good, bad], [{"kind": "avoid", "text": "别用强平列"}], "en")
    assert "1. [carry] 假设：费率高的币之后弱\n   最终表达式：Mean($ret, 4)（评估了 2 版）\n   结局：过门" in text
    assert "盈亏平衡单边成本 3.2 bps" in text
    assert "2. [bad]" in text and "最终表达式：未写出（评估了 0 版）" in text and "失败（连试三次都没写出合法表达式）" in text
    assert "- [avoid] 别用强平列" in text and "英文" in text
    assert "英文" in prompts.system_reflect("en") and "holdout 之前" in prompts.system_reflect("zh")
    # 单条候选的结局出题时另有列表，经验只写换个假设也用得上的规律
    assert "不写某一条候选的复盘" in prompts.system_reflect("zh") and "不抄表达式" in prompts.system_reflect("zh")


def test_eval_brief_rounds_and_maps_nan():
    ev = {
        "horizons": {48: {"ic": {"mean": 0.123456, "icir": float("nan"), "months": 3, "segments": [0.1, float("nan")],
                                 "monotonic": {"rho": 0.9, "spread": 0.0123456, "groups": [0.0] * 5}},
                          "leg": {"netReturn": 0.05, "permPct": 0.99, "hitRate": 0.6, "exposure": 0.3,
                                  "breakEvenBps": 7.123456}}},
        "redundancy": {"maxAbs": 0.2, "with": "MA48", "ready": True},
        "complexity": {"nodes": 3, "windows": 1, "constants": 0, "cross": 0},
    }
    gates = [{"id": "G0", "passed": True, "note": "ok"}]
    brief = prompts.eval_brief(ev, gates)
    assert brief["primary"] == "12h"
    assert brief["horizons"]["12h"]["meanIc"] == 0.1235 and brief["horizons"]["12h"]["icir"] is None
    assert brief["horizons"]["12h"]["segments"] == [0.1, None]
    assert brief["horizons"]["12h"]["monotonic"] == 0.9 and brief["horizons"]["12h"]["spread"] == 0.0123
    # 之前落库的老结果没有顶减底的 t 和形状
    assert brief["horizons"]["12h"]["spreadT"] is None and brief["horizons"]["12h"]["shape"] is None
    assert brief["horizons"]["12h"]["breakEvenBps"] == 7.1235 and brief["horizons"]["12h"]["turnover"] is None
    assert brief["redundancy"] == {"maxAbs": 0.2, "with": "MA48", "ready": True}
    assert brief["gates"] == [{"id": "G0", "passed": True, "note": "ok"}]
    assert "holdout" not in brief and "timeseries" not in brief
    review = prompts.review_message({"name": "x", "hypothesis": "h", "expr": "$ret", "direction": -1}, brief)
    assert "MA48" in review and "声明的方向：因子值越大，未来 24 小时相对越弱" in review
    # 落过库的 holdout、时序报告键是字符串，NaN 已经成了 None
    ev["holdout"] = {"since": 5, "at": 9, "horizons": {"48": {"mean": 0.05, "icir": None, "months": 1, "samples": 60,
                                                              "monotonic": 0.7}}}
    ev["timeseries"] = {"symbols": ["BTCUSDT"], "horizons": {"48": {
        "mean": 0.02, "icir": 0.4, "months": 3, "sameSignSymbols": 1, "bySymbol": {"BTCUSDT": 0.021234},
        "netReturn": -0.1, "permPct": 0.5}}}
    brief = prompts.eval_brief(ev, gates)
    # 之前算的 holdout 没有标准误
    assert brief["holdout"] == {"since": 5, "horizons": {"12h": {"meanIc": 0.05, "icir": None, "se": None, "months": 1,
                                                                 "monotonic": 0.7, "samples": 60}}}
    assert brief["timeseries"] == {"horizons": {"12h": {"meanIc": 0.02, "icir": 0.4, "sameSignSymbols": 1,
                                                        "netReturn": -0.1}},
                                   "bySymbol": {"BTCUSDT": 0.0212}}
    # G9 的增量：老结果是三档 IC，现在是加进交易方案后的净 Sharpe
    old = prompts.eval_brief({**ev, "gain": {"horizons": {"48": {"base": 0, "with": 0.031, "gain": 0.031}},
                                             "sharpe": {"base": 0, "with": 1.2, "gain": 1.2}}}, gates)
    assert old["gain"] == {"12h": 0.031} and old["gainSharpe"]["gain"] == 1.2
    new = prompts.eval_brief({**ev, "gain": {"sharpe": {"base": 1.0, "with": 1.3, "gain": 0.3},
                                             "netReturn": {"base": 2.0, "with": 2.5}}}, gates)
    assert "gain" not in new and new["gainSharpe"] == {"base": 1.0, "with": 1.3, "gain": 0.3}
