import pytest
from fakes import gen_reply, model, review_reply
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from wiib_alpha import graph as graph_module
from wiib_alpha.data.panel import holdout_cut, truncate
from wiib_alpha.eval.combo import Combo
from wiib_alpha.eval.params import EvalParams, Thresholds
from wiib_alpha.graph import Context, build_graph
from wiib_alpha.store import Store

PARAMS = EvalParams(test_size=96, embargo=4, min_train=500, iterations=200, seed=7, surrogates=20, holdout_days=10)
GATES = Thresholds(min_samples=100, min_symbols_same_sign=3)
FACTS = {"symbols": ["AAA", "BBB", "CCC"], "from": "2025-01-01", "to": "2025-03-04", "rows": 6000, "liqFrom": None,
         "holdoutFrom": "2025-02-22", "costBps": 10.0}


@pytest.fixture
async def store():
    s = await Store.open(":memory:")
    yield s
    await s.close()


def review_blind_to_holdout(prompt: str) -> str:
    """评审只能看到搜索段的数字"""
    assert "holdout" not in prompt
    return review_reply(prompt)


def context(store, panels, gen=gen_reply, review=review_blind_to_holdout) -> Context:
    cut = holdout_cut(panels, PARAMS.holdout_days)
    search = {s: truncate(p, cut) for s, p in panels.items()}
    return Context(gen=model(gen), review=model(review), panels=panels, search=search, cut_ms=cut, references={},
                   combo=Combo.build([], search), known=set(), library=[], facts=FACTS, store=store, params=PARAMS,
                   thresholds=GATES)


async def run_until_stop(graph, payload, cfg):
    events = []
    async for mode, chunk in graph.astream(payload, cfg, stream_mode=["updates", "custom"]):
        events.append((mode, chunk))
    return events


async def test_full_round_with_retry_revise_fdr_and_approval(store, planted_panels):
    await store.create_run("r1", 2, "", "zh")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_graph(saver, context(store, planted_panels))
        cfg = {"configurable": {"thread_id": "r1"}}
        events = await run_until_stop(graph, {"run_id": "r1", "lang": "zh", "k": 2, "note": ""}, cfg)

        interrupts = [c for m, c in events if m == "updates" and "__interrupt__" in c]
        assert len(interrupts) == 1
        survivors = interrupts[0]["__interrupt__"][0].value["candidates"]
        assert [s["name"] for s in survivors] == ["planted"]
        assert survivors[0]["expr"] == "Mean($volume, 4)" and survivors[0]["review"]["verdict"] == "pass"
        assert survivors[0]["brief"]["surrogate"]["k"] == 20
        # 幸存者带着 holdout 的数字去审批；埋进去的信号在 holdout 上也该是正的
        held = survivors[0]["brief"]["holdout"]
        assert held["since"] == holdout_cut(planted_panels, PARAMS.holdout_days)
        assert held["horizons"]["12h"]["meanIc"] > 0 and held["horizons"]["12h"]["samples"] > 0

        rows = {c["name"]: c for c in await store.candidates("r1")}
        # planted：第一次少窗口重写、第二次 $volume 被评审打回、第三次过
        assert rows["planted"]["status"] == "passed" and rows["planted"]["attempts"] == 3
        assert rows["planted"]["eval"]["holdout"]["since"] == held["since"]
        # 库是空的：G9 的基线是 0，增量就是它自己的组合 IC
        assert rows["planted"]["gates"][9]["id"] == "G9" and rows["planted"]["gates"][9]["passed"]
        assert rows["planted"]["eval"]["gain"]["horizons"]["48"]["base"] == 0
        assert rows["planted"]["eval"]["gain"]["horizons"]["48"]["gain"] > 0.3 and survivors[0]["brief"]["gain"]["12h"] > 0.3
        assert rows["noise"]["status"] == "rejected_gates" and "G2" in rows["noise"]["reason"]
        assert rows["noise"]["gates"] is not None and "holdout" not in rows["noise"]["eval"]
        custom = [c for m, c in events if m == "custom"]
        # planted 评估了两版、noise 一版
        assert any(c["event"] == "log" and "累计试验 3 次" in c["data"]["text"] for c in custom)
        assert (await store.run("r1"))["status"] == "running"

        planted_id = survivors[0]["id"]
        events = await run_until_stop(graph, Command(resume={"decisions": {planted_id: True}}), cfg)
        assert any(m == "custom" and c["event"] == "done" and c["data"] == {"approved": 1} for m, c in events)

    assert (await store.run("r1"))["status"] == "done"
    factors = await store.factors()
    assert len(factors) == 1 and factors[0]["expr"] == "Mean($volume, 4)" and factors[0]["columns"] == ["volume"]
    assert factors[0]["eval"]["holdout"]["since"] == held["since"]
    rows = {c["name"]: c for c in await store.candidates("r1")}
    assert rows["planted"]["status"] == "approved" and rows["planted"]["decision"] == "approve"
    tried = await store.tried()
    assert {t["outcome"] for t in tried} == {"approved", "rejected_gates"}
    assert await store.trials() == 3
    kinds = [e["event"] for e in await store.events("r1")]
    assert kinds[0] == "status" and "done" in kinds and kinds.count("candidate") >= 6


async def test_declined_survivor_is_not_stored(store, planted_panels):
    await store.create_run("r2", 2, "", "en")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_graph(saver, context(store, planted_panels))
        cfg = {"configurable": {"thread_id": "r2"}}
        await run_until_stop(graph, {"run_id": "r2", "lang": "en", "k": 2, "note": ""}, cfg)
        await run_until_stop(graph, Command(resume={"decisions": {}}), cfg)
    assert await store.factors() == []
    rows = {c["name"]: c for c in await store.candidates("r2")}
    assert rows["planted"]["status"] == "declined"


async def test_batch_rechecks_later_survivors_against_earlier_ones(store, planted_panels, monkeypatch):
    """同批里规范形相同的只留一条（G8 拒）；结构不同的照样过，G9 备注里写明排在谁后面增量多少，留给人判"""
    def no_gain(combo, name, expr, params):
        return {"horizons": {b: {"base": 0.5, "with": 0.49, "gain": -0.01} for b in (24, 48, 96)},
                "months": 2, "monthsBetter": 0}

    monkeypatch.setattr(graph_module, "gain", no_gain)
    await store.create_run("r5", 4, "", "zh")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_graph(saver, context(store, planted_panels))
        cfg = {"configurable": {"thread_id": "r5"}}
        events = await run_until_stop(graph, {"run_id": "r5", "lang": "zh", "k": 4, "note": ""}, cfg)
    interrupts = [c for m, c in events if m == "updates" and "__interrupt__" in c]
    survivors = interrupts[0]["__interrupt__"][0].value["candidates"]
    rows = {c["name"]: c for c in await store.candidates("r5")}
    # planted 与 clone 都是 Mean($volume, w)：排后面的那条按 G8 拒，不算 holdout
    dropped = [c for c in (rows["planted"], rows["clone"]) if c["status"] == "rejected_gates"]
    assert len(dropped) == 1 and dropped[0]["reason"].startswith("G8") and "holdout" not in dropped[0]["eval"]
    assert not dropped[0]["gates"][8]["passed"]
    assert {s["name"] for s in survivors} == {"twin", ("planted", "clone")[dropped[0]["name"] == "planted"]}
    # 排在后面的那条 G9 没被翻成不过，备注里带着相对前一条的增量
    noted = [s for s in survivors if "排在同批" in s["gates"][9]["note"]]
    assert len(noted) == 1 and "-0.0100" in noted[0]["gates"][9]["note"] and all(g["passed"] for g in noted[0]["gates"])
    assert all("holdout" in rows[s["name"]]["eval"] for s in survivors)
    assert rows["noise"]["status"] == "rejected_gates"


async def test_no_hypotheses_goes_straight_to_done(store, planted_panels):
    await store.create_run("r3", 1, "", "zh")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_graph(saver, context(store, planted_panels, gen=lambda p: "[]"))
        cfg = {"configurable": {"thread_id": "r3"}}
        events = await run_until_stop(graph, {"run_id": "r3", "lang": "zh", "k": 1, "note": ""}, cfg)
    assert not any(m == "updates" and "__interrupt__" in c for m, c in events)
    assert (await store.run("r3"))["status"] == "done" and await store.trials() == 0


async def test_unparseable_expression_three_times_is_failed(store, planted_panels):
    def stubborn(prompt: str) -> str:
        if "JSON 数组" in prompt:
            return '[{"name": "bad", "hypothesis": "写不出来"}]'
        return '{"expr": "Mean(close, 5)"}'

    await store.create_run("r4", 1, "", "zh")
    async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
        graph = build_graph(saver, context(store, planted_panels, gen=stubborn))
        cfg = {"configurable": {"thread_id": "r4"}}
        await run_until_stop(graph, {"run_id": "r4", "lang": "zh", "k": 1, "note": ""}, cfg)
    rows = await store.candidates("r4")
    assert rows[0]["status"] == "failed" and rows[0]["attempts"] == 3
    assert (await store.run("r4"))["status"] == "done"
