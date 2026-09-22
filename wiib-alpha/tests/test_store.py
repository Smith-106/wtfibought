import math

import numpy as np
import pytest

from wiib_alpha.store import Store, clean, dumps


@pytest.fixture
async def store():
    s = await Store.open(":memory:")
    yield s
    await s.close()


def test_clean_turns_nan_and_numpy_into_json():
    obj = {"a": float("nan"), "b": np.float64(1.5), "c": [np.int64(2), float("inf")], "d": {"e": None}}
    assert clean(obj) == {"a": None, "b": 1.5, "c": [2, None], "d": {"e": None}}
    assert dumps({"x": math.nan}) == '{"x": null}'


async def test_run_and_candidate_roundtrip(store):
    await store.create_run("r1", 3, "试试", "zh")
    run = await store.run("r1")
    assert run["status"] == "running" and run["k"] == 3 and run["note"] == "试试"
    assert [r["id"] for r in await store.running()] == ["r1"]

    cand = {"id": "r1-1", "run_id": "r1", "seq": 1, "name": "mom", "hypothesis": "动量", "status": "proposed"}
    await store.upsert_candidate(cand)
    cand.update(expr="$ret", canonical="$ret", status="evaluating", attempts=2,
                eval={"ic": float("nan"), "n": np.int64(3)}, gates=[{"id": "G0", "passed": True}])
    await store.upsert_candidate(cand)
    rows = await store.candidates("r1")
    assert len(rows) == 1 and rows[0]["status"] == "evaluating" and rows[0]["attempts"] == 2
    assert rows[0]["eval"] == {"ic": None, "n": 3} and rows[0]["gates"][0]["id"] == "G0"

    await store.set_run_status("r1", "failed", "boom")
    assert (await store.run("r1"))["error"] == "boom"
    assert await store.running() == []
    assert (await store.runs())[0]["id"] == "r1"


async def test_claim_run_is_first_come_first_served(store):
    await store.create_run("r9", 1, "", "zh")
    await store.set_run_status("r9", "awaiting")
    assert await store.claim_run("r9", "awaiting", "running") is True
    assert await store.claim_run("r9", "awaiting", "running") is False
    assert (await store.run("r9"))["status"] == "running"


async def test_events_tried_factors_trials(store):
    await store.create_run("r1", 1, "", "en")
    first = await store.add_event("r1", "status", {"stage": "proposing"})
    await store.add_event("r1", "log", {"text": "x"})
    events = await store.events("r1")
    assert [e["event"] for e in events] == ["status", "log"] and events[0]["data"] == {"stage": "proposing"}
    assert [e["event"] for e in await store.events("r1", after=first)] == ["log"]

    await store.add_tried("r1", "假设 A", "$ret", "rejected_gates", "G2 不过")
    await store.add_tried("r1", "假设 B", None, "failed", None)
    await store.add_tried("r1", "假设 A", "$ret", "rejected_gates", "重跑一遍")  # 同轮同假设只留一条
    tried = await store.tried()
    assert [t["hypothesis"] for t in tried] == ["假设 A", "假设 B"] and tried[0]["reason"] == "重跑一遍"

    factor = {"id": "f1", "name": "mom", "hypothesis": "h", "expr": "$ret", "canonical": "$ret",
              "columns": ["ret"], "eval": {"x": 1}, "gates": [], "review": None, "approved_at": 5, "run_id": "r1"}
    await store.add_factor(factor)
    await store.add_factor({**factor, "eval": {"x": 2}})  # commit 重跑：覆盖写，不撞主键
    factor = await store.factor("f1")
    assert factor["columns"] == ["ret"] and factor["eval"] == {"x": 2} and factor["recheck"] is None
    assert len(await store.factors()) == 1
    await store.set_recheck("f1", {"ic": 0.1})
    assert (await store.factors())[0]["recheck"] == {"ic": 0.1}
    assert await store.factor("nope") is None

    assert await store.trials() == 0
    assert await store.bump_trials(3) == 3
    assert await store.bump_trials(2) == 5
