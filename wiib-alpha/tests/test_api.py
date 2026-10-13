import asyncio
import json

import httpx
import pytest
from fakes import gen_reply, model, review_reply

from wiib_alpha import main as m
from wiib_alpha import reference
from wiib_alpha.auth import Authenticator
from wiib_alpha.config import config
from wiib_alpha.data.panel import PanelStore
from wiib_alpha.dsl import parse
from wiib_alpha.eval.params import EvalParams, Thresholds

ADMIN = {"satoken": "admin"}


def sim(request: httpx.Request) -> httpx.Response:
    token = request.headers.get("satoken")
    if token == "admin":
        return httpx.Response(200, json={"code": 0, "msg": "", "data": {"id": 1}})
    if token == "user":
        return httpx.Response(200, json={"code": 0, "msg": "", "data": {"id": 7}})
    return httpx.Response(401)


@pytest.fixture
def prepared(tmp_path, monkeypatch, planted_panels):
    """面板落成 parquet、环境变量指到 tmp、参照集缩成一条、模型换成假的；返回还没进 lifespan 的 app"""
    store = PanelStore(tmp_path)
    for s, p in planted_panels.items():
        p.to_parquet(store.panels / f"{s}.parquet")
    monkeypatch.setenv("ALPHA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("ALPHA_DB_PATH", str(tmp_path / "alpha.sqlite"))
    monkeypatch.setenv("ALPHA_SYMBOLS", "AAA,BBB,CCC")
    config.cache_clear()
    # 参照集只留一条与埋进去的信号无关的；用价格类的会和 planted 真的相关（G6 会正确地拒掉它）
    monkeypatch.setattr(reference, "parsed", lambda: {"noise4": parse("Mean($trades, 4)", ["trades"])})
    monkeypatch.setattr(m, "build_models", lambda cfg: (model(gen_reply), model(review_reply)))
    monkeypatch.setattr(m, "_refresh_panels", lambda cfg: None)
    app = m.app
    app.state.params = EvalParams(test_size=96, embargo=4, min_train=500, iterations=200, seed=7, surrogates=20,
                                  holdout_days=10)
    app.state.thresholds = Thresholds(min_samples=100, min_symbols_same_sign=3)
    app.state.authenticator = Authenticator("http://sim", httpx.AsyncClient(transport=httpx.MockTransport(sim)))
    yield app
    config.cache_clear()


@pytest.fixture
async def client(prepared):
    async with m.lifespan(prepared):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=prepared), base_url="http://t") as c:
            yield c


async def events(client: httpx.AsyncClient, run_id: str, after: int = 0) -> list[dict]:
    """流在这一轮不再 running 时结束，ASGI 传输把整条流收完再给"""
    async with client.stream("GET", f"/api/alpha/runs/{run_id}/events", params={"after": after}, headers=ADMIN) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        raw = (await r.aread()).decode()
    out = []
    for block in raw.split("\n\n"):
        if not block.strip() or block.startswith(":"):
            continue
        fields = dict(line.split(": ", 1) for line in block.split("\n") if ": " in line)
        out.append({"id": int(fields["id"]), "event": fields["event"], "data": json.loads(fields["data"])})
    return out


def data(resp: httpx.Response):
    body = resp.json()
    assert resp.status_code == 200 and body["code"] == 0, body
    return body["data"]


async def wait_until(check, seconds: float = 20.0):
    for _ in range(int(seconds / 0.05)):
        if await check():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("等超时了")


# ========== 完整一轮 ==========


async def test_full_round_over_http(client, planted_panels):
    assert data(await client.get("/api/alpha/library", headers=ADMIN)) == {"factors": [], "combo": None}
    run_id = data(await client.post("/api/alpha/runs", json={"k": 2}, headers=ADMIN))["id"]

    evts = await events(client, run_id)
    trail = [(e["event"], e["data"].get("name") or e["data"].get("stage") or e["data"].get("text"),
              e["data"].get("status"), e["data"].get("reason") or e["data"].get("message")) for e in evts]
    assert evts[0]["event"] == "status", trail
    assert evts[-1] == {"id": evts[-2]["id"], "event": "end",
                        "data": {"status": "awaiting", "lastId": evts[-2]["id"]}}, trail
    awaiting = [e for e in evts if e["event"] == "awaiting"]
    assert len(awaiting) == 1 and [c["name"] for c in awaiting[0]["data"]["candidates"]] == ["planted"]
    ids = [e["id"] for e in evts if e["event"] != "end"]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)

    state = data(await client.get(f"/api/alpha/runs/{run_id}", headers=ADMIN))
    # planted 评估了两版（$volume 被打回、Mean($volume, 4)），noise 一版
    assert state["run"]["status"] == "awaiting" and state["trials"] == 3
    by_name = {c["name"]: c for c in state["candidates"]}
    assert by_name["planted"]["status"] == "passed" and by_name["planted"]["brief"]["gates"][2]["passed"]
    assert by_name["planted"]["brief"]["holdout"]["horizons"]["12h"]["samples"] > 0
    assert by_name["noise"]["status"] == "rejected_gates" and "holdout" not in by_name["noise"]["brief"]
    assert "eval" not in by_name["planted"]

    planted_id = by_name["planted"]["id"]
    assert data(await client.post(f"/api/alpha/runs/{run_id}/decide", json={"decisions": {planted_id: True}},
                                  headers=ADMIN)) == {"resumed": True}
    tail = await events(client, run_id, after=evts[-1]["data"]["lastId"])
    assert [e["event"] for e in tail][-2:] == ["done", "end"] and tail[-1]["data"]["status"] == "done"
    assert tail[-1]["data"]["lastId"] == tail[-2]["id"]
    assert all(e["id"] > evts[-2]["id"] for e in tail[:-1])

    resp = await client.post(f"/api/alpha/runs/{run_id}/decide", json={"decisions": {}}, headers=ADMIN)
    assert resp.status_code == 409 and resp.json()["code"] == 409

    lib = data(await client.get("/api/alpha/library", headers=ADMIN))
    factors = lib["factors"]
    assert len(factors) == 1 and factors[0]["expr"] == "Mean($volume, 4)" and "eval" not in factors[0]
    assert factors[0]["brief"]["horizons"]["12h"]["meanIc"] > 0.3 and "holdout" in factors[0]["brief"]
    # 入库后库的组合就有了：搜索段三档加 holdout、主档权重、每币最后一根的信号
    combo = lib["combo"]
    assert [w["name"] for w in combo["weights"]] == ["planted"] and combo["weights"][0]["weight"] > 0
    assert combo["brief"]["horizons"]["12h"]["meanIc"] > 0.3 and combo["brief"]["holdout"]["horizons"]["12h"]["samples"] > 0
    assert set(combo["latest"]) == {"AAA", "BBB", "CCC"} and "surrogate" not in combo["brief"]
    assert combo["asOf"] == int(planted_panels["AAA"].index[-1])
    full = data(await client.get(f"/api/alpha/library/{factors[0]['id']}", headers=ADMIN))
    assert "eval" in full and full["gates"][0]["id"] == "G0"

    recheck = data(await client.post(f"/api/alpha/library/{factors[0]['id']}/recheck", headers=ADMIN))
    assert recheck["horizons"]["48"]["samples"] == 0
    assert data(await client.get(f"/api/alpha/library/{factors[0]['id']}", headers=ADMIN))["recheck"]["since"] == full["approved_at"]

    runs = data(await client.get("/api/alpha/runs", headers=ADMIN))
    assert runs["runs"][0]["id"] == run_id and runs["runs"][0]["status"] == "done" and runs["trials"] == 3
    assert (await client.get("/api/alpha/runs/nope", headers=ADMIN)).status_code == 404

    # 入库之后它就进了冗余对比集：再评估同一条表达式，G6 认出是库里的那条
    out = data(await client.post("/api/alpha/eval", json={"expr": "Mean($volume, 4)"}, headers=ADMIN))
    assert out["brief"]["redundancy"]["with"] == "lib:planted" and out["brief"]["redundancy"]["maxAbs"] > 0.99
    assert not out["passed"]


async def test_failed_round_marks_pending_candidates(client):
    def boom(prompt: str) -> str:
        if "写成一条表达式" in prompt:
            raise RuntimeError("上游挂了")
        return gen_reply(prompt)

    m.app.state.models = (model(boom), model(review_reply))
    run_id = data(await client.post("/api/alpha/runs", json={"k": 2}, headers=ADMIN))["id"]
    evts = await events(client, run_id)
    assert evts[-1]["data"]["status"] == "failed" and any(e["event"] == "error" for e in evts)
    # 停在 proposed 的两条都标成 failed，而且推了事件，开着的页面能看到
    failed = [e["data"]["name"] for e in evts if e["event"] == "candidate" and e["data"]["status"] == "failed"]
    assert sorted(failed) == ["noise", "planted"]
    state = data(await client.get(f"/api/alpha/runs/{run_id}", headers=ADMIN))
    assert state["run"]["status"] == "failed"
    assert all(c["status"] == "failed" and c["reason"] == "这一轮没跑完" for c in state["candidates"])


async def test_only_one_round_runs_at_a_time(client):
    run_id = data(await client.post("/api/alpha/runs", json={"k": 2}, headers=ADMIN))["id"]
    resp = await client.post("/api/alpha/runs", json={"k": 1}, headers=ADMIN)
    assert resp.status_code == 409
    await events(client, run_id)  # 等这一轮停在审批
    assert data(await client.post("/api/alpha/runs", json={"k": 1}, headers=ADMIN))["id"] != run_id


# ========== 其它接口 ==========


async def test_eval_endpoint(client):
    out = data(await client.post("/api/alpha/eval", json={"expr": "$volume"}, headers=ADMIN))
    assert out["passed"] is True and len(out["gates"]) == 10 and out["canonical"] == "$volume"
    assert out["brief"]["redundancy"]["with"] == "noise4" and out["brief"]["redundancy"]["maxAbs"] < 0.2
    assert out["gates"][9]["id"] == "G9" and out["brief"]["gain"]["12h"] > 0.3
    # 手填的也带 holdout：门看搜索段，holdout 另算
    assert out["brief"]["holdout"]["horizons"]["12h"]["meanIc"] > 0 and out["eval"]["holdout"]["since"] == out["brief"]["holdout"]["since"]
    resp = await client.post("/api/alpha/eval", json={"expr": "Mean(close, 5)"}, headers=ADMIN)
    assert resp.status_code == 400 and "列要写成" in resp.json()["msg"]
    assert (await client.post("/api/alpha/runs", json={"k": 99}, headers=ADMIN)).status_code == 422


async def test_auth_rules(client):
    resp = await client.get("/api/alpha/library")
    assert resp.status_code == 200 and resp.json()["code"] == 401
    resp = await client.get("/api/alpha/library", headers={"satoken": "user"})
    assert resp.status_code == 403 and resp.json()["code"] == 403


async def test_data_status_and_refresh(client):
    status = data(await client.get("/api/alpha/data/status", headers=ADMIN))
    assert status["loaded"] and set(status["symbols"]) == {"AAA", "BBB", "CCC"} and status["facts"]["rows"] == 6000
    # 6000 根从 2025-01-01 起共 62.5 天，锁最近 10 天
    assert status["facts"]["holdoutFrom"] == "2025-02-22"
    before = status["loadedAt"]
    assert data(await client.post("/api/alpha/data/refresh", headers=ADMIN)) == {"started": True}

    async def finished():
        s = data(await client.get("/api/alpha/data/status", headers=ADMIN))
        return not s["refresh"]["running"] and s["refresh"].get("error") is None

    await wait_until(finished)
    assert data(await client.get("/api/alpha/data/status", headers=ADMIN))["loadedAt"] >= before


async def test_reference_run(client):
    assert data(await client.post("/api/alpha/reference/run", json={}, headers=ADMIN))["total"] == 1

    async def finished():
        return not data(await client.get("/api/alpha/reference", headers=ADMIN))["job"]["running"]

    await wait_until(finished)
    out = data(await client.get("/api/alpha/reference", headers=ADMIN))
    assert out["job"]["done"] == 1 and out["items"][0]["name"] == "noise4" and isinstance(out["items"][0]["passed"], bool)
    assert out["items"][0]["brief"]["gates"][0]["id"] == "G0"


async def test_restart_marks_running_runs_failed(prepared):
    async with m.lifespan(prepared):
        await prepared.state.store.create_run("zz", 1, "", "zh")
        await prepared.state.store.upsert_candidate({"id": "zz-1", "run_id": "zz", "seq": 1, "name": "x",
                                                     "hypothesis": "h", "status": "evaluating"})
    async with m.lifespan(prepared):
        run = await prepared.state.store.run("zz")
        assert run["status"] == "failed" and "重启" in run["error"]
        cand = (await prepared.state.store.candidates("zz"))[0]
        assert cand["status"] == "failed" and "重启" in cand["reason"]
