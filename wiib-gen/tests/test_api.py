import json
import time
from typing import Any

import httpx
import pytest
from langchain_core.exceptions import ModelAuthenticationError
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from wiib_gen import graph, main, prompts
from wiib_gen.config import config
from wiib_gen.llm import LlmError
from wiib_gen.platform import AuthError

USERS = {"tok-a": 7, "tok-b": 9}
SPEC = {"leverageMin": 3, "leverageMax": 20, "marginPctMin": 5, "marginPctMax": 20,
        "allowMultiPosition": True, "allowHedge": False}
MINE = {"pub": {"symbols": "BTCUSDT,ETHUSDT", "intervalCode": "1h"},
        "spec": SPEC, "customPrompt": "老指令正文", "wakeWindow": None, "llmEndpointId": 3}
DETAIL_JSON = '{"key": "exit_rule", "text": "怎么退出？", "options": ["固定止盈", "跟踪止损"]}'
MARK = prompts.NOTES_MARK


class FakePlatform:
    """mine_view 改成 None 就是「还没建 trader」那条路"""

    mine_view: Any = MINE

    def __init__(self, token, lang=None, client=None):
        self.token = token

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def current_user_id(self):
        if self.token not in USERS:
            raise AuthError("登录态失效")
        return USERS[self.token]

    async def mine(self):
        return type(self).mine_view

    async def default_instructions(self):
        return "【方法】默认底稿"

    async def prompt_template(self, interval_code, symbols, spec, wake_window):
        return "平台系统提示词全文"


class ScriptedModel(GenericFakeChatModel):
    reply: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.reply("\n".join(m.text for m in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])


def reply(text: str) -> str:
    if "只输出一个 JSON 对象" in text:
        return DETAIL_JSON
    return f"【方法】跟着趋势做\n{MARK}\n第一版"


def use_model(monkeypatch, fn) -> None:
    monkeypatch.setattr(graph, "build_chat_model",
                        lambda row: ScriptedModel(messages=iter(()), reply=fn))


@pytest.fixture
async def client(tmp_path, monkeypatch):
    monkeypatch.setenv("GEN_DB_PATH", str(tmp_path / "gen.sqlite"))
    config.cache_clear()
    main._auth_cache.clear()
    monkeypatch.setattr(main, "Platform", FakePlatform)
    monkeypatch.setattr(main, "load_endpoint", lambda endpoint_id, user_id: {"id": endpoint_id})
    monkeypatch.setattr(graph, "load_endpoint", lambda endpoint_id, user_id: {"id": endpoint_id})
    use_model(monkeypatch, reply)
    async with main.app.router.lifespan_context(main.app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                     base_url="http://gen") as c:
            yield c
    config.cache_clear()


def head(token: str = "tok-a") -> dict[str, str]:
    return {"satoken": token, "X-Lang": "zh"}


def events(response: httpx.Response) -> list[tuple[str, Any]]:
    out = []
    for raw in response.text.strip().split("\n\n"):
        # 心跳是注释帧，不算事件
        if raw.startswith(":"):
            continue
        name, data = raw.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def shape(evts: list[tuple[str, Any]]) -> list[str]:
    """事件名序列，连着的同名压成一条（token 一串、status 重复都压掉）"""
    names = []
    for name, _ in evts:
        if not names or names[-1] != name:
            names.append(name)
    return names


def pick(evts: list[tuple[str, Any]], name: str) -> Any:
    return next(data for event, data in evts if event == name)


async def start(client: httpx.AsyncClient) -> tuple[str, list[tuple[str, Any]]]:
    resp = await client.post("/api/gen/threads", json={"endpointId": 1, "outputLang": "zh"},
                             headers=head())
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    assert resp.headers["x-accel-buffering"] == "no"
    evts = events(resp)
    return pick(evts, "thread")["id"], evts


async def resume(client: httpx.AsyncClient, thread_id: str, body: dict, token: str = "tok-a"):
    return await client.post(f"/api/gen/threads/{thread_id}/resume", json=body, headers=head(token))


async def to_first_version(client: httpx.AsyncClient) -> str:
    """开线程 → 答增量/从零 → 答基础三题，停在打法题上"""
    thread_id, _ = await start(client)
    await resume(client, thread_id, {"answer": {"mode": "从零写一份"}})
    await resume(client, thread_id, {"answer": {
        "direction": "只做多", "holding": "更长", "style": "保守"}})
    return thread_id


# ========== 一条线走完 ==========


async def test_sse_line(client):
    thread_id, evts = await start(client)
    assert shape(evts) == ["thread", "status", "question"]
    assert pick(evts, "question")["kind"] == "mode"
    assert pick(evts, "status") == {"stage": "asking"}

    evts = events(await resume(client, thread_id, {"answer": {"mode": "从零写一份"}}))
    assert shape(evts) == ["status", "question"]
    assert pick(evts, "question")["kind"] == "base"

    evts = events(await resume(client, thread_id, {"answer": {
        "direction": "只做多", "holding": "波段 1~3 天", "style": "均衡"}}))
    assert shape(evts) == ["status", "token", "version", "status", "question"]
    assert [d["stage"] for _, d in evts if _ == "status"] == ["asking", "generating", "proposing", "asking"]
    version = pick(evts, "version")
    assert version == {"no": 1, "text": "【方法】跟着趋势做", "notes": "第一版"}
    streamed = "".join(d["text"] for n, d in evts if n == "token")
    assert streamed.startswith("【方法】跟着趋势做")
    assert pick(evts, "question")["kind"] == "play"

    evts = events(await resume(client, thread_id, {"answer": {"play": "趋势跟随"}}))
    assert pick(evts, "version")["no"] == 2
    assert pick(evts, "question")["items"][0]["key"] == "exit_rule"

    state = (await client.get(f"/api/gen/threads/{thread_id}", headers=head())).json()["data"]
    assert [v["no"] for v in state["versions"]] == [1, 2]
    assert state["question"]["kind"] == "detail"
    assert state["done"] is False
    assert state["pending"] is True

    rolled = (await client.post(f"/api/gen/threads/{thread_id}/rollback",
                                json={"version": 1}, headers=head())).json()
    assert rolled["code"] == 0
    rolled = rolled["data"]
    assert [v["no"] for v in rolled["versions"]] == [1]
    assert rolled["question"]["kind"] == "play"

    evts = events(await resume(client, thread_id, {"answer": {"play": "突破"}}))
    assert pick(evts, "version")["no"] == 2
    state = (await client.get(f"/api/gen/threads/{thread_id}", headers=head())).json()["data"]
    assert len(state["versions"]) == 2


async def test_finish_sends_done(client):
    thread_id = await to_first_version(client)
    evts = events(await resume(client, thread_id, {"finish": True}))
    assert shape(evts) == ["status", "done"]
    state = (await client.get(f"/api/gen/threads/{thread_id}", headers=head())).json()["data"]
    assert state["done"] is True
    assert state["question"] is None
    assert state["pending"] is False


# ========== 鉴权与归属 ==========


async def test_other_user_thread_is_404(client):
    thread_id, _ = await start(client)
    for resp in (await client.get(f"/api/gen/threads/{thread_id}", headers=head("tok-b")),
                 await resume(client, thread_id, {"skip": True}, token="tok-b")):
        assert resp.status_code == 404
        assert resp.json()["code"] == 404


async def test_missing_token_is_json_401(client):
    """未登录跟 Java GlobalExceptionHandler 一个口径：HTTP 200 带 code=401"""
    resp = await client.post("/api/gen/threads", json={"endpointId": 1, "outputLang": "zh"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"code": 401, "msg": "未登录"}


async def test_bad_token_is_json_401(client):
    resp = await client.get("/api/gen/threads/whatever", headers=head("tok-x"))
    assert resp.status_code == 200
    assert resp.json() == {"code": 401, "msg": "登录态失效"}


@pytest.mark.parametrize("exc, message", [
    (LlmError("端点 1 不存在或不属于当前用户"), "端点 1 不存在或不属于当前用户"),
    # 上游模型的错按类别说人话
    (ModelAuthenticationError("Incorrect API key"), "API key 无效或无权限，请到配置里检查"),
    # 不认识的异常不把内部细节下发
    (RuntimeError("psycopg connect 10.0.0.7:5432 failed"), "生成失败，请重试"),
])
async def test_error_event_wording(client, monkeypatch, exc, message):
    thread_id, _ = await start(client)
    await resume(client, thread_id, {"answer": {"mode": "从零写一份"}})

    def boom(row):
        raise exc

    monkeypatch.setattr(graph, "build_chat_model", boom)
    evts = events(await resume(client, thread_id, {"answer": {
        "direction": "只做多", "holding": "更长", "style": "保守"}}))
    assert shape(evts)[-1] == "error"
    assert pick(evts, "error")["message"] == message


async def test_unhandled_endpoint_error_is_json_500(client, monkeypatch):
    """兜底处理器：普通接口挂了也是 Result 形状，不是 text/plain"""
    thread_id, _ = await start(client)

    def boom(snapshot):
        raise RuntimeError("psycopg connect 10.0.0.7:5432 failed")

    monkeypatch.setattr(main, "_view", boom)
    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False),
            base_url="http://gen") as c:
        resp = await c.get(f"/api/gen/threads/{thread_id}", headers=head())
    assert resp.status_code == 500
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"code": 500, "msg": "服务内部错误"}


# ========== 生成断了还有出口 ==========


async def test_broken_generate_is_pending_and_resumable(client, monkeypatch):
    """第 2 版生成断了：刷新看到的不该是已经答过的 play 题；空 resume 把挂住的 generate 重跑"""
    thread_id = await to_first_version(client)

    def boom(row):
        raise LlmError("上游抽风")

    monkeypatch.setattr(graph, "build_chat_model", boom)
    evts = events(await resume(client, thread_id, {"answer": {"play": "趋势跟随"}}))
    assert shape(evts)[-1] == "error"
    # error 帧里带着出错后的线程快照，刷新页面拿到的也是它
    state = pick(evts, "error")["state"]
    assert state == (await client.get(f"/api/gen/threads/{thread_id}", headers=head())).json()["data"]
    assert state["question"] is None and state["pending"] is True
    assert [v["no"] for v in state["versions"]] == [1]

    use_model(monkeypatch, reply)
    evts = events(await resume(client, thread_id, {}))
    assert pick(evts, "version")["no"] == 2
    assert pick(evts, "question")["items"][0]["key"] == "exit_rule"


# ========== 心跳 ==========


async def test_silent_graph_sends_pings(client, monkeypatch):
    """模型思考期一个 token 都没有，也得隔一会儿发帧注释撑住连接"""
    monkeypatch.setattr(main, "_PING_SECONDS", 0.01)

    def slow_endpoint(endpoint_id, user_id):
        time.sleep(0.05)
        return {"id": endpoint_id}

    monkeypatch.setattr(graph, "load_endpoint", slow_endpoint)
    thread_id, _ = await start(client)
    await resume(client, thread_id, {"answer": {"mode": "从零写一份"}})
    resp = await resume(client, thread_id, {"answer": {
        "direction": "只做多", "holding": "更长", "style": "保守"}})
    assert ": ping\n\n" in resp.text
    assert pick(events(resp), "version")["no"] == 1


# ========== 端点先验 ==========


async def test_unknown_endpoint_is_400_before_any_thread(client, monkeypatch):
    def boom(endpoint_id, user_id):
        raise LlmError(f"端点 {endpoint_id} 不存在或不属于当前用户")

    monkeypatch.setattr(main, "load_endpoint", boom)
    resp = await client.post("/api/gen/threads", json={"endpointId": 99, "outputLang": "zh"},
                             headers=head())
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {"code": 400, "msg": "端点 99 不存在或不属于当前用户"}
    async with main.app.state.db.execute("SELECT COUNT(*) FROM gen_thread") as cur:
        assert (await cur.fetchone())[0] == 0


# ========== 超长首稿重写 ==========


async def test_long_first_draft_restarts_the_body(client, monkeypatch):
    def long_reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            return DETAIL_JSON
        if "压到" in text:
            return f"压好的正文\n{MARK}\n压过了"
        return "超长稿" * 1500 + f"\n{MARK}\n第一版"

    use_model(monkeypatch, long_reply)
    thread_id, _ = await start(client)
    await resume(client, thread_id, {"answer": {"mode": "从零写一份"}})
    evts = events(await resume(client, thread_id, {"answer": {
        "direction": "只做多", "holding": "更长", "style": "保守"}}))

    marks = [i for i, (name, data) in enumerate(evts)
             if name == "status" and data["stage"] == "generating"]
    assert len(marks) == 2
    rewritten = "".join(d["text"] for n, d in evts[marks[1]:] if n == "token")
    assert rewritten.startswith("压好的正文")
    assert "超长稿" not in rewritten
    assert pick(evts, "version")["text"] == "压好的正文"


# ========== 没建 trader 走平台默认 ==========


async def test_no_trader_falls_back_to_defaults(client, monkeypatch):
    monkeypatch.setattr(FakePlatform, "mine_view", None)
    thread_id, evts = await start(client)
    # 没有现成指令，不问「增量还是从零」
    assert pick(evts, "question")["kind"] == "base"

    facts = (await main.app.state.graph.aget_state(main._cfg(thread_id))).values["facts"]
    assert facts["has_trader"] is False
    assert facts["spec"] == main.DEFAULT_SPEC
    assert facts["symbols"] == main.DEFAULT_SYMBOLS
    assert facts["interval_code"] == main.DEFAULT_INTERVAL
    assert facts["wake_window"] is None
    assert facts["existing_prompt"] == ""


async def test_rollback_to_missing_version_is_400(client):
    thread_id, _ = await start(client)
    resp = await client.post(f"/api/gen/threads/{thread_id}/rollback",
                             json={"version": 3}, headers=head())
    assert resp.status_code == 400
    assert resp.json()["code"] == 400
