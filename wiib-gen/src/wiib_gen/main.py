"""FastAPI：鉴权、线程归属、四个端点、SSE。

平台事实在请求里拉齐（satoken 只在这一层出现），算好 risk_table 当图的初始输入。
线程归属表与 checkpoint 同一个 SQLite 文件，不往 PG 写东西。
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

import aiosqlite
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, StateSnapshot
from pydantic import BaseModel

from . import prompts
from .config import config
from .crypto import CryptoError
from .graph import build_graph
from .llm import LlmError, load_endpoint
from .platform import AuthError, Platform, PlatformError

logger = logging.getLogger(__name__)

# 没建 trader 时用平台默认（与 MyTrader.tsx 的 DEFAULT_SPEC / EMPTY_FORM 同值）
DEFAULT_SPEC = {"leverageMin": 3, "leverageMax": 20, "marginPctMin": 5, "marginPctMax": 20,
                "allowMultiPosition": True, "allowHedge": False}
DEFAULT_SYMBOLS = "BTCUSDT"
DEFAULT_INTERVAL = "15m"

_THREAD_TABLE = ("CREATE TABLE IF NOT EXISTS gen_thread("
                 "thread_id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created_at INTEGER NOT NULL)")


# ========== 鉴权 ==========

_AUTH_TTL = 60.0
_auth_cache: dict[str, tuple[int, float]] = {}


@dataclass(frozen=True)
class Auth:
    user_id: int
    token: str
    lang: str


async def auth(request: Request) -> Auth:
    token = request.headers.get("satoken")
    if not token:
        raise AuthError("未登录")
    lang = prompts.pick_lang(request.headers.get("X-Lang"))
    hit = _auth_cache.get(token)
    now = time.monotonic()
    if hit and hit[1] > now:
        return Auth(hit[0], token, lang)
    async with Platform(token, lang) as platform:
        user_id = await platform.current_user_id()
    _auth_cache[token] = (user_id, now + _AUTH_TTL)
    return Auth(user_id, token, lang)


# ========== 平台事实 ==========


async def _facts(who: Auth, endpoint_id: int, output_lang: str) -> dict[str, Any]:
    async with Platform(who.token, who.lang) as platform:
        mine = await platform.mine()
        default_instructions = await platform.default_instructions()
        symbols = mine["pub"]["symbols"] if mine else DEFAULT_SYMBOLS
        interval = mine["pub"]["intervalCode"] if mine else DEFAULT_INTERVAL
        spec = mine["spec"] if mine else DEFAULT_SPEC
        wake_window = mine["wakeWindow"] if mine else None
        template = await platform.prompt_template(interval, symbols, spec, wake_window)
    return {
        "mode": "scratch",
        "output_lang": prompts.pick_lang(output_lang),
        "dialog_lang": who.lang,
        "endpoint_id": endpoint_id,
        "user_id": who.user_id,
        "has_trader": mine is not None,
        "spec": spec,
        "symbols": symbols,
        "interval_code": interval,
        "wake_window": wake_window,
        "template": template,
        "default_instructions": default_instructions,
        "existing_prompt": (mine or {}).get("customPrompt") or "",
        "risk_table": prompts.risk_table(spec, who.lang),
    }


# ========== SSE ==========

_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
# 节点 → 页面上的状态；连着的同一状态只发一次
_STAGE = {"ask_mode": "asking", "ask_base": "asking", "ask_next": "asking",
          "propose_next": "asking", "generate": "generating"}


def _frame(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _events(graph, payload: Any, thread_id: str,
                  head: dict[str, Any] | None) -> AsyncIterator[str]:
    if head is not None:
        yield _frame("thread", head)
    cfg = _cfg(thread_id)
    stage = None
    try:
        async for mode, chunk in graph.astream(
                payload, cfg, stream_mode=["tasks", "updates", "messages", "custom"],
                durability="sync"):
            if mode == "tasks" and "input" in chunk:
                nxt = _STAGE.get(chunk["name"])
                if nxt and nxt != stage:
                    stage = nxt
                    yield _frame("status", {"stage": stage})
            elif mode == "custom" and chunk.get("restart"):
                # 超长重写：再发一次 generating，页面据此把首稿的字清掉
                yield _frame("status", {"stage": "generating"})
            elif mode == "messages":
                message, meta = chunk
                if meta.get("langgraph_node") == "generate" and message.text:
                    yield _frame("token", {"text": message.text})
            elif mode == "updates":
                if "__interrupt__" in chunk:
                    yield _frame("question", chunk["__interrupt__"][0].value)
                elif "generate" in chunk:
                    yield _frame("version", chunk["generate"]["versions"][-1])
    except (LlmError, PlatformError, CryptoError) as exc:
        yield _frame("error", {"message": str(exc)})
        return
    except Exception:
        # 内部细节不下发，日志里留全栈
        logger.exception("生成失败 thread=%s", thread_id)
        yield _frame("error", {"message": "生成失败，请重试"})
        return
    if not (await graph.aget_state(cfg)).next:
        yield _frame("done", {})


def _sse(graph, payload: Any, thread_id: str, head: dict[str, Any] | None = None) -> StreamingResponse:
    return StreamingResponse(_events(graph, payload, thread_id, head),
                             media_type="text/event-stream", headers=_SSE_HEADERS)


# ========== 线程归属 ==========


def _cfg(thread_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": thread_id}}


async def _remember(request: Request, thread_id: str, user_id: int) -> None:
    db = request.app.state.db
    await db.execute("INSERT INTO gen_thread(thread_id, user_id, created_at) VALUES (?, ?, ?)",
                     (thread_id, user_id, int(time.time())))
    await db.commit()


async def _own(request: Request, thread_id: str, user_id: int) -> None:
    """别人的线程一律当不存在"""
    async with request.app.state.db.execute(
            "SELECT user_id FROM gen_thread WHERE thread_id = ?", (thread_id,)) as cur:
        row = await cur.fetchone()
    if row is None or row[0] != user_id:
        raise HTTPException(404, "线程不存在")


def _view(snapshot: StateSnapshot) -> dict[str, Any]:
    """刷新页面与回退后都看这个形状"""
    values = snapshot.values
    # 停在 interrupt 上时题目在 interrupt 里，刚分叉回退时题目在 state 里
    question = snapshot.interrupts[0].value if snapshot.interrupts else values.get("question")
    return {
        "versions": values.get("versions") or [],
        "question": question if snapshot.next else None,
        "done": bool(values.get("done")),
        # 图还停在半路（生成断了也算），前端据此给「继续生成」
        "pending": bool(snapshot.next),
    }


# ========== 端点 ==========

router = APIRouter(prefix="/api/gen")


class CreateBody(BaseModel):
    endpointId: int
    outputLang: str


class ResumeBody(BaseModel):
    answer: dict[str, str] | None = None
    skip: bool = False
    finish: bool = False


class RollbackBody(BaseModel):
    version: int


@router.post("/threads")
async def create_thread(body: CreateBody, request: Request, who: Auth = Depends(auth)):
    # 端点不对先 400 掉，别留一条开不动的线程
    await asyncio.to_thread(load_endpoint, body.endpointId, who.user_id)
    facts = await _facts(who, body.endpointId, body.outputLang)
    thread_id = uuid.uuid4().hex
    await _remember(request, thread_id, who.user_id)
    state = {"facts": facts, "answers": [], "asked": [], "question": None,
             "versions": [], "done": False}
    return _sse(request.app.state.graph, state, thread_id, {"id": thread_id})


@router.post("/threads/{thread_id}/resume")
async def resume_thread(thread_id: str, body: ResumeBody, request: Request, who: Auth = Depends(auth)):
    await _own(request, thread_id, who.user_id)
    return _sse(request.app.state.graph, Command(resume=body.model_dump()), thread_id)


def _ok(data: dict[str, Any]) -> dict[str, Any]:
    """与 Java 侧 Result 同形，前端同一个拦截器解包"""
    return {"code": 0, "msg": "", "data": data}


@router.get("/threads/{thread_id}")
async def thread_state(thread_id: str, request: Request, who: Auth = Depends(auth)):
    await _own(request, thread_id, who.user_id)
    return _ok(_view(await request.app.state.graph.aget_state(_cfg(thread_id))))


@router.post("/threads/{thread_id}/rollback")
async def rollback(thread_id: str, body: RollbackBody, request: Request, who: Auth = Depends(auth)):
    await _own(request, thread_id, who.user_id)
    graph = request.app.state.graph
    cfg = _cfg(thread_id)
    target = None
    async for snapshot in graph.aget_state_history(cfg):
        # 该版刚生成完、正停在等答题那一步的那个 checkpoint
        if snapshot.next == ("ask_next",) and len(snapshot.values.get("versions") or []) == body.version:
            target = snapshot
            break
    if target is None:
        raise HTTPException(400, f"没有第 {body.version} 版可回退")
    # 空更新即分叉：线程头挪到该 checkpoint，旧分支留在 history 里
    await graph.aupdate_state(target.config, values={})
    return _ok(_view(await graph.aget_state(cfg)))


# ========== 应用 ==========


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = config().gen_db_path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(path) as saver, aiosqlite.connect(path) as db:
        await db.execute(_THREAD_TABLE)
        await db.commit()
        app.state.graph = build_graph(saver)
        app.state.db = db
        yield


def _error(code: int, msg: str) -> JSONResponse:
    """前端 consumeSse 见 application/json 就按接口错误处理。
    401 跟 Java GlobalExceptionHandler 一个口径：HTTP 200 带 code=401，前端拦截器认这个跳登录页
    """
    return JSONResponse({"code": code, "msg": msg}, status_code=200 if code == 401 else code)


def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """兜底：没接住的异常也走 Result 形状，别让 FastAPI 吐 text/plain"""
    logger.exception("未处理的异常 %s %s", request.method, request.url.path)
    return _error(500, "服务内部错误")


app = FastAPI(lifespan=lifespan, title="wiib-gen")
app.include_router(router)
app.add_exception_handler(AuthError, lambda request, exc: _error(401, str(exc)))
app.add_exception_handler(PlatformError, lambda request, exc: _error(502, str(exc)))
app.add_exception_handler(LlmError, lambda request, exc: _error(400, str(exc)))
app.add_exception_handler(HTTPException, lambda request, exc: _error(exc.status_code, exc.detail))
app.add_exception_handler(Exception, _unhandled)
