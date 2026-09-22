"""FastAPI：鉴权、数据、跑一轮、SSE、审批、因子库、参照集、手填评估。

图跑在服务端后台任务里，不绑定 HTTP 连接：关页面不会杀掉一轮。事件先落库再推给在线订阅者，刷新页从库里回放。
返回体与 Java Result 同形；401 走 HTTP 200 + code 401，前端拦截器认这个跳登录页。
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from . import reference
from .auth import Auth, AuthError, Authenticator, Forbidden, UpstreamError
from .config import Config, config
from .data.binance import Binance
from .data.panel import FEATURE_COLUMNS, PanelStore, describe
from .dsl import DslError, parse
from .eval.evaluate import evaluate
from .eval.gates import judge, passed
from .eval.params import EvalParams, Thresholds
from .eval.recheck import recheck
from .graph import Context, brief, build_graph
from .llm import LlmError, build_chat_model
from .prompts import eval_brief
from .store import Store, clean, now_ms

logger = logging.getLogger(__name__)

MAX_K = 10
QUEUE_LIMIT = 5000
_SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
_PING_SECONDS = 15.0


# ========== 数据上下文 ==========


@dataclass
class Data:
    panels: dict[str, pd.DataFrame]
    references: dict[str, pd.DataFrame]
    reference_canonicals: set[str]
    facts: dict[str, Any]
    loaded_at: int


def load_data(cfg: Config) -> Data | None:
    """读 parquet 面板、算参照集；一张面板都没有就是 None（先刷新数据）"""
    store = PanelStore(cfg.data_dir)
    panels = {s: store.load(s) for s in cfg.symbols if (store.panels / f"{s}.parquet").exists()}
    if not panels:
        return None
    exprs = reference.parsed()
    return Data(
        panels=panels,
        references=reference.values(panels, exprs),
        reference_canonicals={e.canonical() for e in exprs.values()},
        facts=_facts(panels),
        loaded_at=now_ms(),
    )


def _facts(panels: dict[str, pd.DataFrame]) -> dict[str, Any]:
    described = {s: describe(p) for s, p in panels.items()}
    liq = [d["liqFrom"] for d in described.values() if d["liqFrom"]]
    return {
        "symbols": list(panels),
        "from": _day(min(d["from"] for d in described.values())),
        "to": _day(max(d["to"] for d in described.values())),
        "rows": min(d["rows"] for d in described.values()),
        "liqFrom": _day(min(liq)) if liq else None,
    }


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def build_models(cfg: Config) -> tuple[Any, Any]:
    return build_chat_model(cfg.gen, "gen"), build_chat_model(cfg.review, "review")


# ========== 后台任务与订阅 ==========


class Runs:
    """在跑的任务与在线订阅者"""

    def __init__(self) -> None:
        self.tasks: dict[str, asyncio.Task] = {}
        self.subscribers: dict[str, set[asyncio.Queue]] = {}
        self.jobs: dict[str, dict[str, Any]] = {"refresh": {"running": False}, "reference": {"running": False}}

    def subscribe(self, run_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_LIMIT)
        self.subscribers.setdefault(run_id, set()).add(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: asyncio.Queue) -> None:
        queues = self.subscribers.get(run_id)
        if queues is None:
            return
        queues.discard(queue)
        if not queues:
            del self.subscribers[run_id]

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        """订阅端读得太慢队列满了就丢这条；事件都在库里，重连按 id 能补回来"""
        for queue in self.subscribers.get(run_id, ()):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                pass


async def _publish(app: FastAPI, run_id: str, event: str, data: Any) -> None:
    """main 自己发的事件也先落库再推"""
    data = clean(data)
    event_id = await app.state.store.add_event(run_id, event, data)
    app.state.runs.publish(run_id, {"id": event_id, "event": event, "data": data})


async def _drive(app: FastAPI, run_id: str, payload: Any) -> None:
    store: Store = app.state.store
    cfg = {"configurable": {"thread_id": run_id}}
    try:
        graph = await _graph(app)
        # checkpoint 同步落盘：interrupt 一出来就能被另一个图实例接着跑，不会读到旧状态
        async for mode, chunk in graph.astream(payload, cfg, stream_mode=["updates", "custom"], durability="sync"):
            if mode == "custom":
                app.state.runs.publish(run_id, chunk)
            elif mode == "updates" and "__interrupt__" in chunk:
                await store.set_run_status(run_id, "awaiting")
                await _publish(app, run_id, "awaiting", chunk["__interrupt__"][0].value)
    except Exception as exc:
        logger.exception("run %s 失败", run_id)
        message = str(exc) if isinstance(exc, (LlmError, DslError)) else "运行失败，看服务日志"
        await store.set_run_status(run_id, "failed", str(exc)[:500])
        await _publish(app, run_id, "error", {"message": message})
    finally:
        app.state.runs.tasks.pop(run_id, None)


async def _graph(app: FastAPI):
    """每次现建 Context：库摘要取当下的，模型第一次建好后复用"""
    data: Data = _data(app)
    if app.state.models is None:
        app.state.models = build_models(config())
    gen, review = app.state.models
    references, known, factors = await _references(app)
    library = [{"name": f["name"], "expr": f["expr"], "hypothesis": f["hypothesis"], "canonical": f["canonical"]}
               for f in factors]
    ctx = Context(gen=gen, review=review, panels=data.panels, references=references, known=known,
                  library=library, facts=data.facts, store=app.state.store, params=app.state.params,
                  thresholds=app.state.thresholds)
    return build_graph(app.state.saver, ctx)


async def _references(app: FastAPI) -> tuple[dict[str, pd.DataFrame], set[str], list[dict[str, Any]]]:
    """冗余对比集 = 参照集 + 库里的因子（列名 lib:名字）；库或数据变了才重算"""
    data: Data = _data(app)
    factors = await app.state.store.factors()
    key = (data.loaded_at, len(factors), max((int(f["approved_at"]) for f in factors), default=0))
    cache = app.state.library_refs
    if cache is None or cache[0] != key:
        refs = {s: df.copy() for s, df in data.references.items()}
        for f in factors:
            expr = parse(f["expr"], FEATURE_COLUMNS)
            for s, p in data.panels.items():
                refs[s][f"lib:{f['name']}"] = expr.evaluate(p)
        cache = (key, refs)
        app.state.library_refs = cache
    known = data.reference_canonicals | {f["canonical"] for f in factors}
    return cache[1], known, factors


def _data(app: FastAPI) -> Data:
    if app.state.data is None:
        raise HTTPException(409, "还没有行情数据，先刷新数据")
    return app.state.data


def _start(app: FastAPI, run_id: str, payload: Any) -> None:
    app.state.runs.tasks[run_id] = asyncio.create_task(_drive(app, run_id, payload))


# ========== 鉴权 ==========


async def auth(request: Request) -> Auth:
    return await request.app.state.authenticator(request.headers.get("satoken"), request.headers.get("X-Lang"))


# ========== 接口 ==========

router = APIRouter(prefix="/api/alpha", dependencies=[Depends(auth)])


def _ok(data: Any) -> dict[str, Any]:
    return {"code": 0, "msg": "", "data": clean(data)}


class CreateRun(BaseModel):
    k: int = Field(default=3, ge=1, le=MAX_K)
    note: str = ""


class Decide(BaseModel):
    decisions: dict[str, bool]


class EvalBody(BaseModel):
    expr: str


class ReferenceRun(BaseModel):
    windows: list[int] | None = None


@router.get("/data/status")
async def data_status(request: Request):
    app = request.app
    data: Data | None = app.state.data
    return _ok({
        "loaded": data is not None,
        "loadedAt": data.loaded_at if data else None,
        "facts": data.facts if data else None,
        "symbols": {s: describe(p) for s, p in data.panels.items()} if data else {},
        "referenceCount": len(reference.expressions()),
        "refresh": app.state.runs.jobs["refresh"],
    })


@router.post("/data/refresh")
async def data_refresh(request: Request):
    app = request.app
    job = app.state.runs.jobs["refresh"]
    if job.get("running"):
        raise HTTPException(409, "已经在刷新了")
    job.update(running=True, startedAt=now_ms(), error=None)

    async def work() -> None:
        cfg = config()
        try:
            await asyncio.to_thread(_refresh_panels, cfg)
            app.state.data = await asyncio.to_thread(load_data, cfg)
            app.state.library_refs = None
            job.update(finishedAt=now_ms())
        except Exception as exc:
            logger.exception("刷新数据失败")
            job.update(error=str(exc)[:300])
        finally:
            job["running"] = False

    asyncio.create_task(work())
    return _ok({"started": True})


def _refresh_panels(cfg: Config) -> None:
    PanelStore(cfg.data_dir).refresh(Binance(cfg.binance_vision_url), cfg.symbols, cfg.history_days,
                                     cfg.pg_conninfo if cfg.pg_user else None, cfg.force_order_tz)


@router.post("/runs")
async def create_run(body: CreateRun, request: Request, who: Auth = Depends(auth)):
    app = request.app
    _data(app)
    # 一次只跑一轮：一轮已经是 K 条候选并发评估，再叠一轮内存和模型额度都扛不住
    if app.state.runs.tasks:
        raise HTTPException(409, "有一轮还在跑，等它结束")
    if app.state.models is None:
        app.state.models = build_models(config())
    run_id = uuid.uuid4().hex[:12]
    await app.state.store.create_run(run_id, body.k, body.note.strip(), who.lang)
    _start(app, run_id, {"run_id": run_id, "lang": who.lang, "k": body.k, "note": body.note.strip()})
    return _ok({"id": run_id})


@router.get("/runs")
async def list_runs(request: Request):
    return _ok({"runs": await request.app.state.store.runs(), "trials": await request.app.state.store.trials()})


@router.get("/runs/{run_id}")
async def run_state(run_id: str, request: Request):
    store: Store = request.app.state.store
    run = await store.run(run_id)
    if run is None:
        raise HTTPException(404, "没有这一轮")
    candidates = [brief(c) for c in await store.candidates(run_id)]
    return _ok({"run": run, "candidates": candidates, "trials": await store.trials()})


@router.post("/runs/{run_id}/decide")
async def decide(run_id: str, body: Decide, request: Request):
    app = request.app
    run = await app.state.store.run(run_id)
    if run is None:
        raise HTTPException(404, "没有这一轮")
    previous = app.state.runs.tasks.get(run_id)
    if previous is not None:
        await asyncio.wait({previous})
    # 检查与置位一条 UPDATE：双击、两个标签页各点一次，只有一个能抢到
    if not await app.state.store.claim_run(run_id, "awaiting", "running"):
        raise HTTPException(409, "这一轮不在等审批")
    _start(app, run_id, Command(resume={"decisions": body.decisions}))
    return _ok({"resumed": True})


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str, request: Request, after: int = 0):
    if await request.app.state.store.run(run_id) is None:
        raise HTTPException(404, "没有这一轮")
    return StreamingResponse(_events(request.app, run_id, after), media_type="text/event-stream",
                             headers=_SSE_HEADERS)


async def _events(app: FastAPI, run_id: str, after: int) -> AsyncIterator[str]:
    """先回放库里 id > after 的，再收在线的；运行不在 running 状态时补齐收尾并结束"""
    store: Store = app.state.store
    runs: Runs = app.state.runs
    queue = runs.subscribe(run_id)
    last = after
    try:
        for e in await store.events(run_id, after):
            yield _frame(e)
            last = e["id"]
        while True:
            run = await store.run(run_id)
            if run["status"] != "running" and queue.empty():
                for e in await store.events(run_id, last):
                    yield _frame(e)
                    last = e["id"]
                # lastId 给前端：审批后带着它重连，只收后面的
                yield _frame({"id": last, "event": "end", "data": {"status": run["status"], "lastId": last}})
                return
            try:
                e = await asyncio.wait_for(queue.get(), timeout=_PING_SECONDS)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if e["id"] > last:
                yield _frame(e)
                last = e["id"]
    finally:
        runs.unsubscribe(run_id, queue)


def _frame(e: dict[str, Any]) -> str:
    return f"id: {e['id']}\nevent: {e['event']}\ndata: {json.dumps(e['data'], ensure_ascii=False)}\n\n"


@router.get("/library")
async def library(request: Request):
    rows = await request.app.state.store.factors()
    return _ok({"factors": [_factor_view(f, full=False) for f in rows]})


@router.get("/library/{factor_id}")
async def library_item(factor_id: str, request: Request):
    f = await request.app.state.store.factor(factor_id)
    if f is None:
        raise HTTPException(404, "没有这个因子")
    return _ok(_factor_view(f, full=True))


def _factor_view(f: dict[str, Any], full: bool) -> dict[str, Any]:
    out = {k: f.get(k) for k in ("id", "name", "hypothesis", "expr", "canonical", "columns", "approved_at",
                                  "run_id", "review", "recheck")}
    out["brief"] = eval_brief(f["eval"], f["gates"], _primary_of(f["eval"]))
    if full:
        out["eval"] = f["eval"]
        out["gates"] = f["gates"]
    return out


def _primary_of(ev: dict[str, Any]) -> int:
    keys = [int(k) for k in ev["horizons"]]
    return 48 if 48 in keys else keys[0]


@router.post("/library/{factor_id}/recheck")
async def library_recheck(factor_id: str, request: Request):
    app = request.app
    f = await app.state.store.factor(factor_id)
    if f is None:
        raise HTTPException(404, "没有这个因子")
    data = _data(app)
    expr = parse(f["expr"], FEATURE_COLUMNS)
    result = await asyncio.to_thread(recheck, expr, data.panels, int(f["approved_at"]), app.state.params)
    await app.state.store.set_recheck(factor_id, result)
    return _ok(result)


@router.post("/eval")
async def eval_expression(body: EvalBody, request: Request):
    """手填一条表达式直接评估，不进循环不计试验数"""
    app = request.app
    data = _data(app)
    expr = parse(body.expr, FEATURE_COLUMNS)
    references, known, _ = await _references(app)
    ev = await asyncio.to_thread(evaluate, expr, data.panels, app.state.params, references)
    gates = judge(ev, app.state.thresholds, app.state.params.primary, known)
    return _ok({"expr": str(expr), "canonical": expr.canonical(), "passed": passed(gates),
                "brief": eval_brief(ev, gates, app.state.params.primary), "gates": gates, "eval": ev})


@router.get("/reference")
async def reference_list(request: Request):
    rows = await request.app.state.store.reference_evals()
    return _ok({
        "job": request.app.state.runs.jobs["reference"],
        "items": [{"name": r["name"], "expr": r["expr"], "passed": bool(r["passed"]), "updatedAt": r["updated_at"],
                   "brief": eval_brief(r["eval"], r["gates"], _primary_of(r["eval"]))} for r in rows],
    })


@router.post("/reference/run")
async def reference_run(body: ReferenceRun, request: Request):
    """基线 + Alpha158 过一遍全部门；参照集自己不查冗余、不做 BH"""
    app = request.app
    data = _data(app)
    job = app.state.runs.jobs["reference"]
    if job.get("running"):
        raise HTTPException(409, "参照集已经在跑了")
    exprs = reference.parsed()
    if body.windows:
        keep = {str(w) for w in body.windows}
        exprs = {k: v for k, v in exprs.items() if k in reference.BASELINES or _window_of(k) in keep | {""}}
    job.update(running=True, startedAt=now_ms(), total=len(exprs), done=0, error=None)

    async def work() -> None:
        try:
            for name, expr in exprs.items():
                ev = await asyncio.to_thread(evaluate, expr, data.panels, app.state.params, None)
                gates = judge(ev, app.state.thresholds, app.state.params.primary, set())
                await app.state.store.set_reference_eval(name, str(expr), ev, gates, passed(gates))
                job["done"] += 1
            job.update(finishedAt=now_ms())
        except Exception as exc:
            logger.exception("参照集评估失败")
            job.update(error=str(exc)[:300])
        finally:
            job["running"] = False

    asyncio.create_task(work())
    return _ok({"started": True, "total": len(exprs)})


def _window_of(name: str) -> str:
    return name[len(name.rstrip("0123456789")):]


# ========== 应用 ==========


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = config()
    Path(cfg.db_path).parent.mkdir(parents=True, exist_ok=True)
    app.state.params = getattr(app.state, "params", None) or EvalParams()
    app.state.thresholds = getattr(app.state, "thresholds", None) or Thresholds()
    app.state.models = None
    app.state.library_refs = None
    app.state.runs = Runs()
    app.state.authenticator = getattr(app.state, "authenticator", None) or Authenticator(cfg.sim_base_url)
    async with AsyncSqliteSaver.from_conn_string(cfg.db_path) as saver:
        app.state.saver = saver
        app.state.store = await Store.open(cfg.db_path)
        # 启动中途出错也要把 store 关掉，aiosqlite 的线程不是 daemon，不关进程退不出去
        try:
            # 上次进程死在半路的轮次没法续，明说；等审批的那些 checkpoint 还在，能继续
            for run in await app.state.store.running():
                await app.state.store.set_run_status(run["id"], "failed", "服务重启时中断")
            app.state.data = await asyncio.to_thread(load_data, cfg)
            yield
        finally:
            for task in list(app.state.runs.tasks.values()):
                task.cancel()
            await app.state.store.close()


def _error(code: int, msg: str) -> JSONResponse:
    """前端 consumeSse 见 application/json 就按接口错误处理；401 与 Java 同口径走 HTTP 200"""
    return JSONResponse({"code": code, "msg": msg}, status_code=200 if code == 401 else code)


def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未处理的异常 %s %s", request.method, request.url.path)
    return _error(500, "服务内部错误")


app = FastAPI(lifespan=lifespan, title="wiib-alpha")
app.include_router(router)
app.add_exception_handler(AuthError, lambda request, exc: _error(401, str(exc)))
app.add_exception_handler(Forbidden, lambda request, exc: _error(403, str(exc)))
app.add_exception_handler(UpstreamError, lambda request, exc: _error(502, str(exc)))
app.add_exception_handler(LlmError, lambda request, exc: _error(400, str(exc)))
app.add_exception_handler(DslError, lambda request, exc: _error(400, str(exc)))
app.add_exception_handler(HTTPException, lambda request, exc: _error(exc.status_code, exc.detail))
app.add_exception_handler(Exception, _unhandled)
