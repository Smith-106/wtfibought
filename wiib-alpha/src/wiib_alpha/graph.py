"""挖掘图：提假设 → 每条候选各自实现/评估/判门/评审 → 同批 BH → 幸存者算 holdout → 等审批 → 入库。

评估、门、评审、BH 全部只看搜索段（cut 之前）；holdout 只给同批 BH 之后的幸存者算一次 IC，
结果进 eval 和审批卡片，不进评审提示词，也不写进 tried 回灌给模型。
panels、参照集、模型、store 都在 Context 里，不进 state。interrupt 节点无副作用，入库在它之后的独立节点。
候选节点的落库都是按 id 覆盖写，节点重跑不会留重复。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send, interrupt

from . import prompts
from .data.panel import FEATURE_COLUMNS
from .dsl import DslError, parse
from .eval.combo import Combo, gain
from .eval.evaluate import evaluate
from .eval.gates import apply_fdr, fmt, judge, origin_gate, passed
from .eval.params import EvalParams, Thresholds
from .eval.recheck import recheck
from .store import Store, clean, now_ms

logger = logging.getLogger(__name__)

MAX_PARSE_RETRIES = 2
MAX_REVISES = 2


@dataclass
class Context:
    gen: BaseChatModel
    review: BaseChatModel
    # 全量面板只给 holdout 用；评估和门只看 search（cut 之前）
    panels: dict[str, pd.DataFrame]
    search: dict[str, pd.DataFrame]
    cut_ms: int
    references: dict[str, pd.DataFrame]
    # 库里因子在搜索段上的组合，G9 的基线
    combo: Combo
    known: set[str]
    library: list[dict[str, Any]]
    facts: dict[str, Any]
    store: Store
    params: EvalParams = field(default_factory=EvalParams)
    thresholds: Thresholds = field(default_factory=Thresholds)
    # 同时跑评估的候选数：一次评估要把表达式求值上百遍，扇出全开会把内存吃满
    concurrency: int = 2


def merge_by_id(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """候选按 id 覆盖合并，Send 扇出回来的和 summarize 改过的都走这里"""
    merged = {c["id"]: c for c in left}
    for c in right:
        merged[c["id"]] = c
    return sorted(merged.values(), key=lambda c: c["seq"])


class RunState(TypedDict, total=False):
    run_id: str
    lang: str
    k: int
    note: str
    hypotheses: list[dict[str, Any]]
    candidates: Annotated[list[dict[str, Any]], merge_by_id]
    decisions: dict[str, bool]


def build_graph(checkpointer: BaseCheckpointSaver, ctx: Context):
    evaluating = asyncio.Semaphore(ctx.concurrency)

    async def propose(state: RunState) -> dict[str, Any]:
        await emit(ctx.store, state["run_id"], "status", {"stage": "proposing"})
        tried = await ctx.store.tried()
        system = prompts.system_gen(ctx.facts, state["lang"])
        message = prompts.propose_message(state["k"], ctx.library, tried, state.get("note", ""), state["lang"])
        items = await _ask_json(ctx.gen, system, message, prompts.parse_list)
        hypotheses = []
        for item in items or []:
            if isinstance(item, dict) and item.get("name") and item.get("hypothesis"):
                seq = len(hypotheses) + 1
                hypotheses.append({
                    "id": f"{state['run_id']}-{seq}", "run_id": state["run_id"], "seq": seq,
                    "name": str(item["name"]).strip(), "hypothesis": str(item["hypothesis"]).strip(),
                    "family": str(item.get("family") or "").strip(), "status": "proposed", "attempts": 0,
                })
            if len(hypotheses) >= state["k"]:
                break
        for h in hypotheses:
            await ctx.store.upsert_candidate(h)
            await emit(ctx.store, state["run_id"], "candidate", brief(h))
        await emit(ctx.store, state["run_id"], "log", {"text": f"提出 {len(hypotheses)} 条假设"})
        return {"hypotheses": hypotheses}

    def fan_out(state: RunState):
        if not state.get("hypotheses"):
            return "summarize"
        return [Send("candidate", {"run_id": state["run_id"], "lang": state["lang"], "cand": h})
                for h in state["hypotheses"]]

    async def candidate(payload: dict[str, Any]) -> dict[str, Any]:
        cand = dict(payload["cand"])
        run_id, lang = payload["run_id"], payload["lang"]
        system = prompts.system_gen(ctx.facts, lang)
        feedback: str | None = None
        revises = 0
        while True:
            expr = await _implement(cand, system, feedback)
            if expr is None:
                cand["status"], cand["reason"] = "failed", "连试三次都没写出合法表达式"
                break
            cand["expr"], cand["canonical"], cand["status"] = str(expr), expr.canonical(), "evaluating"
            await _save(ctx.store, run_id, cand)
            async with evaluating:
                ev = await asyncio.to_thread(evaluate, expr, ctx.search, ctx.params, ctx.references, ctx.combo)
            gates = judge(ev, ctx.thresholds, ctx.params.primary, ctx.known)
            cand["eval"], cand["gates"] = clean(ev), clean(gates)
            # 每评估一次就是一次试验，被评审打回重写的版本也算
            cand["evaluations"] = cand.get("evaluations", 0) + 1
            if not passed(gates):
                cand["status"] = "rejected_gates"
                cand["reason"] = "；".join(f"{g['id']} {g['note']}" for g in gates if not g["passed"])
                break
            cand["status"] = "reviewing"
            await _save(ctx.store, run_id, cand)
            review = await _review(cand, lang)
            cand["review"] = review
            verdict = review.get("verdict")
            if verdict == "pass":
                cand["status"] = "passed"
                break
            if verdict == "revise" and revises < MAX_REVISES:
                revises += 1
                feedback = review.get("suggestion") or review.get("comment") or "评审要求修改"
                cand["status"] = "revise"
                await _save(ctx.store, run_id, cand)
                continue
            cand["status"] = "rejected_review"
            cand["reason"] = review.get("comment") or verdict
            break
        await _save(ctx.store, run_id, cand)
        return {"candidates": [cand]}

    async def _implement(cand: dict[str, Any], system: str, feedback: str | None):
        for _ in range(MAX_PARSE_RETRIES + 1):
            cand["attempts"] = cand.get("attempts", 0) + 1
            obj = await _ask_json(ctx.gen, system, prompts.implement_message(cand, feedback), prompts.parse_object)
            text = (obj or {}).get("expr")
            if not text:
                feedback = "上一次没有给出 JSON 里的 expr 字段"
                continue
            try:
                expr = parse(str(text), FEATURE_COLUMNS)
            except DslError as exc:
                feedback = f"表达式不合法：{exc}"
                continue
            cand["note"] = str((obj or {}).get("note") or "")
            return expr
        return None

    async def _review(cand: dict[str, Any], lang: str) -> dict[str, Any]:
        summary = prompts.eval_brief(cand["eval"], cand["gates"], ctx.params.primary)
        obj = await _ask_json(ctx.review, prompts.system_review(lang), prompts.review_message(cand, summary),
                              prompts.parse_object)
        verdict = str((obj or {}).get("verdict") or "").strip().lower()
        if verdict not in ("pass", "revise", "reject"):
            return {"verdict": "reject", "comment": "评审没有给出合法裁决", "suggestion": ""}
        return {"verdict": verdict, "comment": str(obj.get("comment") or ""), "suggestion": str(obj.get("suggestion") or "")}

    async def summarize(state: RunState) -> dict[str, Any]:
        run_id = state["run_id"]
        await emit(ctx.store, run_id, "status", {"stage": "summarizing"})
        candidates = [dict(c) for c in state.get("candidates", [])]
        evaluated = [c for c in candidates if c.get("gates")]
        wrappers = [{"gates": c["gates"]} for c in evaluated]
        # 被评审打回改掉的旧版本也各算一次检验：进 BH 的 m，也进试验数
        superseded = sum(c["evaluations"] - 1 for c in evaluated)
        apply_fdr(wrappers, ctx.thresholds.fdr_q, ctx.params.iterations, superseded)
        for c, w in zip(evaluated, wrappers):
            c["gates"] = clean(w["gates"])
            if c["status"] == "passed" and not w["passed"]:
                c["status"] = "rejected_fdr"
                c["reason"] = "同批 BH-FDR 没扛住"
            await _save(ctx.store, run_id, c)
        total = await ctx.store.bump_trials(len(evaluated) + superseded)
        survivors = [c for c in candidates if c["status"] == "passed"]
        # 同批按增量从大到小排队：规范形和前面的重合就按 G8 拒；结构不同的再对前面已留下的组合算一次增量，写进 G9 备注给人看
        survivors.sort(key=lambda c: -c["eval"]["gain"]["horizons"][str(ctx.params.primary)]["gain"])
        kept, kept_names, kept_canon = ctx.combo, [], {}
        for c in survivors:
            expr = parse(c["expr"], FEATURE_COLUMNS)
            if c["canonical"] in kept_canon:
                c["gates"] = [clean(origin_gate(c["canonical"], kept_canon)) if g["id"] == "G8" else g for g in c["gates"]]
                c["status"] = "rejected_gates"
                c["reason"] = f"G8 规范形与同批 {kept_canon[c['canonical']]} 重合"
                await _save(ctx.store, run_id, c)
                await emit(ctx.store, run_id, "log", {"text": f"{c['name']}：{c['reason'][3:]}，拒掉"})
                continue
            if kept_names:
                again = await asyncio.to_thread(gain, kept, c["name"], expr, ctx.params)
                note = f"；排在同批 {'、'.join(kept_names)} 之后增量 {fmt(again['horizons'][ctx.params.primary]['gain'])}"
                c["gates"] = [{**g, "note": g["note"] + note} if g["id"] == "G9" else g for g in c["gates"]]
            kept = await asyncio.to_thread(kept.plus, c["name"], expr)
            kept_names.append(c["name"])
            kept_canon[c["canonical"]] = c["name"]
            # holdout 只给留下的算，拒掉的碰都不碰
            held = await asyncio.to_thread(recheck, expr, ctx.panels, ctx.cut_ms, ctx.params)
            c["eval"] = {**c["eval"], "holdout": clean(held)}
            await _save(ctx.store, run_id, c)
            main = held["horizons"][ctx.params.primary]
            await emit(ctx.store, run_id, "log", {
                "text": f"{c['name']}：holdout 主档 IC {fmt(main['mean'])}，{main['sameSignSymbols']} 个币同号，{main['samples']} 个点"})
        await emit(ctx.store, run_id, "log", {
            "text": f"本轮评估 {len(evaluated)} 条，过门 {len(kept_names)} 条；累计试验 {total} 次，"
                    f"按 0.05 门槛累计预期假阳 {total * 0.05:.1f} 条"})
        return {"candidates": candidates}

    def need_decisions(state: RunState) -> str:
        return "await_decisions" if any(c["status"] == "passed" for c in state.get("candidates", [])) else "commit"

    async def await_decisions(state: RunState) -> dict[str, Any]:
        survivors = [brief(c) for c in state["candidates"] if c["status"] == "passed"]
        value = interrupt({"candidates": survivors})
        return {"decisions": dict((value or {}).get("decisions") or {})}

    async def commit(state: RunState) -> dict[str, Any]:
        run_id = state["run_id"]
        decisions = state.get("decisions") or {}
        candidates = [dict(c) for c in state.get("candidates", [])]
        approved = 0
        for c in candidates:
            if c["status"] == "passed":
                if decisions.get(c["id"]):
                    c["status"], c["decision"] = "approved", "approve"
                    await ctx.store.add_factor({
                        "id": c["id"], "name": c["name"], "hypothesis": c["hypothesis"], "expr": c["expr"],
                        "canonical": c["canonical"], "columns": c["eval"]["columns"], "eval": c["eval"],
                        "gates": c["gates"], "review": c.get("review"), "approved_at": now_ms(), "run_id": run_id,
                    })
                    approved += 1
                else:
                    c["status"], c["decision"] = "declined", "decline"
            await ctx.store.add_tried(run_id, c["hypothesis"], c.get("expr"), c["status"], c.get("reason"))
            await _save(ctx.store, run_id, c)
        await ctx.store.set_run_status(run_id, "done")
        await emit(ctx.store, run_id, "done", {"approved": approved})
        return {"candidates": candidates}

    g = StateGraph(RunState)
    g.add_node("propose", propose)
    g.add_node("candidate", candidate)
    g.add_node("summarize", summarize)
    g.add_node("await_decisions", await_decisions)
    g.add_node("commit", commit)
    g.add_edge(START, "propose")
    g.add_conditional_edges("propose", fan_out, ["candidate", "summarize"])
    g.add_edge("candidate", "summarize")
    g.add_conditional_edges("summarize", need_decisions, ["await_decisions", "commit"])
    g.add_edge("await_decisions", "commit")
    g.add_edge("commit", END)
    return g.compile(checkpointer=checkpointer)


# ========== 辅助 ==========


def brief(c: dict[str, Any]) -> dict[str, Any]:
    """页面和审批用的候选摘要：不带整份 eval"""
    out = {k: c.get(k) for k in ("id", "seq", "name", "hypothesis", "family", "expr", "note", "status",
                                  "attempts", "reason", "review", "decision")}
    if c.get("gates"):
        out["gates"] = [{"id": g["id"], "name": g.get("name"), "passed": g["passed"], "note": g["note"]}
                        for g in c["gates"]]
    if c.get("eval"):
        out["brief"] = prompts.eval_brief(c["eval"], c["gates"], _primary(c["eval"]))
    return out


def _primary(ev: dict[str, Any]) -> int:
    keys = [int(k) for k in ev["horizons"]]
    return 48 if 48 in keys else keys[0]


async def emit(store: Store, run_id: str, event: str, data: Any) -> None:
    """先落库拿到 id 再推流，订阅端按 id 去重；不在流式上下文里时只落库"""
    data = clean(data)
    event_id = await store.add_event(run_id, event, data)
    try:
        get_stream_writer()({"id": event_id, "event": event, "data": data})
    except RuntimeError:
        pass


async def _save(store: Store, run_id: str, cand: dict[str, Any]) -> None:
    await store.upsert_candidate(cand)
    await emit(store, run_id, "candidate", brief(cand))


async def _ask_json(model: BaseChatModel, system: str, message: str, parser):
    """问一次；不是合法 JSON 就带着原话再要一次"""
    messages = [SystemMessage(system), HumanMessage(message)]
    reply = await model.ainvoke(messages)
    obj = parser(reply.text)
    if obj is not None:
        return obj
    retry = messages + [reply, HumanMessage(prompts.json_retry())]
    return parser((await model.ainvoke(retry)).text)
