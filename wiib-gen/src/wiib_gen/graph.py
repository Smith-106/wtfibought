"""指令生成图：问 → 生成一版 → 再问，每步一个 checkpoint，任意版可回退。

平台事实由 main.py 在请求里拉好当初始输入进来，图里不碰 satoken。
state 只存 endpoint_id 与 user_id，模型每次现建，解密出来的 key 不进 checkpoint。
interrupt 所在的节点（ask_*）一律没有副作用：恢复时节点从头重跑，副作用会重复。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import prompts
from .llm import build_chat_model, load_endpoint

ModelFactory = Callable[[int, int], Awaitable[BaseChatModel]]


class GenState(TypedDict, total=False):
    facts: dict[str, Any]
    answers: list[dict[str, str]]
    # 问过的题 key，答了跳了都算；出下一道题时照它避重
    asked: list[str]
    question: dict[str, Any] | None
    versions: list[dict[str, Any]]
    done: bool


async def default_model_factory(endpoint_id: int, user_id: int) -> BaseChatModel:
    """每次现取端点行现建模型；读 PG 是同步的，挪到线程里"""
    row = await asyncio.to_thread(load_endpoint, endpoint_id, user_id)
    return build_chat_model(row)


def build_graph(checkpointer: BaseCheckpointSaver,
                model_factory: ModelFactory = default_model_factory):
    async def generate(state: GenState) -> dict[str, Any]:
        return await _generate(state, model_factory)

    async def propose_next(state: GenState) -> dict[str, Any]:
        return await _propose_next(state, model_factory)

    g = StateGraph(GenState)
    g.add_node("ask_mode", ask_mode)
    g.add_node("ask_base", ask_base)
    g.add_node("generate", generate)
    g.add_node("propose_next", propose_next)
    g.add_node("ask_next", ask_next)
    g.add_conditional_edges(START, _entry, ["ask_mode", "ask_base"])
    g.add_edge("ask_mode", "ask_base")
    g.add_edge("ask_base", "generate")
    g.add_edge("generate", "propose_next")
    g.add_edge("propose_next", "ask_next")
    return g.compile(checkpointer=checkpointer)


def _entry(state: GenState) -> str:
    """有现成指令才值得问增量还是从零"""
    facts = state["facts"]
    return "ask_mode" if (facts.get("existing_prompt") or "").strip() else "ask_base"


# ========== 问答节点：只 interrupt，不调模型 ==========


async def ask_mode(state: GenState) -> dict[str, Any]:
    facts = state["facts"]
    lang = facts["dialog_lang"]
    question = prompts.mode_question(lang)
    picked = _picked(question, interrupt(question))
    chosen = picked[0]["answer"] if picked else ""
    mode = "incremental" if prompts.is_incremental(chosen, lang) else "scratch"
    return {"facts": {**facts, "mode": mode}, "answers": state["answers"] + picked,
            "asked": _asked(state, question)}


async def ask_base(state: GenState) -> dict[str, Any]:
    question = prompts.base_question(state["facts"]["dialog_lang"])
    return {"answers": state["answers"] + _picked(question, interrupt(question)),
            "asked": _asked(state, question)}


async def ask_next(state: GenState) -> Command:
    question = state["question"]
    value = interrupt(question)
    asked = _asked(state, question)
    if value.get("finish"):
        return Command(goto=END, update={"done": True, "asked": asked})
    if value.get("skip"):
        return Command(goto="propose_next", update={"asked": asked})
    return Command(goto="generate",
                   update={"asked": asked,
                           "answers": state["answers"] + _picked(question, value)})


def _asked(state: GenState, question: dict[str, Any]) -> list[str]:
    """这一屏的 key 记进问过的名单，去重保序"""
    keys = state["asked"] + [item["key"] for item in question["items"]]
    return list(dict.fromkeys(keys))


def _picked(question: dict[str, Any], value: dict[str, Any] | None) -> list[dict[str, str]]:
    """一屏的回答拆成按题记录，空答的丢掉"""
    got = (value or {}).get("answer") or {}
    records = []
    for item in question["items"]:
        answer = str(got.get(item["key"]) or "").strip()
        if answer:
            records.append({"key": item["key"], "question": item["text"], "answer": answer})
    return records


# ========== 生成一版 ==========


async def _generate(state: GenState, factory: ModelFactory) -> dict[str, Any]:
    facts = state["facts"]
    model = await factory(facts["endpoint_id"], facts["user_id"])
    versions = state["versions"]
    last_text = versions[-1]["text"] if versions else ""
    messages: list[BaseMessage] = [
        SystemMessage(prompts.system_prompt(facts)),
        HumanMessage(prompts.user_message(state["answers"], last_text, _fresh(state))),
    ]
    raw = await _stream_text(model, messages)
    text, notes = _split(raw)
    if prompts.ulen(text) > prompts.MAX_CHARS:
        # 只重写一次，再长就截断；先招呼一声，让页面把首稿的字清掉重接
        get_stream_writer()({"restart": True})
        retry = messages + [AIMessage(raw), HumanMessage(prompts.shrink_message())]
        text, notes = _split(await _stream_text(model, retry))
    if prompts.ulen(text) > prompts.MAX_CHARS:
        text = prompts.utrunc(text, prompts.MAX_CHARS)
        notes = f"{notes}\n{prompts.truncated_note(facts['dialog_lang'])}".strip()
    version = {"no": len(versions) + 1, "text": text, "notes": notes}
    return {"versions": versions + [version]}


def _fresh(state: GenState) -> list[dict[str, str]]:
    """本轮新答：上一道题问的那几个 key；第一版之前没问过题，全部都算新的"""
    question = state.get("question")
    if not question:
        return state["answers"]
    keys = {item["key"] for item in question["items"]}
    return [a for a in state["answers"] if a["key"] in keys]


async def _stream_text(model: BaseChatModel, messages: list[BaseMessage]) -> str:
    chunks = []
    async for chunk in model.astream(messages):
        chunks.append(chunk.text)
    return "".join(chunks)


_SPLIT = re.compile(rf"^[ \t]*{re.escape(prompts.NOTES_MARK)}[ \t]*$", re.M)


def _split(raw: str) -> tuple[str, str]:
    """正文 + 单独一行哨兵 + notes；没哨兵就全算正文"""
    parts = _SPLIT.split(raw.replace("\r\n", "\n").strip(), maxsplit=1)
    return parts[0].strip(), (parts[1].strip() if len(parts) > 1 else "")


# ========== 出下一道题：不 interrupt ==========


async def _propose_next(state: GenState, factory: ModelFactory) -> dict[str, Any]:
    facts = state["facts"]
    lang = facts["dialog_lang"]
    asked = state["asked"]
    if "play" not in asked:
        return {"question": prompts.play_question(lang)}

    model = await factory(facts["endpoint_id"], facts["user_id"])
    versions = state["versions"]
    messages: list[BaseMessage] = [
        SystemMessage(prompts.DETAIL_SYSTEM),
        HumanMessage(prompts.detail_message(facts, state["answers"],
                                            versions[-1]["text"] if versions else "", asked)),
    ]
    raw = (await model.ainvoke(messages)).text
    item = _parse_item(raw)
    if item is None:
        retry = messages + [AIMessage(raw), HumanMessage(prompts.json_retry())]
        item = _parse_item((await model.ainvoke(retry)).text)
    if item is None:
        return {"question": prompts.free_question(lang)}
    return {"question": prompts.detail_screen(item)}


def _parse_item(raw: str) -> dict[str, Any] | None:
    """模型爱裹代码块，取里头第一段 {…}"""
    found = re.search(r"\{.*\}", raw, re.S)
    if not found:
        return None
    try:
        obj = json.loads(found.group())
    except json.JSONDecodeError:
        return None
    key = str(obj.get("key") or "").strip()
    text = str(obj.get("text") or "").strip()
    if not key or not text:
        return None
    options = [str(o).strip() for o in (obj.get("options") or []) if str(o).strip()]
    return {"key": key, "text": text, "options": options, "allow_free": True}
