from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from wiib_gen import prompts
from wiib_gen.graph import _split, build_graph

MARK = prompts.NOTES_MARK

SPEC = {"leverageMin": 3, "leverageMax": 20, "marginPctMin": 5, "marginPctMax": 20,
        "allowMultiPosition": True, "allowHedge": False}

DETAIL_JSON = '{"key": "exit_rule", "text": "怎么退出？", "options": ["固定止盈", "跟踪止损", "时间止损"]}'


class ScriptedModel(GenericFakeChatModel):
    """按收到的整段提示词挑回答；_stream 由父类按词切块，图里流式拿得到增量"""

    reply: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.reply("\n".join(m.text for m in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])


def scripted(reply) -> Any:
    async def factory(endpoint_id: int, user_id: int):
        assert (endpoint_id, user_id) == (1, 7)
        return ScriptedModel(messages=iter(()), reply=reply)

    return factory


def facts(**over) -> dict[str, Any]:
    base = {
        "mode": "scratch", "output_lang": "zh", "dialog_lang": "zh",
        "endpoint_id": 1, "user_id": 7, "has_trader": True, "spec": SPEC,
        "symbols": "BTCUSDT", "interval_code": "15m", "wake_window": None,
        "template": "平台系统提示词全文", "default_instructions": "【方法】默认底稿",
        "existing_prompt": "老指令正文", "risk_table": prompts.risk_table(SPEC, "zh"),
    }
    return {**base, **over}


def initial(**over) -> dict[str, Any]:
    return {"facts": facts(**over), "answers": [], "asked": [], "question": None, "fresh": [],
            "versions": [], "done": False}


CFG = {"configurable": {"thread_id": "t"}}


@pytest.fixture
async def saver(tmp_path):
    async with AsyncSqliteSaver.from_conn_string(str(tmp_path / "gen.sqlite")) as s:
        yield s


async def step(app, payload):
    await app.ainvoke(payload, CFG, durability="sync")
    return await app.aget_state(CFG)


def asked(snapshot) -> dict[str, Any]:
    """当前停在哪道题上"""
    return snapshot.interrupts[0].value


def versions(snapshot) -> list[dict[str, Any]]:
    return snapshot.values["versions"]


# ========== 主线：mode → base → v1 → play → v2 → 回退 → v2' ==========


def transcript():
    """记下每次生成看到的提示词，顺带让每版正文各不相同"""
    log: list[str] = []

    def reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            return f"```json\n{DETAIL_JSON}\n```"
        log.append(text)
        # 只认答案行：标签表里常驻「BREAKOUT 突破」
        play = "突破" if "→ 突破" in text else "趋势跟随"
        return f"【方法】第{len(log)}版 {play}\n{MARK}\n本版改了什么"

    return log, reply


async def test_full_line_with_rollback(saver):
    log, reply = transcript()
    app = build_graph(saver, scripted(reply))

    snap = await step(app, initial())
    assert asked(snap)["kind"] == "mode"
    assert asked(snap)["items"][0]["options"] == ["在现有指令上改", "从零写一份"]

    snap = await step(app, Command(resume={"answer": {"mode": "从零写一份"}}))
    assert asked(snap)["kind"] == "base"
    assert [i["key"] for i in asked(snap)["items"]] == ["direction", "holding", "style"]
    assert snap.values["facts"]["mode"] == "scratch"

    snap = await step(app, Command(resume={"answer": {
        "direction": "只做多", "holding": "波段 1~3 天", "style": "均衡"}}))
    assert [v["no"] for v in versions(snap)] == [1]
    assert versions(snap)[0]["text"] == "【方法】第1版 趋势跟随"
    assert versions(snap)[0]["notes"] == "本版改了什么"
    assert asked(snap)["kind"] == "play"
    assert asked(snap)["allow_skip"] and asked(snap)["allow_finish"]

    snap = await step(app, Command(resume={"answer": {"play": "趋势跟随"}}))
    assert [v["no"] for v in versions(snap)] == [1, 2]
    assert asked(snap)["kind"] == "detail"
    assert asked(snap)["items"][0]["key"] == "exit_rule"

    # 回退到第 1 版：找最新的「刚生成完第 1 版、停在 ask_next」的 checkpoint 分叉
    history = [h async for h in app.aget_state_history(CFG)]
    target = next(h for h in history
                  if h.next == ("ask_next",) and len(h.values.get("versions") or []) == 1)
    await app.aupdate_state(target.config, values={})
    snap = await app.aget_state(CFG)
    assert [v["no"] for v in versions(snap)] == [1]
    assert snap.next == ("ask_next",)
    assert snap.values["question"]["kind"] == "play"

    snap = await step(app, Command(resume={"answer": {"play": "突破"}}))
    assert [v["no"] for v in versions(snap)] == [1, 2]
    assert versions(snap)[1]["text"] == "【方法】第3版 突破"
    assert "→ 突破" in log[-1] and "→ 只做多" in log[-1]

    # 两条分支都还在 history 里：第 2 版有两份，正文不一样
    history = [h async for h in app.aget_state_history(CFG)]
    seconds = {tuple(v["text"] for v in h.values["versions"])
               for h in history if len(h.values.get("versions") or []) == 2}
    assert seconds == {("【方法】第1版 趋势跟随", "【方法】第2版 趋势跟随"),
                       ("【方法】第1版 趋势跟随", "【方法】第3版 突破")}


# ========== 跳过与完成 ==========


async def test_skip_asks_another_question(saver):
    _, reply = transcript()
    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    snap = await step(app, Command(resume={"answer": {
        "direction": "多空都做", "holding": "更长", "style": "保守"}}))
    assert asked(snap)["kind"] == "play"

    snap = await step(app, Command(resume={"skip": True}))
    assert asked(snap)["kind"] == "detail"
    assert len(versions(snap)) == 1


async def test_skipped_question_is_not_asked_again(saver):
    """跳过的 play 不能再冒出来，且出题时要告诉模型它已经问过"""
    seen = []

    def reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            seen.append(text)
            return DETAIL_JSON
        return f"【方法】正文\n{MARK}\nnotes"

    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    snap = await step(app, Command(resume={"answer": {
        "direction": "只做多", "holding": "更长", "style": "保守"}}))
    assert asked(snap)["kind"] == "play"

    snap = await step(app, Command(resume={"skip": True}))
    assert asked(snap)["items"][0]["key"] == "exit_rule"
    assert snap.values["asked"] == ["direction", "holding", "style", "play"]

    snap = await step(app, Command(resume={"answer": {"exit_rule": "固定止盈"}}))
    assert asked(snap)["kind"] == "detail"
    assert asked(snap)["items"][0]["key"] != "play"
    # 喂给模型的「已问过的 key」里带着被跳过的 play
    line = next(l for l in seen[-1].splitlines() if l.startswith("已问过的 key"))
    assert "play" in line and "exit_rule" in line


async def test_finish_ends_graph(saver):
    _, reply = transcript()
    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    await step(app, Command(resume={"answer": {
        "direction": "只做空", "holding": "日内几根 K 线", "style": "激进"}}))

    snap = await step(app, Command(resume={"finish": True}))
    assert snap.values["done"] is True
    assert snap.next == ()
    assert len(versions(snap)) == 1


# ========== 没 trader 直接从基础题开始 ==========


async def test_no_trader_skips_mode_question(saver):
    _, reply = transcript()
    app = build_graph(saver, scripted(reply))
    snap = await step(app, initial(has_trader=False, existing_prompt=""))
    assert asked(snap)["kind"] == "base"


# ========== 超长正文 ==========


async def test_rewrite_brings_body_under_limit(saver):
    def reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            return DETAIL_JSON
        if "压到" in text:
            return f"短正文\n{MARK}\n压过了"
        return "长" * 4500 + f"\n{MARK}\n第一版"

    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    snap = await step(app, Command(resume={"answer": {
        "direction": "只做多", "holding": "更长", "style": "均衡"}}))
    assert versions(snap)[0]["text"] == "短正文"
    assert versions(snap)[0]["notes"] == "压过了"


async def test_still_long_after_rewrite_is_truncated(saver):
    def reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            return DETAIL_JSON
        return "长" * 4500 + f"\n{MARK}\n没压住"

    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    snap = await step(app, Command(resume={"answer": {
        "direction": "只做多", "holding": "更长", "style": "均衡"}}))
    version = versions(snap)[0]
    assert len(version["text"]) == prompts.MAX_CHARS
    assert version["notes"].startswith("没压住")
    assert version["notes"].endswith(prompts.truncated_note("zh"))


async def test_emoji_body_counts_as_utf16_units(saver):
    """4000 个 emoji = 8000 个 UTF-16 单元，按 Java 口径算超长；截完不留半个代理对"""
    def reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            return DETAIL_JSON
        return "🚀" * 4000 + f"\n{MARK}\n没压住"

    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    snap = await step(app, Command(resume={"answer": {
        "direction": "只做多", "holding": "更长", "style": "均衡"}}))
    text = versions(snap)[0]["text"]
    assert prompts.ulen(text) == prompts.MAX_CHARS
    assert len(text) == prompts.MAX_CHARS // 2
    assert text.encode("utf-16-le").decode("utf-16-le") == text


# ========== 出题兜底 ==========


async def test_bad_json_falls_back_to_free_question(saver):
    calls = []

    def reply(text: str) -> str:
        if "只输出一个 JSON 对象" in text:
            calls.append(text)
            return "我觉得该问退出方式"
        return f"【方法】正文\n{MARK}\nnotes"

    app = build_graph(saver, scripted(reply))
    await step(app, initial(existing_prompt=""))
    await step(app, Command(resume={"answer": {
        "direction": "只做多", "holding": "更长", "style": "均衡"}}))
    snap = await step(app, Command(resume={"answer": {"play": "区间震荡"}}))
    assert len(calls) == 2  # 解析失败重试一次
    assert asked(snap)["items"][0]["key"] == "free"
    assert asked(snap)["items"][0]["allow_free"] is True


# ========== 正文 / notes 分界 ==========


def test_split_keeps_markdown_rule_in_body():
    raw = f"【方法】上\n\n---\n\n【仓位】下\n{MARK}\nnotes"
    body, notes = _split(raw)
    assert body == "【方法】上\n\n---\n\n【仓位】下"
    assert notes == "notes"


def test_split_without_mark_is_all_body():
    body, notes = _split("【方法】正文\n---\n还是正文")
    assert body == "【方法】正文\n---\n还是正文"
    assert notes == ""


def test_split_takes_first_mark_only():
    body, notes = _split(f"正文\n{MARK}\nnotes 上\n{MARK}\nnotes 下")
    assert body == "正文"
    assert notes == f"notes 上\n{MARK}\nnotes 下"


def test_split_handles_crlf_and_padding():
    assert _split(f"BODY\r\n{MARK}\r\nNOTES") == ("BODY", "NOTES")
    assert _split(f"BODY\n  {MARK}  \nNOTES") == ("BODY", "NOTES")


# ========== UTF-16 口径 ==========


def test_ulen_counts_utf16_units():
    assert prompts.ulen("abc") == 3
    assert prompts.ulen("中文") == 2
    assert prompts.ulen("🚀") == 2


def test_utrunc_never_splits_a_surrogate_pair():
    assert prompts.utrunc("🚀🚀🚀", 5) == "🚀🚀"
    assert prompts.utrunc("a🚀b", 2) == "a"
    assert prompts.utrunc("中文", 9) == "中文"


# ========== 风险表 ==========


def test_risk_table_uses_spec_range():
    rows = prompts.risk_table(SPEC, "zh")
    assert [r["style"] for r in rows] == ["保守", "均衡", "激进"]
    # 均衡 1%：20 倍 × 20% → 0.25%，3 倍 × 5% → 6.67%
    assert rows[1] == {"style": "均衡", "risk_pct": 1.0, "stop_min": 0.25, "stop_max": 6.67}
    # 保守 0.5%：20 倍 × 20% → 0.125%，四舍五入进到 0.13（不是银行家的 0.12）
    assert rows[0]["stop_min"] == 0.13


def test_noise_floor_scales_with_interval():
    assert [prompts.noise_stop_pct(i) for i in ("5m", "15m", "1h", "4h")] == [0.17, 0.3, 0.6, 1.2]


def test_english_bank_and_lang_pick():
    assert prompts.pick_lang("en-US") == "en"
    assert prompts.pick_lang("zh-CN") == "zh"
    assert prompts.pick_lang(None) == "zh"
    assert prompts.play_question("en")["items"][0]["options"][0] == "Breakout"
    assert prompts.is_incremental("Edit the existing one", "en") is True
