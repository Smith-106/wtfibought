"""脚本化的假模型：按收到的整段提示词挑回答，图和接口测试共用。"""

import json
from typing import Any

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedModel(GenericFakeChatModel):
    reply: Any = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        text = self.reply("\n".join(m.text for m in messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(text))])


def model(reply) -> ScriptedModel:
    return ScriptedModel(messages=iter(()), reply=reply)


def gen_reply(prompt: str) -> str:
    """planted：先少写窗口、再写 $volume 被评审打回、最后 Mean($volume, 4)；noise：$trades；
    twin：planted 换个结构（EMA），clone：planted 换个窗口（规范形相同）；k ≥ 3 才轮到后两条"""
    if "JSON 数组" in prompt:
        return "好的：\n```json\n" + json.dumps([
            {"name": "planted", "family": "量能", "hypothesis": "成交量列藏着信号"},
            {"name": "noise", "family": "量能", "hypothesis": "成交笔数没用"},
            {"name": "twin", "family": "量能", "hypothesis": "成交量指数平滑"},
            {"name": "clone", "family": "量能", "hypothesis": "成交量再平滑一点"},
        ], ensure_ascii=False) + "\n```"
    if "写成一条表达式" in prompt:
        if "[noise]" in prompt:
            return '{"expr": "$trades", "note": "直接用"}'
        if "[twin]" in prompt:
            return '{"expr": "EMA($volume, 8)", "note": "指数平滑"}'
        if "[clone]" in prompt:
            return '{"expr": "Mean($volume, 16)", "note": "换个窗口"}'
        if "用 4 根均值" in prompt:
            return '{"expr": "Mean($volume, 4)", "note": "平滑后的成交量"}'
        if "表达式不合法" in prompt:
            return '{"expr": "$volume", "note": "直接用成交量"}'
        return '{"expr": "Mean($volume)", "note": "少写了窗口"}'
    return "？"


def review_reply(prompt: str) -> str:
    if "表达式：Mean($volume," in prompt or "表达式：EMA($volume," in prompt:
        return '{"verdict": "pass", "comment": "对得上", "suggestion": ""}'
    if "表达式：$volume" in prompt:
        return '{"verdict": "revise", "comment": "太毛", "suggestion": "用 4 根均值"}'
    return '{"verdict": "reject", "comment": "不行", "suggestion": ""}'
