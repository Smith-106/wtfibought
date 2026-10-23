"""按角色的模型配置建 chat model。

baseUrl 约定与 Java 侧一致：用户填的不带版本后缀，交给 SDK 的补 /v1，自己拼路径的把后缀剥掉。
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from .config import ModelSpec

logger = logging.getLogger(__name__)

_ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_GEMINI_EFFORTS = {"minimal", "low", "medium", "high"}
# 一条表达式加一段评审用不了多少字，思考 token 也算在里面
_ANTHROPIC_MAX_TOKENS = 16000


class LlmError(RuntimeError):
    """角色没配好或协议不认"""


def build_chat_model(spec: ModelSpec, role: str) -> BaseChatModel:
    if not spec.model.strip() or not spec.api_key.strip():
        raise LlmError(f"{role} 角色的模型名或 API key 是空的")
    protocol = spec.protocol.strip().lower() or "openai"
    base_url = spec.base_url.strip()
    effort = _effort(spec.effort)

    if protocol in ("openai", "responses"):
        kwargs: dict[str, Any] = {}
        if effort:
            kwargs["reasoning_effort"] = effort
        if protocol == "responses":
            kwargs["use_responses_api"] = True
        return ChatOpenAI(model=spec.model, base_url=_for_sdk(base_url) or None, api_key=spec.api_key, **kwargs)

    if protocol == "anthropic":
        return ChatAnthropic(
            model=spec.model,
            base_url=_strip(base_url, "/v1") or None,
            api_key=spec.api_key,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
            **_effort_kwargs(effort, _ANTHROPIC_EFFORTS, protocol),
        )

    if protocol == "gemini":
        return ChatGoogleGenerativeAI(
            model=spec.model,
            base_url=_strip(base_url, "/v1beta") or None,
            api_key=spec.api_key,
            **_effort_kwargs(effort, _GEMINI_EFFORTS, protocol),
        )

    raise LlmError(f"{role} 角色的协议 {protocol} 不支持")


def _effort(value: str) -> str | None:
    """none / 空 = 不传，走模型默认"""
    effort = (value or "").strip().lower()
    return None if effort in ("", "none") else effort


def _effort_kwargs(effort: str | None, allowed: set[str], protocol: str) -> dict[str, str]:
    if effort is None:
        return {}
    if effort not in allowed:
        logger.info("思考档位 %s 不在 %s 协议的取值表里，忽略", effort, protocol)
        return {}
    return {"reasoning_effort": effort}


def _for_sdk(base_url: str) -> str:
    """交给 OpenAI SDK 的补 /v1；手滑带了后缀不能拼成 /v1/v1"""
    url = base_url.rstrip("/")
    return url if not url or url.endswith("/v1") else url + "/v1"


def _strip(base_url: str, suffix: str) -> str:
    """自己拼路径的协议把版本后缀剥掉"""
    url = base_url.rstrip("/")
    if url.endswith(suffix):
        url = url[: -len(suffix)].rstrip("/")
    return url
