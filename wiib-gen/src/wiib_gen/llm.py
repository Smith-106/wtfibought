"""按用户的站内端点建 chat model：读 PG 端点行 → 解密 key → 按协议分叉。

baseUrl 的约定跟 Java 侧一致：用户填的不带版本后缀，交给 SDK 的补 /v1，自己拼路径的把后缀剥掉。
key 现解现用，不落字段、不进日志。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import psycopg
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from psycopg.rows import dict_row

from .config import config
from .crypto import decrypt

logger = logging.getLogger(__name__)

_COLUMNS = "api_protocol, base_url, model, reasoning_effort, api_key_enc"

# 各包认的思考档位，照安装版本的字段签名抄；不在表里的忽略。openai 那边是自由字符串，不设表
_ANTHROPIC_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
_GEMINI_EFFORTS = {"minimal", "low", "medium", "high"}

# Anthropic 的 max_tokens 必填且思考 token 也算在内，包默认的 4096 不够写一版指令；与 Java 侧同值
_ANTHROPIC_MAX_TOKENS = 64000


class LlmError(RuntimeError):
    """端点取不到或协议不认"""


def load_endpoint(endpoint_id: int, user_id: int) -> dict[str, Any]:
    """按 id + user_id 取端点行，两个都得对上；一次主键查，同步够用"""
    with psycopg.connect(**config().pg_conninfo) as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT {_COLUMNS} FROM user_llm_endpoint WHERE id = %s AND user_id = %s",
                (endpoint_id, user_id),
            )
            row = cur.fetchone()
    if row is None:
        raise LlmError(f"端点 {endpoint_id} 不存在或不属于当前用户")
    return row


def build_chat_model(endpoint: Mapping[str, Any]) -> BaseChatModel:
    protocol = (endpoint.get("api_protocol") or "").strip().lower() or "openai"
    base_url = (endpoint.get("base_url") or "").strip()
    model = endpoint["model"]
    api_key = decrypt(endpoint["api_key_enc"])
    effort = _effort(endpoint.get("reasoning_effort"))

    if protocol in ("openai", "responses"):
        kwargs: dict[str, Any] = {}
        if effort:
            kwargs["reasoning_effort"] = effort
        if protocol == "responses":
            kwargs["use_responses_api"] = True
        return ChatOpenAI(model=model, base_url=_for_sdk(base_url), api_key=api_key, **kwargs)

    if protocol == "anthropic":
        return ChatAnthropic(
            model=model,
            base_url=_strip(base_url, "/v1"),
            api_key=api_key,
            max_tokens=_ANTHROPIC_MAX_TOKENS,
            **_effort_kwargs(effort, _ANTHROPIC_EFFORTS, protocol),
        )

    if protocol == "gemini":
        return ChatGoogleGenerativeAI(
            model=model,
            base_url=_strip(base_url, "/v1beta"),
            api_key=api_key,
            **_effort_kwargs(effort, _GEMINI_EFFORTS, protocol),
        )

    raise LlmError(f"端点协议 {protocol} 本工具暂不支持")


def _effort(value: Any) -> str | None:
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
    """交给 OpenAI SDK 的补 /v1；用户手滑带了后缀不能拼成 /v1/v1"""
    url = base_url.rstrip("/")
    return url if not url or url.endswith("/v1") else url + "/v1"


def _strip(base_url: str, suffix: str) -> str:
    """自己拼路径的协议把版本后缀剥掉"""
    url = base_url.rstrip("/")
    if url.endswith(suffix):
        url = url[: -len(suffix)].rstrip("/")
    return url
