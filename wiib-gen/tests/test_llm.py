import base64

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from wiib_gen.config import config
from wiib_gen.crypto import encrypt
from wiib_gen.llm import LlmError, build_chat_model

SECRET = base64.b64encode(bytes(range(32))).decode()


@pytest.fixture(autouse=True)
def _secret(monkeypatch):
    monkeypatch.setenv("WIIB_TRADER_KEY_SECRET", SECRET)
    config.cache_clear()
    yield
    config.cache_clear()


def row(protocol, base_url="https://up.test", model="m-1", effort=None):
    return {
        "api_protocol": protocol,
        "base_url": base_url,
        "model": model,
        "reasoning_effort": effort,
        "api_key_enc": encrypt("sk-plain", SECRET),
    }


def test_openai():
    m = build_chat_model(row("openai", effort="high"))
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "m-1"
    assert m.openai_api_base == "https://up.test/v1"
    assert m.openai_api_key.get_secret_value() == "sk-plain"
    assert m.reasoning_effort == "high"
    assert not m.use_responses_api


def test_openai_base_url_already_has_v1():
    assert build_chat_model(row("openai", base_url="https://up.test/v1/")).openai_api_base == "https://up.test/v1"


def test_blank_protocol_falls_back_to_openai():
    assert isinstance(build_chat_model(row("  ")), ChatOpenAI)


def test_protocol_case_and_space_ignored():
    assert build_chat_model(row(" Responses ")).use_responses_api is True


def test_responses():
    m = build_chat_model(row("responses", effort="medium"))
    assert isinstance(m, ChatOpenAI)
    assert m.use_responses_api is True
    assert m.openai_api_base == "https://up.test/v1"
    assert m.reasoning_effort == "medium"


def test_anthropic_strips_v1():
    m = build_chat_model(row("anthropic", base_url="https://claude.test/v1", effort="high"))
    assert isinstance(m, ChatAnthropic)
    assert m.anthropic_api_url == "https://claude.test"
    assert m.model == "m-1"
    assert m.anthropic_api_key.get_secret_value() == "sk-plain"
    assert m.reasoning_effort == "high"
    assert m.max_tokens == 64000


def test_gemini_strips_v1beta():
    m = build_chat_model(row("gemini", base_url="https://gem.test/v1beta", effort="low"))
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert m.base_url == "https://gem.test"
    assert m.model == "m-1"
    assert m.google_api_key.get_secret_value() == "sk-plain"
    assert m.reasoning_effort == "low"


@pytest.mark.parametrize("effort", [None, "", "none", "NONE", " none "])
def test_effort_none_not_passed(effort):
    assert build_chat_model(row("openai", effort=effort)).reasoning_effort is None
    assert build_chat_model(row("anthropic", effort=effort)).reasoning_effort is None
    assert build_chat_model(row("gemini", effort=effort)).reasoning_effort is None


def test_effort_outside_package_table_ignored():
    # xhigh anthropic 认、gemini 不认
    assert build_chat_model(row("anthropic", effort="xhigh")).reasoning_effort == "xhigh"
    assert build_chat_model(row("gemini", effort="xhigh")).reasoning_effort is None


def test_unknown_protocol_raises():
    with pytest.raises(LlmError, match="暂不支持"):
        build_chat_model(row("azure"))
