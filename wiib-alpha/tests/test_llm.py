import pytest
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from wiib_alpha.config import ModelSpec
from wiib_alpha.llm import LlmError, build_chat_model


def spec(protocol, base_url="https://up.test", model="m-1", effort="", api_key="sk-plain") -> ModelSpec:
    return ModelSpec(protocol=protocol, base_url=base_url, model=model, api_key=api_key, effort=effort)


def test_openai():
    m = build_chat_model(spec("openai", effort="high"), "gen")
    assert isinstance(m, ChatOpenAI)
    assert m.model_name == "m-1" and m.openai_api_base == "https://up.test/v1"
    assert m.openai_api_key.get_secret_value() == "sk-plain"
    assert m.reasoning_effort == "high" and not m.use_responses_api


def test_openai_base_url_variants():
    assert build_chat_model(spec("openai", base_url="https://up.test/v1/"), "gen").openai_api_base == "https://up.test/v1"
    assert build_chat_model(spec("openai", base_url=""), "gen").openai_api_base is None
    assert isinstance(build_chat_model(spec("  "), "gen"), ChatOpenAI)
    assert build_chat_model(spec(" Responses "), "gen").use_responses_api is True


def test_anthropic_strips_v1():
    m = build_chat_model(spec("anthropic", base_url="https://claude.test/v1", effort="high"), "review")
    assert isinstance(m, ChatAnthropic)
    assert m.anthropic_api_url == "https://claude.test" and m.model == "m-1"
    assert m.anthropic_api_key.get_secret_value() == "sk-plain"
    assert m.reasoning_effort == "high" and m.max_tokens == 16000


def test_gemini_strips_v1beta():
    m = build_chat_model(spec("gemini", base_url="https://gem.test/v1beta", effort="low"), "review")
    assert isinstance(m, ChatGoogleGenerativeAI)
    assert m.base_url == "https://gem.test" and m.model == "m-1"
    assert m.google_api_key.get_secret_value() == "sk-plain"
    assert m.reasoning_effort == "low"


@pytest.mark.parametrize("effort", ["", "none", "NONE", " none "])
def test_effort_none_not_passed(effort):
    assert build_chat_model(spec("openai", effort=effort), "gen").reasoning_effort is None
    assert build_chat_model(spec("anthropic", effort=effort), "gen").reasoning_effort is None
    assert build_chat_model(spec("gemini", effort=effort), "gen").reasoning_effort is None


def test_effort_outside_package_table_ignored():
    assert build_chat_model(spec("anthropic", effort="xhigh"), "gen").reasoning_effort == "xhigh"
    assert build_chat_model(spec("gemini", effort="xhigh"), "gen").reasoning_effort is None


def test_missing_model_or_key_and_unknown_protocol():
    with pytest.raises(LlmError, match="gen 角色"):
        build_chat_model(spec("openai", model=""), "gen")
    with pytest.raises(LlmError, match="review 角色"):
        build_chat_model(spec("openai", api_key=" "), "review")
    with pytest.raises(LlmError, match="不支持"):
        build_chat_model(spec("azure"), "gen")
