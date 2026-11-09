from dataclasses import replace

import pytest
from fake_llm import answer, api_error
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel

from src import config
from src.utils import llm
from src.utils.llm import LLMError, invoke_structured


class Answer(BaseModel):
    items: list[str]


def call():
    return invoke_structured(Answer, "system prompt", "user prompt")


def test_returns_the_validated_object(llm_api):
    llm_api.respond = lambda: answer({"items": ["a", "b"]})

    assert call() == Answer(items=["a", "b"])


def test_request_uses_structured_output_without_forced_tools_or_temperature(llm_api):
    llm_api.respond = lambda: answer({"items": []})

    call()

    [request] = llm_api.requests
    body = request["body"]
    assert request["path"].endswith("/v1/messages")
    assert body["model"] == config.settings.model
    assert body["max_tokens"] == config.settings.max_tokens
    assert llm_api.system_prompt == "system prompt"
    assert llm_api.prompt == "user prompt"
    assert body["output_config"]["format"]["type"] == "json_schema"
    assert set(llm_api.schema_properties) == {"items"}
    # claude-sonnet-5-5 kthen 400 për thirrje të detyruar funksioni dhe për temperaturë jo të paracaktuar.
    assert "tool_choice" not in body
    assert "temperature" not in body


def test_request_enables_the_refusal_fallback(llm_api):
    llm_api.respond = lambda: answer({"items": []})

    call()

    [request] = llm_api.requests
    assert request["body"]["fallbacks"] == "default"
    assert config.FALLBACK_BETA in request["headers"]["anthropic-beta"]


def test_missing_api_key_means_no_request(llm_api, monkeypatch):
    monkeypatch.setattr(llm, "settings", replace(config.settings, anthropic_api_key=""))

    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        call()
    assert llm_api.requests == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (answer("", "refusal"), "refuzoi"),
        # JSON i prerë: LangChain do ta plotësonte në heshtje si të vlefshëm.
        (answer('{"items": ["a", "b', "max_tokens"), "u ndërpre"),
        (answer('{"items": 5}'), "skemën"),
        (answer("not json"), "skemën"),
    ],
)
def test_unusable_responses_raise(llm_api, response, message):
    llm_api.respond = lambda: response

    with pytest.raises(LLMError, match=message):
        call()


@pytest.mark.parametrize(
    ("status", "error_type", "api_message", "expected"),
    [
        (401, "authentication_error", "invalid x-api-key", "ANTHROPIC_API_KEY"),
        (403, "permission_error", "This key cannot use this model.", "This key cannot use this model."),
        (404, "not_found_error", "model: claude-3-5-sonnet-20241022", "TECHSPEC_CLAUDE_MODEL"),
        (429, "rate_limit_error", "Rate limited.", "kufiri"),
        (400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API.", "Plans & Billing"),
        (400, "invalid_request_error", "max_tokens: too large.", "Claude e refuzoi kërkesën: max_tokens: too large."),
        (500, "api_error", "Internal error.", "Claude ktheu gabim (500): Internal error."),
        (529, "overloaded_error", "Overloaded", "Claude ktheu gabim (529): Overloaded"),
    ],
)
def test_api_errors_raise_readable_messages(llm_api, status, error_type, api_message, expected):
    llm_api.respond = lambda: api_error(status, error_type, api_message)

    with pytest.raises(LLMError) as error:
        call()
    assert expected in str(error.value)


def test_error_shows_only_the_readable_message(llm_api):
    llm_api.respond = lambda: api_error(400, "invalid_request_error", "max_tokens: too large.")

    with pytest.raises(LLMError) as error:
        call()
    assert str(error.value) == "Claude e refuzoi kërkesën: max_tokens: too large."


def test_connection_error_raises(monkeypatch):
    model = ChatAnthropic(model="claude-test", api_key="test-key", base_url="http://127.0.0.1:9", max_retries=0, timeout=5)
    monkeypatch.setattr(llm, "get_llm", lambda: model)
    monkeypatch.setattr(llm, "settings", replace(config.settings, anthropic_api_key="test-key"))

    with pytest.raises(LLMError, match="lidhja"):
        call()


def test_unrelated_error_is_not_hidden(llm_api, monkeypatch):
    def broken():
        raise RuntimeError("bug in our code")

    monkeypatch.setattr(llm, "get_llm", broken)

    with pytest.raises(RuntimeError, match="bug in our code"):
        call()


def test_settings_repr_never_shows_the_api_key():
    settings = config.Settings(anthropic_api_key="secret-key-123")

    assert "secret-key-123" not in repr(settings)
    assert "secret-key-123" not in str(settings)


def test_get_llm_uses_settings(monkeypatch):
    monkeypatch.setattr(
        config, "settings", replace(config.settings, anthropic_api_key="key-123", model="claude-x", max_tokens=1234)
    )
    config.get_llm.cache_clear()
    try:
        model = config.get_llm()
        assert model.model == "claude-x"
        assert model.anthropic_api_key.get_secret_value() == "key-123"
        assert model.max_tokens == 1234
        assert model.temperature is None
        assert model.betas == [config.FALLBACK_BETA]
    finally:
        config.get_llm.cache_clear()
