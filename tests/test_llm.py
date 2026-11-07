from dataclasses import replace

import pytest
from fake_gemini import answer, api_error
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src import config
from src.utils import llm
from src.utils.llm import LLMError, invoke_structured


class Answer(BaseModel):
    items: list[str]


def call():
    return invoke_structured(Answer, "system prompt", "user prompt")


def test_returns_the_validated_object(gemini):
    gemini.respond = lambda: answer({"items": ["a", "b"]})

    assert call() == Answer(items=["a", "b"])


def test_request_sends_model_prompts_and_schema(gemini):
    gemini.respond = lambda: answer({"items": []})

    call()

    [request] = gemini.requests
    assert f"models/{config.settings.model}:generateContent" in request["path"]
    assert gemini.system_prompt == "system prompt"
    assert gemini.prompt == "user prompt"
    generation = request["body"]["generationConfig"]
    assert generation["responseMimeType"] == "application/json"
    assert set(generation["responseJsonSchema"]["properties"]) == {"items"}


def test_missing_api_key_means_no_request(gemini, monkeypatch):
    monkeypatch.setattr(llm, "settings", replace(config.settings, gemini_api_key=""))

    with pytest.raises(LLMError, match="GEMINI_API_KEY"):
        call()
    assert gemini.requests == []


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ((200, {"candidates": [{"finishReason": "SAFETY"}]}), "refuzoi"),
        ((200, {"promptFeedback": {"blockReason": "SAFETY"}}), "refuzoi"),
        # JSON i prerë: LangChain do ta plotësonte në heshtje si të vlefshëm.
        (answer('{"items": ["a", "b', "MAX_TOKENS"), "u ndërpre"),
        (answer('{"items": 5}'), "skemën"),
        (answer("not json"), "skemën"),
    ],
)
def test_unusable_responses_raise(gemini, response, message):
    gemini.respond = lambda: response

    with pytest.raises(LLMError, match=message):
        call()


@pytest.mark.parametrize(
    ("status", "error_status", "api_message", "expected"),
    [
        (400, "INVALID_ARGUMENT", "API key not valid. Please pass a valid API key.", "GEMINI_API_KEY"),
        (400, "INVALID_ARGUMENT", "Request contains an invalid argument.", "invalid argument"),
        (403, "PERMISSION_DENIED", "API has not been used in project 1.", "Generative Language API"),
        (404, "NOT_FOUND", "models/x is no longer available to new users.", "TECHSPEC_GEMINI_MODEL"),
        (404, "NOT_FOUND", "models/x is no longer available to new users.", "no longer available"),
        (429, "RESOURCE_EXHAUSTED", "Quota exceeded.", "kufiri"),
        (500, "INTERNAL", "Internal error.", "Gemini ktheu gabim: Internal error."),
        (503, "UNAVAILABLE", "The model is overloaded.", "Gemini ktheu gabim: The model is overloaded."),
    ],
)
def test_api_errors_raise_readable_messages(gemini, status, error_status, api_message, expected):
    gemini.respond = lambda: api_error(status, error_status, api_message)

    with pytest.raises(LLMError) as error:
        call()
    assert expected in str(error.value)


def test_error_shows_only_the_readable_message(gemini):
    gemini.respond = lambda: api_error(400, "INVALID_ARGUMENT", "Request contains an invalid argument.")

    with pytest.raises(LLMError) as error:
        call()
    assert str(error.value) == "Gemini e refuzoi kërkesën: Request contains an invalid argument."


def test_connection_error_raises(monkeypatch):
    model = ChatGoogleGenerativeAI(
        model="gemini-test", google_api_key="test-key", base_url="http://127.0.0.1:9", max_retries=0, timeout=5
    )
    monkeypatch.setattr(llm, "get_llm", lambda: model)
    monkeypatch.setattr(llm, "settings", replace(config.settings, gemini_api_key="test-key"))

    with pytest.raises(LLMError, match="lidhja"):
        call()


def test_unrelated_error_is_not_hidden(gemini, monkeypatch):
    def broken():
        raise RuntimeError("bug in our code")

    monkeypatch.setattr(llm, "get_llm", broken)

    with pytest.raises(RuntimeError, match="bug in our code"):
        call()


def test_get_llm_uses_key_and_model_from_settings(monkeypatch):
    monkeypatch.setattr(config, "settings", replace(config.settings, gemini_api_key="key-123", model="gemini-x"))
    config.get_llm.cache_clear()
    try:
        model = config.get_llm()
        assert model.model.endswith("gemini-x")
        assert model.google_api_key.get_secret_value() == "key-123"
        assert model.max_retries == config.settings.max_retries
    finally:
        config.get_llm.cache_clear()
