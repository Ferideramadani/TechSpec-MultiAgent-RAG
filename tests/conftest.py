from dataclasses import replace

import pytest
from fake_llm import FakeLLM, answer
from langchain_anthropic import ChatAnthropic

from src import config
from src.utils import llm


@pytest.fixture(scope="session")
def llm_server():
    server = FakeLLM()
    yield server
    server.server.shutdown()


@pytest.fixture
def llm_api(llm_server, monkeypatch):
    """Çdo thirrje te Claude shkon te serveri lokal, me një çelës provë."""
    llm_server.requests.clear()
    llm_server.respond = lambda: answer({})
    model = ChatAnthropic(
        model=config.settings.model,
        api_key="test-key",
        base_url=llm_server.url,
        max_tokens=config.settings.max_tokens,
        max_retries=0,
        timeout=10,
        **config.fallback_options(config.settings.model),
    )
    monkeypatch.setattr(llm, "get_llm", lambda: model)
    monkeypatch.setattr(llm, "settings", replace(config.settings, anthropic_api_key="test-key"))
    return llm_server
