from dataclasses import replace

import pytest
from fake_gemini import FakeGemini, answer
from langchain_google_genai import ChatGoogleGenerativeAI

from src import config
from src.utils import llm


@pytest.fixture(scope="session")
def gemini_server():
    server = FakeGemini()
    yield server
    server.server.shutdown()


@pytest.fixture
def gemini(gemini_server, monkeypatch):
    """Çdo thirrje te Gemini shkon te serveri lokal, me një çelës provë."""
    gemini_server.requests.clear()
    gemini_server.respond = lambda: answer({})
    model = ChatGoogleGenerativeAI(
        model=config.settings.model,
        google_api_key="test-key",
        base_url=gemini_server.url,
        max_retries=0,
        timeout=10,
    )
    monkeypatch.setattr(llm, "get_llm", lambda: model)
    monkeypatch.setattr(llm, "settings", replace(config.settings, gemini_api_key="test-key"))
    return gemini_server
