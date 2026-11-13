"""Konfigurimi i lidhjes me Claude (Anthropic API).

Importimi i këtij moduli nuk kërkon çelës API dhe nuk bën asnjë thirrje në rrjet.
Çelësi lexohet nga variabla ANTHROPIC_API_KEY në skedarin .env.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Kur filtrat e sigurisë refuzojnë një kërkesë, Anthropic e ridrejton vetë te një model rezervë.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Modelet që e pranojnë modelin rezervë; claude-haiku-5-5 nuk e ka këtë mundësi.
FALLBACK_MODELS = frozenset({"claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1"})


@dataclass(frozen=True)
class Settings:
    """Parametrat e përbashkët për të gjithë agjentët."""

    # repr=False: çelësi nuk shfaqet në log-e, mesazhe gabimi apo raporte testesh.
    anthropic_api_key: str = field(default=os.getenv("ANTHROPIC_API_KEY", ""), repr=False)
    # claude-haiku-5-5 kushton $0.10 / $0.50 për milion tokena, 20 herë më pak se claude-sonnet-5-5.
    # Për cilësi më të lartë: TECHSPEC_CLAUDE_MODEL=claude-sonnet-5-5 te .env.
    model: str = os.getenv("TECHSPEC_CLAUDE_MODEL", "claude-haiku-5-5")
    # Kufi sigurie, jo mjet kursimi: modeli nuk e sheh, dhe një përgjigje e prerë paguhet sërish.
    # Te modelet e reja edhe të menduarit numërohet këtu, ndaj nuk duhet ulur shumë.
    max_tokens: int = int(os.getenv("TECHSPEC_MAX_TOKENS", "16000"))
    timeout_seconds: float = float(os.getenv("TECHSPEC_TIMEOUT_SECONDS", "300"))
    max_retries: int = int(os.getenv("TECHSPEC_MAX_RETRIES", "2"))


settings = Settings()


def fallback_options(model: str) -> dict[str, Any]:
    """Parametrat e modelit rezervë, vetëm për modelet që e mbështesin."""
    if model not in FALLBACK_MODELS:
        return {}
    return {"betas": [FALLBACK_BETA], "model_kwargs": {"fallbacks": "default"}}


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Kthen një model të vetëm, të ripërdorur nga të gjithë agjentët.

    `temperature` nuk dërgohet, sepse modelet e reja Claude e refuzojnë çdo vlerë jo të paracaktuar.
    Agjentët e kontrollojnë `settings.anthropic_api_key` para se ta thërrasin.
    """
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        **fallback_options(settings.model),
    )
