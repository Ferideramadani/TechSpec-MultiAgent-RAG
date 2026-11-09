"""Konfigurimi i lidhjes me Claude (Anthropic API).

Importimi i këtij moduli nuk kërkon çelës API dhe nuk bën asnjë thirrje në rrjet.
Çelësi lexohet nga variabla ANTHROPIC_API_KEY në skedarin .env.
"""

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Kur filtrat e sigurisë refuzojnë një kërkesë, Anthropic e ridrejton vetë te një model rezervë.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass(frozen=True)
class Settings:
    """Parametrat e përbashkët për të gjithë agjentët."""

    # repr=False: çelësi nuk shfaqet në log-e, mesazhe gabimi apo raporte testesh.
    anthropic_api_key: str = field(default=os.getenv("ANTHROPIC_API_KEY", ""), repr=False)
    # claude-3-5-sonnet-20241022 u tërhoq më 28 tetor 2025; claude-sonnet-5-5 është pasardhësi i tij.
    model: str = os.getenv("TECHSPEC_CLAUDE_MODEL", "claude-sonnet-5-5")
    # Mjafton për përgjigjet më të gjata (verifikimi i shumë elementeve) pa streaming.
    max_tokens: int = int(os.getenv("TECHSPEC_MAX_TOKENS", "16000"))
    timeout_seconds: float = float(os.getenv("TECHSPEC_TIMEOUT_SECONDS", "300"))
    max_retries: int = int(os.getenv("TECHSPEC_MAX_RETRIES", "2"))


settings = Settings()


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Kthen një model të vetëm, të ripërdorur nga të gjithë agjentët.

    `temperature` nuk dërgohet, sepse claude-sonnet-5-5 e refuzon çdo vlerë jo të paracaktuar.
    Agjentët e kontrollojnë `settings.anthropic_api_key` para se ta thërrasin.
    """
    return ChatAnthropic(
        model=settings.model,
        api_key=settings.anthropic_api_key,
        max_tokens=settings.max_tokens,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
        betas=[FALLBACK_BETA],
        model_kwargs={"fallbacks": "default"},
    )
