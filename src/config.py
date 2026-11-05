"""Konfigurimi i lidhjes me Claude API.

Importimi i këtij moduli nuk kërkon çelës API dhe nuk bën asnjë thirrje në rrjet.
Kredencialet lexohen nga SDK-ja kur krijohet klienti, zakonisht nga variabla
ANTHROPIC_API_KEY në skedarin .env.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Beta-ja që aktivizon kalimin automatik te modeli rezervë kur kërkesa refuzohet.
_FALLBACK_BETA = "server-side-fallback-2026-06-01"


@dataclass(frozen=True)
class Settings:
    """Parametrat e përbashkët për të gjithë agjentët."""

    # Prefiksi TECHSPEC_ shmang përplasjen me variabla si CLAUDE_EFFORT,
    # që Claude Code i vendos në terminalet e veta.
    model: str = os.getenv("TECHSPEC_MODEL", "claude-opus-5-5")
    fallback_model: str = os.getenv("TECHSPEC_FALLBACK_MODEL", "claude-opus-4-8")
    max_tokens: int = int(os.getenv("TECHSPEC_MAX_TOKENS", "64000"))
    # Vlerat e lejuara: low, medium, high, xhigh, max.
    effort: str = os.getenv("TECHSPEC_EFFORT", "high")
    timeout_seconds: float = float(os.getenv("TECHSPEC_TIMEOUT_SECONDS", "600"))
    max_retries: int = int(os.getenv("TECHSPEC_MAX_RETRIES", "3"))


settings = Settings()


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    """Kthen një klient të vetëm, të ripërdorur nga të gjithë agjentët."""
    return anthropic.Anthropic(
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )


def request_params() -> dict[str, Any]:
    """Kthen parametrat bazë për `client.beta.messages.stream(...)`.

    Agjenti shton vetëm `system` dhe `messages`. Duhet përdorur streaming, sepse
    vlera e lartë e `max_tokens` e tejkalon kufirin e SDK-së për kërkesat pa
    streaming. Para se të lexohet përmbajtja, duhet kontrolluar `stop_reason`.
    """
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "output_config": {"effort": settings.effort},
        "betas": [_FALLBACK_BETA],
        "fallbacks": [{"model": settings.fallback_model}],
    }
