"""Konfigurimi i lidhjes me Google Gemini API.

Importimi i këtij moduli nuk kërkon çelës API dhe nuk bën asnjë thirrje në rrjet.
Çelësi lexohet nga variabla GEMINI_API_KEY në skedarin .env.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Parametrat e përbashkët për të gjithë agjentët."""

    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # Google i tërheq modelet shpesh; atëherë mjafton të ndryshohet TECHSPEC_GEMINI_MODEL te .env.
    # gemini-3.8-flash i kalonte shpesh 120 sekondat në nivelin falas; 3.5-flash përgjigjet për ~20 s.
    model: str = os.getenv("TECHSPEC_GEMINI_MODEL", "gemini-3.5-flash")
    timeout_seconds: float = float(os.getenv("TECHSPEC_TIMEOUT_SECONDS", "120"))
    # Pak riprovime, që kufiri i nivelit falas të mos e mbajë aplikacionin në pritje.
    max_retries: int = int(os.getenv("TECHSPEC_MAX_RETRIES", "1"))


settings = Settings()


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Kthen një model të vetëm, të ripërdorur nga të gjithë agjentët.

    Krijimi dështon nëse çelësi mungon, ndaj agjentët e kontrollojnë
    `settings.gemini_api_key` para se ta thërrasin.
    """
    return ChatGoogleGenerativeAI(
        model=settings.model,
        google_api_key=settings.gemini_api_key,
        timeout=settings.timeout_seconds,
        max_retries=settings.max_retries,
    )
