"""Thirrja e përbashkët te Gemini për agjentët që kërkojnë përgjigje të strukturuar.

Të gjitha gabimet e API-së dhe përgjigjet e papërdorshme shndërrohen në `LLMError`
me një mesazh në shqip, që agjenti ta shkruajë te `errors` dhe aplikacioni ta shfaqë.
"""

import logging

import httpx2
from google.genai import errors as genai_errors
from langchain_core.exceptions import (
    ModelAuthenticationError,
    ModelConnectionError,
    ModelError,
    ModelInvalidRequestError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRefusalError,
    ModelTimeoutError,
)
from pydantic import BaseModel

from src.config import get_llm, settings

logger = logging.getLogger(__name__)

_INVALID_KEY = "Çelësi API nuk është i vlefshëm. Kontrollo GEMINI_API_KEY te .env."


class LLMError(Exception):
    """Thirrja te Gemini dështoi; mesazhi është gati për t'iu shfaqur përdoruesit."""


def invoke_structured[T: BaseModel](schema: type[T], system_prompt: str, user_prompt: str) -> T:
    """Thërret Gemini dhe kthen përgjigjen e vlefshme sipas `schema`.

    Raises:
        LLMError: çelësi mungon, API-ja ktheu gabim, ose përgjigja nuk mund të përdoret.
    """
    if not settings.gemini_api_key:
        raise LLMError("Çelësi API mungon. Shto GEMINI_API_KEY te .env.")

    # Me include_raw, gabimet e analizës kthehen te `parsing_error` në vend që të ngrihen.
    model = get_llm().with_structured_output(schema, method="json_schema", include_raw=True)
    try:
        output = model.invoke([("system", system_prompt), ("human", user_prompt)])
    except ModelAuthenticationError as exc:
        raise LLMError(_INVALID_KEY) from exc
    except ModelInvalidRequestError as exc:
        # Gemini e raporton çelësin e pavlefshëm si 400, jo si 401.
        if "API key" in _api_message(exc):
            raise LLMError(_INVALID_KEY) from exc
        raise LLMError(f"Gemini e refuzoi kërkesën: {_api_message(exc)}") from exc
    except ModelPermissionDeniedError as exc:
        raise LLMError(
            "Çelësi nuk ka leje. Kontrollo që Generative Language API është aktivizuar "
            f"në projektin Google Cloud. ({_api_message(exc)})"
        ) from exc
    except ModelNotFoundError as exc:
        raise LLMError(
            f"Modeli '{settings.model}' nuk është i disponueshëm. Ndrysho TECHSPEC_GEMINI_MODEL "
            f"te .env. ({_api_message(exc)})"
        ) from exc
    except ModelRateLimitError as exc:
        raise LLMError("U tejkalua kufiri i kërkesave të Gemini. Provo përsëri pas pak.") from exc
    except (ModelConnectionError, ModelTimeoutError, httpx2.TransportError) as exc:
        raise LLMError("Nuk u arrit lidhja me Gemini API. Kontrollo internetin.") from exc
    except ModelError as exc:
        raise LLMError(f"Gemini ktheu gabim: {_api_message(exc)}") from exc

    raw = output["raw"]
    if raw.usage_metadata:
        logger.info(
            "Gemini (%s): %d tokena hyrje, %d tokena dalje.",
            schema.__name__,
            raw.usage_metadata["input_tokens"],
            raw.usage_metadata["output_tokens"],
        )
    if isinstance(output["parsing_error"], ModelRefusalError):
        raise LLMError("Gemini refuzoi ta përpunojë këtë përmbajtje.")
    # LangChain e plotëson në heshtje JSON-in e prerë, ndaj ndërprerja kontrollohet veçmas.
    if raw.response_metadata.get("finish_reason") == "MAX_TOKENS":
        raise LLMError("Përgjigja e Gemini u ndërpre para se të përfundonte.")
    if output["parsing_error"] or output["parsed"] is None:
        raise LLMError(f"Gemini ktheu një përgjigje që nuk përputhet me skemën: {output['parsing_error']}")
    return output["parsed"]


def _api_message(exc: Exception) -> str:
    """Kthen vetëm tekstin e gabimit nga Google, pa JSON-in e plotë të përgjigjes."""
    for error in (exc, exc.__cause__):
        if isinstance(error, genai_errors.APIError) and error.message:
            return error.message
    return str(exc)
