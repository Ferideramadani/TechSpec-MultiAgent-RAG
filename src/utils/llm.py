"""Thirrja e përbashkët te Claude për agjentët që kërkojnë përgjigje të strukturuar.

Të gjitha gabimet e API-së dhe përgjigjet e papërdorshme shndërrohen në `LLMError`
me një mesazh në shqip, që agjenti ta shkruajë te `errors` dhe aplikacioni ta shfaqë.
"""

import logging

import anthropic
from langchain_core.exceptions import ModelRefusalError
from pydantic import BaseModel

from src.config import get_llm, settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Thirrja te Claude dështoi; mesazhi është gati për t'iu shfaqur përdoruesit."""


def invoke_structured[T: BaseModel](schema: type[T], system_prompt: str, user_prompt: str) -> T:
    """Thërret Claude dhe kthen përgjigjen e vlefshme sipas `schema`.

    Përdoren daljet e strukturuara (`json_schema`), sepse claude-sonnet-5-5 nuk e
    pranon thirrjen e detyruar të një funksioni, që LangChain e përdor si paracaktim.

    Raises:
        LLMError: çelësi mungon, API-ja ktheu gabim, ose përgjigja nuk mund të përdoret.
    """
    if not settings.anthropic_api_key:
        raise LLMError("Çelësi API mungon. Shto ANTHROPIC_API_KEY te .env.")

    # Me include_raw, gabimet e analizës kthehen te `parsing_error` në vend që të ngrihen.
    model = get_llm().with_structured_output(schema, method="json_schema", include_raw=True)
    try:
        output = model.invoke([("system", system_prompt), ("human", user_prompt)])
    except anthropic.AuthenticationError as exc:
        raise LLMError("Çelësi API nuk është i vlefshëm. Kontrollo ANTHROPIC_API_KEY te .env.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMError(f"Çelësi API nuk ka leje për këtë kërkesë: {_api_message(exc)}") from exc
    except anthropic.NotFoundError as exc:
        raise LLMError(
            f"Modeli '{settings.model}' nuk është i disponueshëm. Ndrysho TECHSPEC_CLAUDE_MODEL "
            f"te .env. ({_api_message(exc)})"
        ) from exc
    except anthropic.RateLimitError as exc:
        raise LLMError("U tejkalua kufiri i kërkesave të Claude. Provo përsëri pas pak.") from exc
    except anthropic.BadRequestError as exc:
        if "credit balance" in _api_message(exc):
            raise LLMError(
                "Llogaria e Anthropic nuk ka kredit të mjaftueshëm. Shto kredit te "
                "console.anthropic.com, te Plans & Billing."
            ) from exc
        raise LLMError(f"Claude e refuzoi kërkesën: {_api_message(exc)}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Claude ktheu gabim ({exc.status_code}): {_api_message(exc)}") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Nuk u arrit lidhja me Anthropic API. Kontrollo internetin.") from exc

    raw = output["raw"]
    if raw.usage_metadata:
        logger.info(
            "Claude (%s): %d tokena hyrje, %d tokena dalje.",
            schema.__name__,
            raw.usage_metadata["input_tokens"],
            raw.usage_metadata["output_tokens"],
        )
    stop_reason = raw.response_metadata.get("stop_reason")
    if stop_reason == "refusal" or isinstance(output["parsing_error"], ModelRefusalError):
        raise LLMError("Claude refuzoi ta përpunojë këtë përmbajtje.")
    # LangChain e plotëson në heshtje JSON-in e prerë, ndaj ndërprerja kontrollohet veçmas.
    if stop_reason == "max_tokens":
        raise LLMError("Përgjigja e Claude u ndërpre para se të përfundonte. Rrit TECHSPEC_MAX_TOKENS te .env.")
    if output["parsing_error"] or output["parsed"] is None:
        raise LLMError(f"Claude ktheu një përgjigje që nuk përputhet me skemën: {output['parsing_error']}")
    return output["parsed"]


def _api_message(exc: anthropic.APIStatusError) -> str:
    """Kthen vetëm tekstin e gabimit nga Anthropic, pa JSON-in e plotë të përgjigjes."""
    error = exc.body.get("error") if isinstance(exc.body, dict) else None
    if isinstance(error, dict) and error.get("message"):
        return error["message"]
    return exc.message
