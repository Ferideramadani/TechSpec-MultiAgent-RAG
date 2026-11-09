"""API & Functional Requirement Extractor Agent.

Dërgon te modeli gjuhësor pjesët e gjetura nga retriever-i dhe merr kërkesat softuerike
dhe pikat API si JSON të strukturuar.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel

from src.state import ApiEndpoint, Requirement, RetrievedChunk, TechSpecState
from src.utils.llm import LLMError, invoke_structured
from src.utils.sources import format_chunk

logger = logging.getLogger(__name__)

AGENT_NAME = "requirement_extractor"

SYSTEM_PROMPT = """\
You extract software requirements and API specifications from excerpts of a \
technical document. Each excerpt is in a <chunk> tag with an id and a page number.

Extract only what the excerpts state or directly describe. Do not add requirements \
from general knowledge about similar systems: a fact-verification step later checks \
every item against the chunks it cites. List in source_chunk_ids the ids of the \
chunks that support each item.

- Functional requirements describe what the system does: features, behaviors, operations.
- Non-functional requirements describe qualities and constraints: performance, security, \
availability, scalability, compliance, usability.
- API endpoints: include only endpoints whose HTTP method and path appear in the text. \
Do not construct a path from a prose description.

Set priority to high for mandatory wording (must, shall, required), low for optional \
wording (may, optional), and medium otherwise. Write each description as one \
self-contained sentence in English.

The excerpts may come from material that is not a specification, such as a market \
report. Then return only what the text presents as a requirement of a system; empty \
lists are a correct answer when the text contains none.\
"""


class ExtractedRequirement(BaseModel):
    kind: Literal["functional", "non_functional"]
    description: str
    priority: Literal["low", "medium", "high"]
    source_chunk_ids: list[str]


class ExtractedEndpoint(BaseModel):
    method: str
    path: str
    description: str
    parameters: list[str]
    source_chunk_ids: list[str]


class ExtractionResult(BaseModel):
    requirements: list[ExtractedRequirement]
    api_endpoints: list[ExtractedEndpoint]


def extract_requirements(state: TechSpecState) -> dict[str, Any]:
    """Nyja e LangGraph: plotëson `requirements` dhe `api_endpoints`.

    Gabimet e modelit shënohen te `errors` në vend që të ngrihen, sepse janë
    dështime të jashtme që aplikacioni duhet t'i shfaqë.
    """
    chunks = state["retrieved_chunks"]
    if not chunks:
        return _failure("Nuk ka pjesë të dokumentit për analizë. Ekzekuto së pari retriever-in.")
    try:
        result = invoke_structured(ExtractionResult, SYSTEM_PROMPT, _build_prompt(state["query"], chunks))
    except LLMError as exc:
        return _failure(str(exc))

    valid_ids = {chunk["chunk_id"] for chunk in chunks}
    return {
        "requirements": _to_requirements(result.requirements, valid_ids),
        "api_endpoints": _to_endpoints(result.api_endpoints, valid_ids),
    }


def _build_prompt(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    excerpts = "\n".join(
        format_chunk(chunk)
        for chunk in chunks
    )
    return f"The user is analyzing this question: {query}\n\n<chunks>\n{excerpts}\n</chunks>"


def _cited_ids(cited: list[str], valid_ids: set[str]) -> list[str]:
    # Ruhen vetëm ID-të që ekzistojnë vërtet, që verifikuesi të mos ndjekë citime të shpikura.
    return [chunk_id for chunk_id in dict.fromkeys(cited) if chunk_id in valid_ids]


def _to_requirements(items: list[ExtractedRequirement], valid_ids: set[str]) -> list[Requirement]:
    requirements: list[Requirement] = []
    for item in items:
        source_ids = _cited_ids(item.source_chunk_ids, valid_ids)
        if not source_ids:
            logger.warning("U hoq një kërkesë pa burim të vlefshëm: %s", item.description)
            continue
        requirements.append(
            {
                "requirement_id": f"REQ-{len(requirements) + 1:03d}",
                "kind": item.kind,
                "description": item.description,
                "priority": item.priority,
                "source_chunk_ids": source_ids,
            }
        )
    return requirements


def _to_endpoints(items: list[ExtractedEndpoint], valid_ids: set[str]) -> list[ApiEndpoint]:
    endpoints: list[ApiEndpoint] = []
    for item in items:
        source_ids = _cited_ids(item.source_chunk_ids, valid_ids)
        if not source_ids:
            logger.warning("U hoq një pikë API pa burim të vlefshëm: %s %s", item.method, item.path)
            continue
        endpoints.append(
            {
                "method": item.method.upper(),
                "path": item.path,
                "description": item.description,
                "parameters": item.parameters,
                "source_chunk_ids": source_ids,
            }
        )
    return endpoints


def _failure(message: str) -> dict[str, Any]:
    logger.warning(message)
    return {
        "requirements": [],
        "api_endpoints": [],
        "errors": [{"agent": AGENT_NAME, "message": message}],
    }
