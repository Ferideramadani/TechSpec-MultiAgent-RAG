"""Requirement Quality Critic Agent.

Vlerëson cilësinë e kërkesave të nxjerra sipas pesë kritereve dhe sugjeron një
version të përmirësuar për secilën. Modeli gjykon vetëm cilat kritere nuk
plotësohen; përqindja llogaritet këtu, që të jetë e përsëritshme dhe e shpjegueshme.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal, get_args

from pydantic import BaseModel

from src.state import (
    DocumentChunk,
    QualityCriterion,
    QualityIssue,
    Requirement,
    RequirementReview,
    TechSpecState,
)
from src.utils.llm import LLMError, invoke_structured
from src.utils.sources import format_chunk

logger = logging.getLogger(__name__)

AGENT_NAME = "critic"
CRITERIA: tuple[QualityCriterion, ...] = get_args(QualityCriterion)

SYSTEM_PROMPT = """\
Review each <requirement> (with its id, kind, priority and source chunk ids; the chunks follow in <chunk> tags) against five criteria, and report an issue only for a failed one:
- clarity: understood on first reading; one requirement per statement.
- ambiguity: no vague words such as "fast", "robust", "appropriate", "user-friendly" or "etc.".
- testability: a pass/fail test can be written for it.
- measurability: a quality has a number, unit or threshold; a function has defined inputs, outputs and conditions.
- completeness: the actor, the action and the conditions are present.

problem: one short sentence naming the exact words or the missing detail. Judge the statement, not the document.
improved_version: a rewrite that passes all five criteria and keeps the meaning. Never invent facts, numbers, names or technologies absent from the chunks; use a placeholder such as [TBD: maximum response time in ms].
summary: two sentences on the overall quality of the set and its most common problem.
Return exactly one review per requirement id."""


class ExtractedIssue(BaseModel):
    criterion: Literal["clarity", "ambiguity", "testability", "measurability", "completeness"]
    problem: str


class ExtractedReview(BaseModel):
    requirement_id: str
    issues: list[ExtractedIssue]
    improved_version: str


class CritiqueResult(BaseModel):
    reviews: list[ExtractedReview]
    summary: str


def critique_requirements(state: TechSpecState) -> dict[str, Any]:
    """Nyja e LangGraph: plotëson `requirement_reviews`, `quality_score` dhe `quality_summary`.

    Mungesa e kërkesave nuk është gabim: dokumenti thjesht mund të mos ketë asnjë.
    Gabimet e modelit shënohen te `errors` në vend që të ngrihen.
    """
    requirements = state["requirements"]
    if not requirements:
        return {
            "requirement_reviews": [],
            "quality_score": None,
            "quality_summary": "Nuk ka kërkesa për të vlerësuar.",
        }

    chunks = state["retrieved_chunks"] or state["document_chunks"]
    try:
        result = invoke_structured(CritiqueResult, SYSTEM_PROMPT, _build_prompt(requirements, chunks))
    except LLMError as exc:
        return _failure(str(exc))

    reviews = _to_reviews(result.reviews, requirements)
    if not reviews:
        return _failure("Modeli nuk ktheu vlerësim për asnjë nga kërkesat.")

    return {
        "requirement_reviews": reviews,
        "quality_score": round(sum(review["score"] for review in reviews) / len(reviews)),
        "quality_summary": result.summary,
    }


def score_for(issues: Sequence[QualityIssue]) -> int:
    """Përqindja e kritereve që plotësohen; disa probleme te i njëjti kriter numërohen një herë."""
    failed = {issue["criterion"] for issue in issues}
    return round(100 * (len(CRITERIA) - len(failed)) / len(CRITERIA))


def _build_prompt(requirements: Sequence[Requirement], chunks: Sequence[DocumentChunk]) -> str:
    items = "\n".join(
        f'<requirement id="{req["requirement_id"]}" kind="{req["kind"]}" priority="{req["priority"]}" '
        f'sources="{",".join(req["source_chunk_ids"])}">{req["description"]}</requirement>'
        for req in requirements
    )
    # Dërgohen vetëm pjesët e cituara, që kërkesa të gjykohet në kontekstin e saj.
    cited = {chunk_id for req in requirements for chunk_id in req["source_chunk_ids"]}
    excerpts = "\n".join(
        format_chunk(chunk)
        for chunk in chunks
        if chunk["chunk_id"] in cited
    )
    return f"<requirements>\n{items}\n</requirements>\n\n<chunks>\n{excerpts}\n</chunks>"


def _to_reviews(items: Sequence[ExtractedReview], requirements: Sequence[Requirement]) -> list[RequirementReview]:
    by_id = {item.requirement_id: item for item in reversed(items)}  # Ruhet vlerësimi i parë për çdo ID.
    reviews: list[RequirementReview] = []
    for req in requirements:
        item = by_id.get(req["requirement_id"])
        if item is None:
            logger.warning("Modeli nuk e vlerësoi kërkesën %s.", req["requirement_id"])
            continue
        issues: list[QualityIssue] = [
            {"criterion": issue.criterion, "problem": issue.problem} for issue in item.issues
        ]
        reviews.append(
            {
                "requirement_id": req["requirement_id"],
                "score": score_for(issues),
                "issues": issues,
                "improved_version": item.improved_version,
            }
        )
    unknown = set(by_id) - {req["requirement_id"] for req in requirements}
    if unknown:
        logger.warning("U hodhën vlerësime për kërkesa që nuk ekzistojnë: %s", sorted(unknown))
    return reviews


def _failure(message: str) -> dict[str, Any]:
    logger.warning(message)
    return {
        "requirement_reviews": [],
        "quality_score": None,
        "quality_summary": "",
        "errors": [{"agent": AGENT_NAME, "message": message}],
    }
