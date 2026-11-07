"""Citation & Fact-Verifier Agent.

Kontrollon me Gemini që kërkesat, versionet e tyre të përmirësuara dhe rastet e
testimit mbështeten te dokumenti dhe nuk shtojnë supozime të pabazuara. Teksti i
verifikuar dhe burimet merren nga gjendja; nga modeli merret vetëm verdikti.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel

from src.state import (
    DocumentChunk,
    Requirement,
    RequirementReview,
    TechSpecState,
    TestCase,
    VerificationResult,
)
from src.utils.llm import LLMError, invoke_structured

logger = logging.getLogger(__name__)

AGENT_NAME = "verifier"
IMPROVED_SUFFIX = ":improved"

SYSTEM_PROMPT = """\
You check generated software documentation against its source document. You \
receive document chunks, requirements extracted from them, improved rewrites of \
some requirements, and test cases written for the requirements.

Give one verdict for every item id:
- A requirement (REQ-...) must be stated or directly described by the chunks listed \
in its sources attribute.
- An improved rewrite (REQ-...:improved) may restructure and clarify its requirement \
and may contain [TBD: ...] placeholders, but must not add facts, numbers, conditions, \
actors or technologies that are not in the requirement or its chunks.
- A test case (TC-...) must verify the requirements it lists. Its steps and expected \
result may contain [TBD: ...] placeholders and ordinary test mechanics such as \
starting the system, sending a request or measuring time. A specific behaviour the \
system is expected to show, such as an error response, a message or a state, is an \
assumption unless the requirement or its chunks state it.

status: supported when nothing is unsupported; partially_supported when the item is \
grounded but contains at least one unsupported detail; unsupported when its main \
claim has no basis in the sources.
unsupported_claims: quote each unsupported detail briefly; empty when supported.
explanation: one sentence.\
"""

_STATUS_LABELS = {
    "supported": "i mbështetur",
    "partially_supported": "pjesërisht i mbështetur",
    "unsupported": "i pambështetur",
}


class Verdict(BaseModel):
    item_id: str
    status: Literal["supported", "partially_supported", "unsupported"]
    explanation: str
    unsupported_claims: list[str]


class VerificationOutput(BaseModel):
    verdicts: list[Verdict]


def verify_outputs(state: TechSpecState) -> dict[str, Any]:
    """Nyja e LangGraph: plotëson `verification_results` dhe `verification_passed`.

    Verifikimi kalon kur çdo element merr verdikt dhe asnjë nuk është "unsupported".
    Rezultati i shtohet edhe dokumentacionit. Gabimet e Gemini shënohen te `errors`.
    """
    claims = _collect_claims(state["requirements"], state["requirement_reviews"], state["test_cases"])
    if not claims:
        return {"verification_results": [], "verification_passed": True}

    chunks = state["retrieved_chunks"] or state["document_chunks"]
    try:
        output = invoke_structured(VerificationOutput, SYSTEM_PROMPT, _build_prompt(state, chunks))
    except LLMError as exc:
        return _failure(str(exc))

    results = _to_results(output.verdicts, claims)
    missing = [item_id for item_id in claims if item_id not in {result["item_id"] for result in results}]
    if missing:
        logger.warning("Gemini nuk dha verdikt për: %s", ", ".join(missing))
    passed = not missing and all(result["status"] != "unsupported" for result in results)

    update: dict[str, Any] = {"verification_results": results, "verification_passed": passed}
    if state["generated_documentation"]:
        update["generated_documentation"] = state["generated_documentation"] + render_verification(
            results, passed, missing
        )
    return update


def _collect_claims(
    requirements: Sequence[Requirement],
    reviews: Sequence[RequirementReview],
    test_cases: Sequence[TestCase],
) -> dict[str, tuple[str, list[str]]]:
    """Kthen {item_id: (teksti, burimet)} për çdo element që duhet verifikuar."""
    sources = {req["requirement_id"]: req["source_chunk_ids"] for req in requirements}
    by_requirement = {review["requirement_id"]: review for review in reviews}
    claims: dict[str, tuple[str, list[str]]] = {}
    for req in requirements:
        claims[req["requirement_id"]] = (req["description"], req["source_chunk_ids"])
        # Versioni i përmirësuar verifikohet vetëm kur kritiku ndryshoi diçka.
        review = by_requirement.get(req["requirement_id"])
        if review and review["issues"]:
            claims[req["requirement_id"] + IMPROVED_SUFFIX] = (review["improved_version"], req["source_chunk_ids"])
    for case in test_cases:
        case_sources = list(dict.fromkeys(cid for rid in case["requirement_ids"] for cid in sources.get(rid, [])))
        claims[case["test_id"]] = (f"{case['title']}: {case['expected_result']}", case_sources)
    return claims


def _build_prompt(state: Mapping[str, Any], chunks: Sequence[DocumentChunk]) -> str:
    requirements: Sequence[Requirement] = state["requirements"]
    reviews = {review["requirement_id"]: review for review in state["requirement_reviews"]}
    cited = {cid for req in requirements for cid in req["source_chunk_ids"]}

    parts = ["<chunks>"]
    parts += [
        f'<chunk id="{chunk["chunk_id"]}" page="{chunk["page"]}">\n{chunk["text"]}\n</chunk>'
        for chunk in chunks
        if chunk["chunk_id"] in cited
    ]
    parts += ["</chunks>", "", "<requirements>"]
    for req in requirements:
        parts.append(
            f'<requirement id="{req["requirement_id"]}" sources="{",".join(req["source_chunk_ids"])}">'
            f'{req["description"]}</requirement>'
        )
        review = reviews.get(req["requirement_id"])
        if review and review["issues"]:
            parts.append(
                f'<improved id="{req["requirement_id"]}{IMPROVED_SUFFIX}">{review["improved_version"]}</improved>'
            )
    parts += ["</requirements>", "", "<test_cases>"]
    for case in state["test_cases"]:
        lines = [f'<test_case id="{case["test_id"]}" requirements="{",".join(case["requirement_ids"])}">']
        lines.append(f"Title: {case['title']}")
        lines += [f"Precondition: {condition}" for condition in case["preconditions"]]
        lines += [f"Step {number}: {step}" for number, step in enumerate(case["steps"], start=1)]
        lines += [f"Expected result: {case['expected_result']}", "</test_case>"]
        parts.append("\n".join(lines))
    parts.append("</test_cases>")
    return "\n".join(parts)


def _to_results(verdicts: Sequence[Verdict], claims: Mapping[str, tuple[str, list[str]]]) -> list[VerificationResult]:
    by_id = {verdict.item_id: verdict for verdict in reversed(verdicts)}  # Ruhet verdikti i parë për çdo ID.
    unknown = set(by_id) - set(claims)
    if unknown:
        logger.warning("U hodhën verdikte për elemente që nuk ekzistojnë: %s", sorted(unknown))

    results: list[VerificationResult] = []
    for item_id, (claim, sources) in claims.items():
        verdict = by_id.get(item_id)
        if verdict is None:
            continue
        unsupported = [text for text in verdict.unsupported_claims if text.strip()]
        results.append(
            {
                "item_id": item_id,
                "claim": claim,
                # "supported" me supozime të listuara është kontradiktë; besohet lista konkrete.
                "status": "partially_supported" if verdict.status == "supported" and unsupported else verdict.status,
                "explanation": verdict.explanation,
                "unsupported_claims": unsupported,
                "source_chunk_ids": sources,
            }
        )
    return results


def render_verification(results: Sequence[VerificationResult], passed: bool, missing: Sequence[str]) -> str:
    """Seksioni Markdown që i shtohet dokumentacionit të gjeneruar."""
    counts = {status: sum(r["status"] == status for r in results) for status in _STATUS_LABELS}
    lines = [
        "",
        "## Verifikimi",
        "",
        f"Rezultati: **{'kaloi' if passed else 'nuk kaloi'}** · "
        + " · ".join(f"{label}: {counts[status]}" for status, label in _STATUS_LABELS.items()),
    ]
    if missing:
        lines += ["", f"Pa verdikt: {', '.join(missing)}"]
    flagged = [result for result in results if result["status"] != "supported"]
    if flagged:
        lines.append("")
        for result in flagged:
            lines.append(f"- **{result['item_id']}** ({_STATUS_LABELS[result['status']]}): {result['explanation']}")
            lines += [f"  - Supozim i pabazuar: {claim}" for claim in result["unsupported_claims"]]
    return "\n".join(lines) + "\n"


def _failure(message: str) -> dict[str, Any]:
    logger.warning(message)
    return {
        "verification_results": [],
        "verification_passed": False,
        "errors": [{"agent": AGENT_NAME, "message": message}],
    }
