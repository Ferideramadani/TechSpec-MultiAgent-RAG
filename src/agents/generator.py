"""Test Case & Documentation Generator Agent.

Gemini shkruan rastet e testimit për kërkesat. Dokumentacioni Markdown ndërtohet
nga kodi me të dhënat e gjendjes, ndaj nuk përmban asgjë të shpikur dhe nuk
kërkon thirrje shtesë te API-ja.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from src.state import (
    DocumentChunk,
    Requirement,
    RequirementReview,
    TechSpecState,
    TestCase,
)
from src.utils.llm import LLMError, invoke_structured

logger = logging.getLogger(__name__)

AGENT_NAME = "generator"

SYSTEM_PROMPT = """\
You write test cases for software requirements. Each requirement is in a \
<requirement> tag with its original statement, extracted from a document. When a \
reviewer has assessed it, an improved version and the quality issues found follow.

Write test cases that verify the original requirement. Use the improved version \
only to make vague steps concrete. Where a value is unknown, keep a placeholder such \
as [TBD: maximum response time in ms] in the step or the expected result; never \
invent numbers, endpoints, names or technologies.

For each requirement, write one positive test case, and add a negative or boundary \
test case when the requirement has a condition, a limit or a failure mode to check. \
Write at most three test cases per requirement.

For each test case:
- title: what is verified, in a few words.
- test_type: positive, negative or boundary.
- preconditions: the state the system must be in before the test; may be empty.
- steps: imperative actions, one action per step.
- expected_result: an observable outcome that decides pass or fail.
- requirement_ids: the ids of the requirements the test verifies.\
"""


class GeneratedTestCase(BaseModel):
    title: str
    test_type: Literal["positive", "negative", "boundary"]
    preconditions: list[str]
    steps: list[str]
    expected_result: str
    requirement_ids: list[str]


class GenerationResult(BaseModel):
    test_cases: list[GeneratedTestCase]


def generate_tests_and_docs(state: TechSpecState) -> dict[str, Any]:
    """Nyja e LangGraph: plotëson `test_cases` dhe `generated_documentation`.

    Pa kërkesa nuk ka çfarë të testohet, por dokumentacioni ndërtohet sërish që
    përdoruesi të shohë se nuk u gjet asnjë kërkesë. Gabimet e Gemini shënohen te
    `errors` në vend që të ngrihen.
    """
    requirements = state["requirements"]
    test_cases: list[TestCase] = []
    if requirements:
        reviews = {review["requirement_id"]: review for review in state["requirement_reviews"]}
        try:
            result = invoke_structured(GenerationResult, SYSTEM_PROMPT, _build_prompt(requirements, reviews))
        except LLMError as exc:
            return _failure(str(exc))

        test_cases = _to_test_cases(result.test_cases, requirements)
        if not test_cases:
            return _failure("Gemini nuk ktheu asnjë rast testimi të vlefshëm.")

    return {
        "test_cases": test_cases,
        "generated_documentation": render_documentation({**state, "test_cases": test_cases}),
    }


def _build_prompt(requirements: Sequence[Requirement], reviews: Mapping[str, RequirementReview]) -> str:
    blocks = []
    for req in requirements:
        lines = [
            f'<requirement id="{req["requirement_id"]}" kind="{req["kind"]}" priority="{req["priority"]}">',
            f"<original>{req['description']}</original>",
        ]
        review = reviews.get(req["requirement_id"])
        if review:
            lines.append(f"<improved>{review['improved_version']}</improved>")
            lines.extend(f'<issue criterion="{issue["criterion"]}">{issue["problem"]}</issue>' for issue in review["issues"])
        lines.append("</requirement>")
        blocks.append("\n".join(lines))
    return "<requirements>\n" + "\n".join(blocks) + "\n</requirements>"


def _to_test_cases(items: Sequence[GeneratedTestCase], requirements: Sequence[Requirement]) -> list[TestCase]:
    valid_ids = {req["requirement_id"] for req in requirements}
    test_cases: list[TestCase] = []
    for item in items:
        # Një test pa kërkesë ekzistuese ose pa hapa nuk mund të gjurmohet as të ekzekutohet.
        requirement_ids = [rid for rid in dict.fromkeys(item.requirement_ids) if rid in valid_ids]
        steps = [step for step in item.steps if step.strip()]
        if not requirement_ids or not steps:
            logger.warning("U hoq një rast testimi pa kërkesë të vlefshme ose pa hapa: %s", item.title)
            continue
        test_cases.append(
            {
                "test_id": f"TC-{len(test_cases) + 1:03d}",
                "title": item.title,
                "test_type": item.test_type,
                "preconditions": [condition for condition in item.preconditions if condition.strip()],
                "steps": steps,
                "expected_result": item.expected_result,
                "requirement_ids": requirement_ids,
            }
        )

    covered = {rid for case in test_cases for rid in case["requirement_ids"]}
    uncovered = [req["requirement_id"] for req in requirements if req["requirement_id"] not in covered]
    if uncovered:
        logger.warning("Kërkesa pa asnjë rast testimi: %s", ", ".join(uncovered))
    return test_cases


def render_documentation(state: Mapping[str, Any]) -> str:
    """Ndërton dokumentacionin Markdown nga gjendja, pa thirrje te Gemini."""
    requirements: Sequence[Requirement] = state["requirements"]
    reviews = {review["requirement_id"]: review for review in state["requirement_reviews"]}
    test_cases: Sequence[TestCase] = state["test_cases"]
    pages = _pages_by_chunk(state["retrieved_chunks"] or state["document_chunks"])

    lines = [
        f"# Specifikimi teknik: {Path(state['document_path']).name}",
        "",
        f"Pyetja e analizës: {state['query']}",
        "",
        "## Kërkesat",
        "",
    ]
    if requirements:
        lines += ["| ID | Lloji | Prioriteti | Cilësia | Kërkesa | Faqet |", "|---|---|---|---|---|---|"]
        for req in requirements:
            review = reviews.get(req["requirement_id"])
            quality = f"{review['score']}%" if review else "pa vlerësim"
            source_pages = sorted({pages[cid] for cid in req["source_chunk_ids"] if pages.get(cid) is not None})
            lines.append(
                f"| {req['requirement_id']} | {_KIND_LABELS[req['kind']]} | {req['priority']} | {quality} "
                f"| {_cell(req['description'])} | {', '.join(map(str, source_pages)) or '-'} |"
            )
    else:
        lines.append("Nuk u gjet asnjë kërkesë në pjesët e analizuara të dokumentit.")

    if state["api_endpoints"]:
        lines += ["", "## Pikat API", "", "| Metoda | Rruga | Përshkrimi | Parametrat |", "|---|---|---|---|"]
        for endpoint in state["api_endpoints"]:
            lines.append(
                f"| {endpoint['method']} | `{_cell(endpoint['path'])}` | {_cell(endpoint['description'])} "
                f"| {_cell(', '.join(endpoint['parameters'])) or '-'} |"
            )

    if state["quality_score"] is not None:
        lines += ["", "## Cilësia e kërkesave", "", f"Vlerësimi i përgjithshëm: **{state['quality_score']}%**", ""]
        if state["quality_summary"]:
            lines += [state["quality_summary"], ""]
        for req in requirements:
            review = reviews.get(req["requirement_id"])
            if review and review["issues"]:
                lines.append(f"- **{req['requirement_id']}**: {review['improved_version']}")

    if test_cases:
        lines += ["", "## Rastet e testimit"]
        for case in test_cases:
            lines += [
                "",
                f"### {case['test_id']}: {case['title']}",
                "",
                f"Lloji: {_TYPE_LABELS[case['test_type']]} · Verifikon: {', '.join(case['requirement_ids'])}",
            ]
            if case["preconditions"]:
                lines += ["", "Parakushtet:"] + [f"- {condition}" for condition in case["preconditions"]]
            lines += ["", "Hapat:"] + [f"{number}. {step}" for number, step in enumerate(case["steps"], start=1)]
            lines += ["", f"Rezultati i pritur: {case['expected_result']}"]

    if requirements:
        lines += ["", "## Matrica e gjurmueshmërisë", "", "| Kërkesa | Rastet e testimit |", "|---|---|"]
        for req in requirements:
            tests = [case["test_id"] for case in test_cases if req["requirement_id"] in case["requirement_ids"]]
            lines.append(f"| {req['requirement_id']} | {', '.join(tests) or 'pa test'} |")

    return "\n".join(lines) + "\n"


_KIND_LABELS = {"functional": "funksionale", "non_functional": "jofunksionale"}
_TYPE_LABELS = {"positive": "pozitiv", "negative": "negativ", "boundary": "kufitar"}


def _pages_by_chunk(chunks: Sequence[DocumentChunk]) -> dict[str, int | None]:
    return {chunk["chunk_id"]: chunk["page"] for chunk in chunks}


def _cell(text: str) -> str:
    """Teksti brenda një qelize tabele Markdown nuk duhet të ketë '|' ose rresht të ri."""
    return " ".join(text.split()).replace("|", "\\|")


def _failure(message: str) -> dict[str, Any]:
    logger.warning(message)
    return {
        "test_cases": [],
        "generated_documentation": "",
        "errors": [{"agent": AGENT_NAME, "message": message}],
    }
