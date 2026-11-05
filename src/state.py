"""Gjendja globale që ndajnë 6 agjentët në grafin e LangGraph.

Çdo agjent lexon gjendjen dhe kthen vetëm fushat që ndryshon. Çdo rezultat i
nxjerrë nga dokumenti mban `source_chunk_ids`, që Citation & Fact-Verifier Agent
ta gjurmojë deri te pjesa origjinale e dokumentit.
"""

import operator
from typing import Annotated, Literal, TypedDict

Severity = Literal["low", "medium", "high", "critical"]
Priority = Literal["low", "medium", "high"]
VerificationStatus = Literal["supported", "partially_supported", "unsupported"]


class DocumentChunk(TypedDict):
    """Një pjesë e dokumentit, e krijuar nga parser-i."""

    chunk_id: str
    text: str
    source: str
    page: int | None
    section: str | None


class RetrievedChunk(DocumentChunk):
    """Një pjesë e dokumentit e gjetur nga kërkimi në bazën vektoriale."""

    score: float


class ApiEndpoint(TypedDict):
    method: str
    path: str
    description: str
    parameters: list[str]
    source_chunk_ids: list[str]


class FunctionalRequirement(TypedDict):
    requirement_id: str
    description: str
    priority: Priority
    source_chunk_ids: list[str]


class ArchitectureFinding(TypedDict):
    title: str
    severity: Severity
    description: str
    recommendation: str
    source_chunk_ids: list[str]


class TestCase(TypedDict):
    test_id: str
    title: str
    steps: list[str]
    expected_result: str
    requirement_ids: list[str]


class VerificationResult(TypedDict):
    claim: str
    status: VerificationStatus
    explanation: str
    source_chunk_ids: list[str]


class AgentError(TypedDict):
    agent: str
    message: str


class TechSpecState(TypedDict):
    # Hyrja nga përdoruesi
    document_path: str
    query: str

    # 1. Document Parser & Tech Spec Ingestion Agent
    document_chunks: list[DocumentChunk]

    # 2. Retrieval & Code Context Agent
    retrieved_chunks: list[RetrievedChunk]

    # 3. API & Functional Requirement Extractor Agent
    api_endpoints: list[ApiEndpoint]
    functional_requirements: list[FunctionalRequirement]

    # 4. Architecture Critic & Anti-Pattern Analyzer
    architecture_findings: list[ArchitectureFinding]

    # 5. Test Case & Documentation Generator Agent
    test_cases: list[TestCase]
    generated_documentation: str

    # 6. Citation & Fact-Verifier Agent
    verification_results: list[VerificationResult]
    verification_passed: bool
    revision_count: int

    # Reducer-i `operator.add` shton gabimet e reja te lista, pa i mbishkruar të vjetrat.
    errors: Annotated[list[AgentError], operator.add]


def create_initial_state(document_path: str, query: str) -> TechSpecState:
    """Krijon gjendjen fillestare për një ekzekutim të ri të grafit."""
    return {
        "document_path": document_path,
        "query": query,
        "document_chunks": [],
        "retrieved_chunks": [],
        "api_endpoints": [],
        "functional_requirements": [],
        "architecture_findings": [],
        "test_cases": [],
        "generated_documentation": "",
        "verification_results": [],
        "verification_passed": False,
        "revision_count": 0,
        "errors": [],
    }
