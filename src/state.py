"""Gjendja globale që ndajnë 6 agjentët në grafin e LangGraph.

Çdo agjent lexon gjendjen dhe kthen vetëm fushat që ndryshon. Çdo rezultat i
nxjerrë nga dokumenti mban `source_chunk_ids`, që Citation & Fact-Verifier Agent
ta gjurmojë deri te pjesa origjinale e dokumentit.
"""

import operator
from typing import Annotated, Literal, TypedDict

Priority = Literal["low", "medium", "high"]
RequirementKind = Literal["functional", "non_functional"]
QualityCriterion = Literal["clarity", "ambiguity", "testability", "measurability", "completeness"]
TestType = Literal["positive", "negative", "boundary"]
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


class Requirement(TypedDict):
    requirement_id: str
    kind: RequirementKind
    description: str
    priority: Priority
    source_chunk_ids: list[str]


class QualityIssue(TypedDict):
    criterion: QualityCriterion
    problem: str


class RequirementReview(TypedDict):
    requirement_id: str
    score: int  # 0 deri në 100: pjesa e kritereve të cilësisë që plotëson kërkesa.
    issues: list[QualityIssue]
    improved_version: str


class TestCase(TypedDict):
    test_id: str
    title: str
    test_type: TestType
    preconditions: list[str]
    steps: list[str]
    expected_result: str
    requirement_ids: list[str]


class VerificationResult(TypedDict):
    item_id: str  # REQ-001, REQ-001:improved ose TC-001.
    claim: str  # Teksti i verifikuar, marrë nga gjendja dhe jo nga modeli.
    status: VerificationStatus
    explanation: str
    unsupported_claims: list[str]
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
    page_count: int  # Vetëm faqet me tekst; faqet bosh ose të palexueshme nuk numërohen.
    chunk_count: int

    # 2. Retrieval & Code Context Agent
    retrieved_chunks: list[RetrievedChunk]

    # 3. API & Functional Requirement Extractor Agent
    api_endpoints: list[ApiEndpoint]
    requirements: list[Requirement]

    # 4. Requirement Quality Critic
    requirement_reviews: list[RequirementReview]
    quality_score: int | None  # Mesatarja e vlerësimeve; None kur nuk ka kërkesa të vlerësuara.
    quality_summary: str

    # 5. Test Case & Documentation Generator Agent
    test_cases: list[TestCase]
    generated_documentation: str  # Markdown, i ndërtuar nga kodi pa thirrje te Gemini.

    # 6. Citation & Fact-Verifier Agent
    verification_results: list[VerificationResult]
    verification_passed: bool  # True kur çdo element u verifikua dhe asnjë nuk është "unsupported".

    # Reducer-i `operator.add` shton gabimet e reja te lista, pa i mbishkruar të vjetrat.
    errors: Annotated[list[AgentError], operator.add]


def create_initial_state(document_path: str, query: str) -> TechSpecState:
    """Krijon gjendjen fillestare për një ekzekutim të ri të grafit."""
    return {
        "document_path": document_path,
        "query": query,
        "document_chunks": [],
        "page_count": 0,
        "chunk_count": 0,
        "retrieved_chunks": [],
        "api_endpoints": [],
        "requirements": [],
        "requirement_reviews": [],
        "quality_score": None,
        "quality_summary": "",
        "test_cases": [],
        "generated_documentation": "",
        "verification_results": [],
        "verification_passed": False,
        "errors": [],
    }
