"""Document Parser & Tech Spec Ingestion Agent.

Lexon PDF-në, e ndan në pjesë dhe i ruan në bazën vektoriale. Ky agjent nuk
thërret modelin gjuhësor.
"""

import logging
from typing import Any

from src.state import TechSpecState
from src.utils.pdf_loader import PDFLoadError, load_and_chunk_pdf
from src.utils.vector_db import VectorStore

logger = logging.getLogger(__name__)

AGENT_NAME = "document_parser"


def parse_document(state: TechSpecState) -> dict[str, Any]:
    """Nyja e LangGraph: kthen vetëm fushat e gjendjes që ndryshon ky agjent.

    Një PDF e palexueshme shënohet te `errors`, që grafi ta ndalojë ekzekutimin
    dhe aplikacioni t'ia shfaqë mesazhin përdoruesit. Gabimet e tjera, si dështimi
    i bazës vektoriale, nuk kapen këtu, sepse tregojnë problem në sistem dhe jo në
    dokument.
    """
    document_path = state["document_path"]

    try:
        chunks = load_and_chunk_pdf(document_path)
    except PDFLoadError as exc:
        logger.warning("Dokumenti nuk u përpunua: %s", exc)
        return {"errors": [{"agent": AGENT_NAME, "message": str(exc)}]}

    VectorStore().add_chunks(chunks)

    page_count = len({chunk["page"] for chunk in chunks})
    logger.info("%s: %d faqe, %d pjesë.", document_path, page_count, len(chunks))
    return {
        "document_chunks": chunks,
        "page_count": page_count,
        "chunk_count": len(chunks),
    }
