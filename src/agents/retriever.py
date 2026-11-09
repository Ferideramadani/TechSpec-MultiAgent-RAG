"""Retrieval & Code Context Agent.

Gjen në bazën vektoriale pjesët e dokumentit që lidhen më shumë me pyetjen. Ky
agjent nuk thërret modelin gjuhësor.
"""

import logging
from pathlib import Path
from typing import Any

from src.state import TechSpecState
from src.utils.vector_db import VectorStore

logger = logging.getLogger(__name__)

AGENT_NAME = "retriever"


def retrieve_chunks(state: TechSpecState) -> dict[str, Any]:
    """Nyja e LangGraph: plotëson `retrieved_chunks` me pjesët më relevante.

    Kërkimi kufizohet te dokumenti aktual, sepse baza mund të mbajë edhe dokumente
    të ngarkuara më parë. Pyetja bosh dhe mungesa e rezultateve shënohen te
    `errors`, që agjentët pasardhës të mos punojnë pa kontekst. Gabimet e bazës
    vektoriale nuk kapen këtu, sepse tregojnë problem në sistem.
    """
    query = state["query"]
    if not query.strip():
        return _failure("Pyetja është bosh.")

    source = Path(state["document_path"]).name
    chunks = VectorStore().search(query, source=source)
    if not chunks:
        return _failure(f"Nuk u gjet asnjë pjesë për dokumentin: {source}")

    logger.info("%s: u gjetën %d pjesë.", source, len(chunks))
    return {"retrieved_chunks": chunks}


def _failure(message: str) -> dict[str, Any]:
    logger.warning(message)
    return {
        "retrieved_chunks": [],
        "errors": [{"agent": AGENT_NAME, "message": message}],
    }
