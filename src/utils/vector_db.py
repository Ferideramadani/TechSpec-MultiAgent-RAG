"""Baza vektoriale lokale me ChromaDB për ruajtjen dhe kërkimin e pjesëve të dokumenteve."""

import os
from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import Any

import chromadb

from src.config import PROJECT_ROOT
from src.state import DocumentChunk, RetrievedChunk
from src.utils.keyword_search import bm25_scores, reciprocal_rank_fusion

DEFAULT_PERSIST_DIR = Path(os.getenv("TECHSPEC_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
DEFAULT_COLLECTION = "tech_specs"
DEFAULT_TOP_K = 5
# Kufiri i pjesëve që rirenditen nga BM25; mjafton për dokumente me qindra faqe.
MAX_CANDIDATES = 500


class VectorStore:
    """Menaxhon një koleksion ChromaDB që ruhet në disk.

    Kur `embedding_function` mungon, Chroma përdor modelin e vet lokal
    all-MiniLM-L6-v2, i cili shkarkohet automatikisht herën e parë që përdoret.
    """

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_function: Any | None = None,
    ) -> None:
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        options: dict[str, Any] = {"configuration": {"hnsw": {"space": "cosine"}}}
        if embedding_function is not None:
            options["embedding_function"] = embedding_function
        self._collection = self._client.get_or_create_collection(collection_name, **options)

    def add_chunks(self, chunks: Sequence[DocumentChunk]) -> int:
        """Shton pjesët dhe zëvendëson versionin e mëparshëm të të njëjtit dokument.

        Pjesët e vjetra fshihen vetëm pasi të jenë ruajtur të rejat, që një gabim
        gjatë embedding-ut të mos e lërë dokumentin jashtë bazës. Kthen numrin e
        pjesëve të ruajtura.
        """
        new_ids = {chunk["chunk_id"] for chunk in chunks}
        stale_ids: list[str] = []
        for source in {chunk["source"] for chunk in chunks}:
            existing = self._collection.get(where={"source": source}, include=[])["ids"]
            stale_ids.extend(chunk_id for chunk_id in existing if chunk_id not in new_ids)

        for batch in batched(chunks, self._client.get_max_batch_size()):
            self._collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch],
                documents=[chunk["text"] for chunk in batch],
                metadatas=[_to_metadata(chunk) for chunk in batch],
            )
        if stale_ids:
            self._collection.delete(ids=stale_ids)
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source: str | None = None,
        hybrid: bool = True,
    ) -> list[RetrievedChunk]:
        """Kthen pjesët më relevante për pyetjen, nga më e mira te më e dobëta.

        Me `hybrid`, renditja semantike bashkohet me BM25 (fjalë kyçe) përmes RRF,
        që të gjenden edhe termat e saktë si emrat e API-ve ose kodet e gabimeve.
        `score` mbetet ngjashmëria kosinus dhe `keyword_score` pikët BM25 (0 kur asnjë
        fjalë e pyetjes nuk gjendet në pjesë). Me `source` kërkimi kufizohet te një dokument.
        """
        if not query.strip():
            raise ValueError("Pyetja nuk mund të jetë bosh.")
        if top_k < 1:
            raise ValueError("top_k duhet të jetë të paktën 1.")

        where = {"source": source} if source else None
        available = len(self._collection.get(where=where, include=[])["ids"])
        if available == 0:
            return []

        # Kandidatët renditen semantikisht; BM25 i rirendit vetëm këta, që kërkimi të mbetet i shpejtë.
        result = self._collection.query(
            query_texts=[query],
            n_results=min(available, MAX_CANDIDATES) if hybrid else min(available, top_k),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids, documents = result["ids"][0], result["documents"][0]
        keyword_scores = bm25_scores(query, documents) if hybrid else [0.0] * len(ids)

        order = list(range(len(ids)))
        if hybrid:
            keyword_ranking = sorted((i for i in order if keyword_scores[i] > 0), key=lambda i: -keyword_scores[i])
            fused = reciprocal_rank_fusion([order, keyword_ranking])
            order.sort(key=lambda i: -fused[i])  # Renditja e qëndrueshme ruan rendin semantik te barazimet.

        return [
            {
                "chunk_id": ids[i],
                "text": documents[i],
                "source": result["metadatas"][0][i]["source"],
                "page": result["metadatas"][0][i].get("page"),
                "section": result["metadatas"][0][i].get("section"),
                "score": 1.0 - result["distances"][0][i],
                "keyword_score": keyword_scores[i],
            }
            for i in order[:top_k]
        ]

    def delete_source(self, source: str) -> None:
        """Fshin të gjitha pjesët e një dokumenti."""
        self._collection.delete(where={"source": source})

    def count(self) -> int:
        """Kthen numrin e pjesëve të ruajtura në koleksion."""
        return self._collection.count()


def _to_metadata(chunk: DocumentChunk) -> dict[str, str | int]:
    # Metadatat e Chroma nuk pranojnë vlera None, ndaj fushat bosh nuk ruhen.
    fields = {"source": chunk["source"], "page": chunk["page"], "section": chunk["section"]}
    return {key: value for key, value in fields.items() if value is not None}
