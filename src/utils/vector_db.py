"""Baza vektoriale lokale me ChromaDB për ruajtjen dhe kërkimin e pjesëve të dokumenteve."""

import os
from collections.abc import Sequence
from itertools import batched
from pathlib import Path
from typing import Any

import chromadb

from src.config import PROJECT_ROOT
from src.state import DocumentChunk, RetrievedChunk

DEFAULT_PERSIST_DIR = Path(os.getenv("TECHSPEC_CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
DEFAULT_COLLECTION = "tech_specs"
DEFAULT_TOP_K = 5


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
    ) -> list[RetrievedChunk]:
        """Kthen pjesët më të ngjashme, nga më e përafërta te më e largëta.

        `score` është ngjashmëria kosinus: sa më e lartë, aq më e afërt me pyetjen.
        Me `source` kërkimi kufizohet te një dokument i vetëm.
        """
        if not query.strip():
            raise ValueError("Pyetja nuk mund të jetë bosh.")
        if top_k < 1:
            raise ValueError("top_k duhet të jetë të paktën 1.")
        if self.count() == 0:
            return []

        result = self._collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"source": source} if source else None,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": metadata["source"],
                "page": metadata.get("page"),
                "section": metadata.get("section"),
                "score": 1.0 - distance,
            }
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
            )
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
