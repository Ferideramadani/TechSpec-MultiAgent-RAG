import hashlib
import math
import re

import pytest
from chromadb.api.types import EmbeddingFunction

from src.utils.vector_db import VectorStore


class HashEmbedding(EmbeddingFunction):
    """Embedding i thjeshtë, pa shkarkim modeli: tekste me fjalë të përbashkëta afrohen."""

    def __init__(self) -> None:
        pass

    def __call__(self, input):
        return [self._embed(text) for text in input]

    @staticmethod
    def name() -> str:
        return "hash-test"

    @staticmethod
    def build_from_config(config):
        return HashEmbedding()

    def get_config(self):
        return {}

    @staticmethod
    def _embed(text: str) -> list[float]:
        vector = [0.0] * 128
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 128] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def make_chunk(chunk_id, text, source="spec.pdf", page=1):
    return {"chunk_id": chunk_id, "text": text, "source": source, "page": page, "section": None}


AUTH = make_chunk("spec-c0", "OAuth token authentication for the login endpoint", page=3)
CACHE = make_chunk("spec-c1", "Redis cache eviction policy and memory limits", page=7)


@pytest.fixture
def store(tmp_path):
    return VectorStore(tmp_path / "db", embedding_function=HashEmbedding())


def test_search_returns_most_similar_chunk_first_with_all_fields(store):
    store.add_chunks([AUTH, CACHE])

    results = store.search("login token authentication", top_k=2)

    assert [result["chunk_id"] for result in results] == ["spec-c0", "spec-c1"]
    assert results[0] == {
        **AUTH,
        "score": pytest.approx(results[0]["score"]),
        "keyword_score": pytest.approx(results[0]["keyword_score"]),
    }
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["keyword_score"] > 0
    assert results[1]["keyword_score"] == 0


class MisleadingEmbedding(HashEmbedding):
    """Embedding që e vendos qëllimisht pjesën e duhur të fundit në renditjen semantike."""

    def __call__(self, input):
        return [[0.0, 1.0] + [0.0] * 126 if "Errors" in text else [1.0] + [0.0] * 127 for text in input]

    @staticmethod
    def name() -> str:
        return "misleading-test"


def test_hybrid_search_finds_exact_terms_that_semantic_search_misses(tmp_path):
    store = VectorStore(tmp_path / "db", embedding_function=MisleadingEmbedding())
    filler = [make_chunk(f"spec-c{i}", f"General overview paragraph number {i}.") for i in range(8)]
    target = make_chunk("spec-target", "Errors return HTTP 429 from POST /api/v1/orders when limits are hit.")
    store.add_chunks([*filler, target])

    semantic = store.search("/api/v1/orders 429", top_k=3, hybrid=False)
    hybrid = store.search("/api/v1/orders 429", top_k=3)

    assert "spec-target" not in [r["chunk_id"] for r in semantic]
    assert hybrid[0]["chunk_id"] == "spec-target"
    assert hybrid[0]["keyword_score"] > 0


def test_semantic_only_search_is_sorted_by_similarity(store):
    store.add_chunks([AUTH, CACHE])

    results = store.search("redis memory", top_k=2, hybrid=False)

    assert [r["chunk_id"] for r in results] == ["spec-c1", "spec-c0"]
    assert all(r["keyword_score"] == 0 for r in results)


def test_hybrid_search_respects_source_and_top_k(store):
    other = make_chunk("other-c0", "OAuth token authentication", source="other.pdf")
    store.add_chunks([AUTH, CACHE, other])

    results = store.search("OAuth token", top_k=1, source="spec.pdf")

    assert [r["chunk_id"] for r in results] == ["spec-c0"]


def test_score_is_cosine_similarity(store):
    store.add_chunks([AUTH])

    assert store.search(AUTH["text"])[0]["score"] == pytest.approx(1.0, abs=1e-3)


def test_search_can_be_limited_to_one_source(store):
    other = make_chunk("other-c0", "OAuth token authentication details", source="other.pdf")
    store.add_chunks([AUTH, other])

    results = store.search("OAuth token", source="other.pdf")

    assert [result["chunk_id"] for result in results] == ["other-c0"]


def test_search_handles_empty_store_and_top_k_above_count(store):
    assert store.search("anything") == []

    store.add_chunks([AUTH])

    assert len(store.search("anything", top_k=50)) == 1


def test_search_with_unknown_source_returns_nothing(store):
    store.add_chunks([AUTH])

    assert store.search("OAuth", source="missing.pdf") == []


@pytest.mark.parametrize("query", ["", "   ", "\n"])
def test_search_rejects_blank_query(store, query):
    with pytest.raises(ValueError):
        store.search(query)


def test_search_rejects_invalid_top_k(store):
    with pytest.raises(ValueError):
        store.search("query", top_k=0)


def test_reingesting_a_document_removes_its_old_chunks(store):
    other = make_chunk("other-c0", "Unrelated document", source="other.pdf")
    store.add_chunks([AUTH, CACHE, other])

    updated = make_chunk("spec-c0", "Updated authentication text")
    store.add_chunks([updated])

    assert store.count() == 2
    assert {result["chunk_id"] for result in store.search("text", top_k=5)} == {
        "spec-c0",
        "other-c0",
    }
    assert store.search("Updated authentication", source="spec.pdf")[0]["text"] == updated["text"]


def test_failed_embedding_keeps_the_previous_version(tmp_path):
    class FailOnNewVersion(HashEmbedding):
        def __call__(self, input):
            if "New version" in input:
                raise RuntimeError("embedding failed")
            return super().__call__(input)

    store = VectorStore(tmp_path / "db", embedding_function=FailOnNewVersion())
    store.add_chunks([AUTH, CACHE])

    with pytest.raises(RuntimeError, match="embedding failed"):
        store.add_chunks([make_chunk("spec-new", "New version")])

    assert store.count() == 2
    assert store.search("OAuth token")[0]["chunk_id"] == "spec-c0"


def test_add_chunks_with_no_chunks_is_a_noop(store):
    assert store.add_chunks([]) == 0
    assert store.count() == 0


def test_delete_source_removes_only_that_document(store):
    other = make_chunk("other-c0", "Other", source="other.pdf")
    store.add_chunks([AUTH, CACHE, other])

    store.delete_source("spec.pdf")

    assert store.count() == 1


def test_data_persists_after_reopening(tmp_path):
    VectorStore(tmp_path / "db", embedding_function=HashEmbedding()).add_chunks([AUTH])

    reopened = VectorStore(tmp_path / "db", embedding_function=HashEmbedding())

    assert reopened.count() == 1
    assert reopened.search("OAuth")[0]["chunk_id"] == "spec-c0"
