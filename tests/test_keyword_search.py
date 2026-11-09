import pytest

from src.utils.keyword_search import RRF_K, bm25_scores, reciprocal_rank_fusion, tokenize


def test_tokenize_keeps_technical_terms_and_drops_stopwords():
    assert tokenize("What is the rate limit of POST /api/v1/orders in OAuth2 v1.2?") == [
        "rate", "limit", "post", "api/v1/orders", "oauth2", "v1.2",
    ]


def test_documents_with_query_terms_score_higher():
    documents = ["The cache stores sessions.", "Requests above the rate limit get HTTP 429.", "Nothing here."]

    scores = bm25_scores("rate limit 429", documents)

    assert scores[1] > 0
    assert scores[0] == scores[2] == 0


def test_rare_terms_weigh_more_than_common_ones():
    documents = ["api gateway", "api catalog", "api monitoring", "api token rotation"]

    scores = bm25_scores("api rotation", documents)

    assert scores.index(max(scores)) == 3


def test_shorter_documents_win_on_equal_term_frequency():
    scores = bm25_scores("token", ["token", "token " + "filler " * 50])

    assert scores[0] > scores[1] > 0


@pytest.mark.parametrize(("query", "documents"), [("", ["text"]), ("the of and", ["text"]), ("token", [])])
def test_empty_or_stopword_queries_score_zero(query, documents):
    assert bm25_scores(query, documents) == [0.0] * len(documents)


def test_rank_fusion_rewards_items_ranked_well_by_both_methods():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b"]])

    assert fused["b"] == pytest.approx(1 / (RRF_K + 2) * 2)
    assert sorted(fused, key=fused.get, reverse=True) == ["c", "b", "a"]
