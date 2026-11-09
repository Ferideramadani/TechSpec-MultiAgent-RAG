"""Kërkimi me fjalë kyçe (BM25) dhe bashkimi i tij me kërkimin semantik.

Kërkimi semantik gjen tekste me kuptim të ngjashëm, por shpesh humbet termat e
saktë: emra API-sh, kode gabimi, numra versionesh. BM25 i gjen pikërisht këto.
Dy renditjet bashkohen me Reciprocal Rank Fusion (RRF), që nuk kërkon që
pikët e dy metodave të jenë në të njëjtën shkallë.
"""

import math
import re
from collections import Counter
from collections.abc import Iterable, Sequence

# Ruhen bashkë termat teknikë si "/api/v1/orders", "oauth2" ose "v1.2".
_TOKEN = re.compile(r"\w+(?:[./-]\w+)*")
_STOPWORDS = frozenset(
    "a an and are as at be by can do does for from has have how in is it its of on or "
    "that the their this to was were what when where which who why will with".split()
)

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


def bm25_scores(query: str, documents: Sequence[str]) -> list[float]:
    """Pikët Okapi BM25 të çdo dokumenti për pyetjen; 0 kur nuk ka asnjë term të përbashkët."""
    query_terms = set(tokenize(query))
    tokenized = [tokenize(document) for document in documents]
    if not query_terms or not tokenized:
        return [0.0] * len(documents)

    average_length = sum(map(len, tokenized)) / len(tokenized) or 1.0
    document_frequency = Counter(term for tokens in tokenized for term in set(tokens) & query_terms)
    count = len(tokenized)

    scores = []
    for tokens in tokenized:
        frequencies = Counter(tokens)
        length_norm = BM25_K1 * (1 - BM25_B + BM25_B * len(tokens) / average_length)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency:
                idf = math.log((count - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5) + 1)
                score += idf * frequency * (BM25_K1 + 1) / (frequency + length_norm)
        scores.append(score)
    return scores


def reciprocal_rank_fusion(rankings: Iterable[Sequence[str]]) -> dict[str, float]:
    """Bashkon disa renditje: çdo element merr 1 / (RRF_K + pozicioni) nga secila renditje ku shfaqet."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, item in enumerate(ranking, start=1):
            fused[item] = fused.get(item, 0.0) + 1 / (RRF_K + position)
    return fused
