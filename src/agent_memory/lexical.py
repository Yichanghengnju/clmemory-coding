"""Deterministic, user-scoped Okapi BM25 ranking without external indexes."""

import math
import re
from collections import Counter
from dataclasses import replace

from .domain import Evidence

_WORDS = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+|[^\W_]+", re.UNICODE)
_CJK = re.compile(r"^[\u3400-\u4dbf\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for word in _WORDS.findall(text.casefold()):
        if _CJK.fullmatch(word):
            # Unigrams cover single-character queries; bigrams retain phrase signal.
            tokens.extend(word)
            tokens.extend(word[i : i + 2] for i in range(len(word) - 1))
        else:
            tokens.append(word)
    return tokens


def rank(
    documents: list[tuple[Evidence, str]], query: str, top_k: int
) -> list[Evidence]:
    """Score only the supplied corpus with BM25 (k1=1.5, b=0.75).

    The caller supplies one user's enabled evidence kinds, so other users and
    disabled kinds cannot change document frequencies or length normalization.
    """
    if not documents:
        return []
    terms = set(tokenize(query))
    if not terms:
        return []
    counts = [Counter(tokenize(text)) for _, text in documents]
    lengths = [sum(count.values()) for count in counts]
    average_length = sum(lengths) / len(lengths)
    if not average_length:
        return []
    document_frequency = Counter(
        term for count in counts for term in terms if term in count
    )
    results: list[Evidence] = []
    for (evidence, _), count, length in zip(documents, counts, lengths):
        score = 0.0
        # Sorted terms make floating-point accumulation reproducible.
        for term in sorted(terms):
            frequency = count[term]
            if not frequency:
                continue
            df = document_frequency[term]
            idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * length / average_length)
            score += idf * frequency * 2.5 / denominator
        if score > 0:
            results.append(replace(evidence, score=score))
    results.sort(key=lambda item: (-item.score, item.id))
    return results[:top_k]
