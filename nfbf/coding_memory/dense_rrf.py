"""Optional dense + BM25 RRF fusion for Coding Memory.

Default path stays BM25-only. When enable_dense_rrf is on, BM25 ranks and
dense cosine ranks are fused with classic RRF. Zero generative LLM.
"""
from __future__ import annotations

import math
from typing import Callable, Mapping, MutableMapping, Sequence


DEFAULT_RRF_K = 60


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("vector length mismatch")
    dot = math.fsum(x * y for x, y in zip(a, b))
    na = math.sqrt(math.fsum(x * x for x in a))
    nb = math.sqrt(math.fsum(y * y for y in b))
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (na * nb)


def ranks_from_scores(ids: Sequence[str], scores: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(ids, key=lambda i: (-scores.get(i, 0.0), i))
    return {i: r + 1 for r, i in enumerate(ordered)}


def rrf_fuse(
    rank_lists: Sequence[Mapping[str, int]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    if k < 1:
        raise ValueError("rrf_k must be >= 1")
    if weights is None:
        weights = [1.0] * len(rank_lists)
    if len(weights) != len(rank_lists):
        raise ValueError("weights length must match rank_lists")
    fused: dict[str, float] = {}
    for weight, ranks in zip(weights, rank_lists):
        for doc_id, rank in ranks.items():
            fused[doc_id] = fused.get(doc_id, 0.0) + float(weight) / (k + int(rank))
    return fused


def dense_scores_for_chunks(
    query_vec: Sequence[float],
    chunk_texts: Mapping[str, str],
    embed_documents: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    *,
    vector_cache: MutableMapping[str, Sequence[float]] | None = None,
) -> dict[str, float]:
    """Embed missing chunk texts and return cosine scores vs query."""
    cache = vector_cache if vector_cache is not None else {}
    missing_ids = [cid for cid, text in chunk_texts.items() if cid not in cache]
    if missing_ids:
        texts = [chunk_texts[cid] for cid in missing_ids]
        vectors = list(embed_documents(texts))
        if len(vectors) != len(missing_ids):
            raise ValueError("embed_documents returned wrong count")
        for cid, vec in zip(missing_ids, vectors):
            cache[cid] = vec
    return {cid: cosine(query_vec, cache[cid]) for cid in chunk_texts}


def fuse_bm25_dense(
    bm25_ranked_ids: Sequence[str],
    dense_score_by_id: Mapping[str, float],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Return (id, rrf_score) sorted desc.

    Ids absent from dense still receive BM25 RRF contribution only.
    """
    bm25_ranks = {doc_id: i + 1 for i, doc_id in enumerate(bm25_ranked_ids)}
    all_ids = list(dict.fromkeys([*bm25_ranked_ids, *dense_score_by_id.keys()]))
    dense_ranks = ranks_from_scores(all_ids, dense_score_by_id)
    fused = rrf_fuse(
        [bm25_ranks, dense_ranks],
        k=rrf_k,
        weights=[bm25_weight, dense_weight],
    )
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class DenseRRFNotAvailable(RuntimeError):
    """Raised when dense RRF is requested but credentials/embedder are missing."""


def require_dense_enabled(config: Mapping) -> bool:
    return bool(config.get("enable_dense_rrf", False))


def dense_applies_for_profile(config: Mapping, profile: str) -> bool:
    """Whether dense RRF should run for this resolved search profile.

    Default apply list is nf+blend (BF skip): A/B showed dense helps NF and
    hurts BF hit@k. Override with config dense_rrf_apply_profiles (list/tuple
    of profile names). Empty list means never apply even when enable_dense_rrf.
    """
    if not require_dense_enabled(config):
        return False
    raw = config.get("dense_rrf_apply_profiles")
    if raw is None:
        allowed = {"nf", "blend"}
    elif isinstance(raw, str):
        allowed = {p.strip().lower() for p in raw.split(",") if p.strip()}
    else:
        allowed = {str(p).strip().lower() for p in raw if str(p).strip()}
    return str(profile).lower() in allowed
