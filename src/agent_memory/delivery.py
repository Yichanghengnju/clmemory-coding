"""clmemory delivery extensions over the attributed MemMachine event backend.

BM25, dense search and RRF are upstream components, not clmemory inventions.
This module controls source delivery; it never generates answers or patches.
"""
from collections import Counter
from datetime import UTC, datetime
import hashlib
import re

from .anchor_channel import append_evidence, normalized
from .domain import Evidence, SearchRequest
from .method_memmachine import MemMachineMemory, _index_lock


def source_hit(source, kind="source_append"):
    stamp = source.get("timestamp")
    date = "unknown" if stamp is None else datetime.fromtimestamp(stamp / 1000, UTC).isoformat()
    return Evidence(
        id="clmemory_" + hashlib.sha256(source["id"].encode()).hexdigest()[:24],
        content=f"[source={source['id']}; role={source['role']}; timestamp={date}]\n{source['content']}",
        source_ids=(source["id"],), score=0.0, kind=kind)


def append_unique(hits, candidates, sources, limit):
    """Keep the ranked prefix; deduplicate by source and unchanged body text."""
    seen = {sid for hit in hits for sid in hit.source_ids}
    texts = {normalized(sources[sid]["content"]).lower() for sid in seen if sid in sources}
    result = list(hits)
    for hit in candidates:
        if len(result) >= limit:
            break
        keys = {normalized(sources[sid]["content"]).lower() for sid in hit.source_ids if sid in sources}
        if seen.intersection(hit.source_ids) or not keys or keys.intersection(texts):
            continue
        result.append(hit)
        seen.update(hit.source_ids)
        texts.update(keys)
    return result


def neighbor_hits(hits, ordered, cap=8):
    by = {source["id"]: i for i, source in enumerate(ordered)}
    seen = {sid for hit in hits for sid in hit.source_ids}
    result = []
    for hit in hits:
        for sid in hit.source_ids:
            i = by[sid]
            for j in (i - 1, i + 1):
                if (0 <= j < len(ordered) and ordered[j]["session_id"] == ordered[i]["session_id"]
                        and ordered[j]["id"] not in seen):
                    result.append(source_hit(ordered[j], "same_session_neighbor"))
                    seen.add(ordered[j]["id"])
                    if len(result) >= cap:
                        return result
    return result


_FILE_HEADER = re.compile(r"^\[([^\]\n]+):\d+-\d+\](?:\s|$)")


def file_key(hit, sources):
    # Only explicit, unambiguous source headers are eligible. Never infer paths
    # from current queries, snippets, line contents or benchmark labels.
    files = set()
    for sid in hit.source_ids:
        match = _FILE_HEADER.match(sources[sid]["content"])
        if not match:
            return None
        files.add(match.group(1))
    return next(iter(files)) if len(files) == 1 else None


def coding_tail(ranked, sources, limit):
    """C7: preserve 1..32; cap known-file 33..100 at two per file.

    Freed slots receive same-file siblings in first-file-rank/source-ID order.
    Unknown headers pass through. Backfill may include demoted siblings, as in
    the original mechanical policy; this is not a global per-file cap.
    """
    ranked = ranked[:limit]
    prefix, tail = ranked[:32], ranked[32:]
    kept, counts = [], Counter()
    for hit in tail:
        key = file_key(hit, sources)
        if key is None or counts[key] < 2:
            kept.append(hit)
            if key is not None:
                counts[key] += 1
    result = prefix + kept
    first_rank = {}
    for pos, hit in enumerate(prefix):
        key = file_key(hit, sources)
        if key is not None:
            first_rank.setdefault(key, pos)
    siblings = []
    for source in sources.values():
        hit = source_hit(source, "coding_file_sibling")
        key = file_key(hit, sources)
        if key in first_rank:
            siblings.append((first_rank[key], source["id"], hit))
    siblings.sort(key=lambda item: (item[0], item[1]))
    return append_unique(result, [item[2] for item in siblings], sources, len(ranked))


class SubmissionMemory(MemMachineMemory):
    def __init__(self, db_path, *, embedder, track):
        if track not in {"textual", "coding"}:
            raise ValueError("track must be textual or coding")
        self.track = track
        super().__init__(db_path, embedder=embedder,
                         config={"embedding_dimensions": embedder.dimensions,
                                 "max_results": 100, "expand_context": 0, "dedup_overfetch": 4})

    @property
    def identity(self):
        return {**super().identity, "name": "clmemory", "release": "v1.0",
                "track": self.track, "delivery_revision": "clmemory-delivery-v1",
                "head_search_limit": 16, "deep_tail_limit": 100,
                "textual_neighbors": 8 if self.track == "textual" else 0,
                "textual_anchors": 8 if self.track == "textual" else 0,
                "coding_tail_policy": "C7-explicit-source-headers" if self.track == "coding" else None}

    def search(self, request):
        request = SearchRequest.model_validate(request.model_dump())
        # Cover the entire delivery operation, so concurrent Add cannot change
        # the source view between head, neighbor selection and tail retrieval.
        with _index_lock(self._user_path(request.user_id) / "delivery.lock"):
            head = super().search(request.model_copy(update={"top_k": min(request.top_k, 16)}))
            if not head or len(head) >= request.top_k:
                return head
            ordered = self._store.all_sources(request.user_id)
            sources = {source["id"]: source for source in ordered}
            hits = head
            if self.track == "textual":
                hits = append_unique(hits, neighbor_hits(head, ordered), sources, request.top_k)
                anchors = append_evidence(hits, ordered, request.query,
                                          m=min(8, request.top_k - len(hits)))
                hits = append_unique(hits, anchors, sources, request.top_k)
            if len(hits) < request.top_k:
                # Independent deeper retrieval supplies only the tail. The
                # 16-result head is never replaced by this deeper ranking.
                deep = super().search(request)
                hits = append_unique(hits, deep, sources, request.top_k)
            if self.track == "coding":
                hits = coding_tail(hits, sources, request.top_k)
            return hits

    def add(self, request):
        with _index_lock(self._user_path(request.user_id) / "delivery.lock"):
            return super().add(request)
