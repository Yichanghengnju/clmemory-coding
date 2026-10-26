"""Native MemMachine event backend, pinned to a public upstream revision.

This adapter calls upstream EventMemory, its SQLite segment/vector stores and
USearch engine directly. It is an explicit event-only configuration, not a
reproduction of the unidentified leaderboard deployment. Semantic profiles,
short-term summaries and the retrieval agent are disabled; no generative model
is called. The injected embedder remains responsible for BGE preprocessing.

MethodStore is the authority for completed requests and unchanged sources.
Native writes precede its atomic completion; retrieval filters native records
to committed request IDs, so a failed Add cannot expose partial evidence. A
retry first forgets its stable event IDs, including any partial native write.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import Lock, RLock
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5
from weakref import WeakValueDictionary

from .domain import AddReceipt, AddRequest, Evidence, SearchRequest
from .method_store import MethodStore

UPSTREAM_COMMIT = "d5a39a8e774e977fb1e6c22efb03555c4e51e05e"
UPSTREAM_URL = "https://github.com/MemMachine/MemMachine"
_DEFAULTS = {
    "embedding_dimensions": 384,
    "deriver": "whole_text",
    "segmenter": "passthrough",
    "rrf_k": 60,
    "bm25_tokenizer": "simple",
    "dedup_overfetch": 4,
    "expand_context": 0,
    "max_results": 8,
}
_GUARD = Lock()
_LOCKS: WeakValueDictionary[str, RLock] = WeakValueDictionary()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


@contextmanager
def _index_lock(path: Path):
    """Serialize native index ownership across both threads and processes."""
    key = os.path.normcase(str(path.resolve()))
    with _GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _LOCKS[key] = lock
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            handle.seek(0, 2)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                handle.seek(0)
                if os.name == "nt":
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _native():
    """Load installed upstream code; missing dependencies are never a fallback."""
    try:
        from sqlalchemy import event
        from sqlalchemy.ext.asyncio import create_async_engine
        from memmachine_server.common.data_types import SimilarityMetric
        from memmachine_server.common.embedder import Embedder
        from memmachine_server.common.filter.filter_parser import In
        from memmachine_server.common.reranker.bm25_reranker import BM25Reranker, BM25RerankerParams
        from memmachine_server.common.reranker.identity_reranker import IdentityReranker
        from memmachine_server.common.reranker.rrf_hybrid_reranker import RRFHybridReranker, RRFHybridRerankerParams
        from memmachine_server.common.vector_store.data_types import VectorStoreCollectionConfig
        from memmachine_server.common.vector_store.sqlite_vector_store import SQLiteVectorStore, SQLiteVectorStoreParams
        from memmachine_server.common.vector_store.vector_search_engine.usearch_engine import USearchVectorSearchEngine
        from memmachine_server.episodic_memory.event_memory.data_types import Event, TextBlock, ProducerContext, FormatOptions
        from memmachine_server.episodic_memory.event_memory.deriver.text_deriver import WholeTextDeriver, SentenceTextDeriver
        from memmachine_server.episodic_memory.event_memory.event_memory import EventMemory, EventMemoryParams
        from memmachine_server.episodic_memory.event_memory.segmenter.passthrough_segmenter import PassthroughSegmenter
        from memmachine_server.episodic_memory.event_memory.segment_store import SegmentStorePartitionConfig
        from memmachine_server.episodic_memory.event_memory.segment_store.sqlalchemy_segment_store import SQLAlchemySegmentStore, SQLAlchemySegmentStoreParams
    except ImportError as exc:
        raise RuntimeError(
            "MemMachine native event dependencies are required. Install pinned "
            f"{UPSTREAM_URL}@{UPSTREAM_COMMIT} packages/common and packages/server "
            "in the dedicated Python >=3.12 environment."
        ) from exc
    return SimpleNamespace(**locals())


def _bridge(native, embedder, dimensions):
    class Bridge(native.Embedder):
        @property
        def model_id(self):
            return "agent-memory-explicit-embedding:" + _json(embedder.identity)

        @property
        def dimensions(self):
            return dimensions

        @property
        def similarity_metric(self):
            return native.SimilarityMetric.COSINE

        @staticmethod
        def _vectors(values, count):
            values = list(values)
            if len(values) != count:
                raise ValueError("embedding provider returned the wrong batch size")
            result = []
            for value in values:
                vector = [float(x) for x in value]
                if (len(vector) != dimensions or not any(vector)
                        or any(not math.isfinite(x) for x in vector)):
                    raise ValueError("embedding must be finite, nonzero and match embedding_dimensions")
                result.append(vector)
            return result

        async def _ingest_embed(self, inputs, max_attempts=1):
            return self._vectors(embedder.embed_documents(inputs), len(inputs))

        async def _search_embed(self, queries, max_attempts=1):
            return self._vectors([embedder.embed_query(q) for q in queries], len(queries))

    return Bridge()


class MemMachineMemory:
    """Synchronous Memory protocol over the actual upstream event implementation."""

    def __init__(self, db_path: str | Path, provider=None, embedder=None,
                 configdict=None, *, config=None, storage_identity=None):
        if configdict is not None and config is not None:
            raise ValueError("Specify only config or configdict")
        if embedder is None:
            raise ValueError("MemMachine requires an explicit embedder")
        if str(db_path) == ":memory:":
            raise ValueError("MemMachine native stores require a persistent database path")
        self.embedder = embedder
        supplied = dict(config if config is not None else configdict or {})
        if set(supplied) - set(_DEFAULTS):
            raise ValueError("unknown MemMachine configuration: " + str(sorted(set(supplied) - set(_DEFAULTS))))
        self.config = {**_DEFAULTS, **supplied}
        for name in ("embedding_dimensions", "dedup_overfetch", "max_results"):
            if type(self.config[name]) is not int or self.config[name] < 1:
                raise ValueError(name + " must be a positive integer")
        for name in ("rrf_k", "expand_context"):
            if type(self.config[name]) is not int or self.config[name] < 0:
                raise ValueError(name + " must be a nonnegative integer")
        if self.config["deriver"] not in {"whole_text", "sentence_text"}:
            raise ValueError("deriver must be whole_text or sentence_text")
        if self.config["segmenter"] != "passthrough":
            raise ValueError("this native configuration supports passthrough segmenter only")
        if self.config["bm25_tokenizer"] != "simple":
            raise ValueError("this native configuration supports upstream simple BM25 tokenizer only")
        self._identity = _json(self.identity)
        self._path = Path(db_path).resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._root = Path(str(self._path) + ".memmachine")
        self._store = MethodStore(self._path, self.identity if storage_identity is None else storage_identity)
        self._closed = False

    @property
    def identity(self):
        embedding = getattr(self.embedder, "identity", None)
        if not embedding:
            raise ValueError("embedder identity must be nonempty finite JSON")
        return json.loads(_json({
            "name": "memmachine", "version": 1,
            "upstream_repository": UPSTREAM_URL, "upstream_commit": UPSTREAM_COMMIT,
            "original_license": "Apache-2.0",
            "adaptation": "native EventMemory SQLite+USearch; event-only explicit configuration",
            "leaderboard_deployment_match": False,
            "embedding_identity": embedding,
            "disabled": ["semantic_memory", "short_term_summary", "retrieval_agent", "generative_llm"],
            "timestamp_policy": "source milliseconds; unknown remains null in authority; internal epoch sentinel hidden from embeddings/rerank/evidence",
            "timeline_expansion": "requires all committed sources to have explicit timestamps",
            "embedding_time_format": "date_style=None,time_style=None; upstream producer context retained",
            "completion_visibility": "native request-ID filter and source hydration restricted to completed requests",
            "user_directory": "96-bit SHA256 prefix with mandatory full-SHA256 ownership binding",
            "options_policy": "concatenate query and provided options without labels or gold",
            "source_output": "upstream ranked context hydrated from immutable MethodStore sources",
            **self.config,
        }))

    def _check_identity(self):
        if self._closed:
            raise RuntimeError("MemMachineMemory is closed")
        if self._identity != _json(self.identity):
            raise ValueError("MemMachine configuration or embedding identity changed")

    def _user_path(self, user):
        # Runner DB names already contain a full digest. Keeping another full
        # digest here exceeds Windows' usual 260-character path boundary.
        # The complete digest is separately bound below; a prefix collision
        # raises rather than opening another user's native index.
        return self._root / hashlib.sha256(user.encode("utf-8")).hexdigest()[:24]

    @asynccontextmanager
    async def _event_memory(self, user):
        n = _native()
        root = self._user_path(user)
        root.mkdir(parents=True, exist_ok=True)
        owner = root / "owner.sha256"
        expected_owner = hashlib.sha256(user.encode("utf-8")).hexdigest()
        if owner.exists():
            if owner.read_text(encoding="ascii") != expected_owner:
                raise ValueError("MemMachine native directory ownership differs")
        else:
            with owner.open("x", encoding="ascii") as handle:
                handle.write(expected_owner)
        engine = n.create_async_engine("sqlite+aiosqlite:///" + (root / "native.sqlite").as_posix())

        @n.event.listens_for(engine.sync_engine, "connect")
        def foreign_keys(connection, _record):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        vectors = n.SQLiteVectorStore(n.SQLiteVectorStoreParams(
            sqlalchemy_engine=engine,
            vector_search_engine_factory=lambda dim, metric: n.USearchVectorSearchEngine(
                num_dimensions=dim, similarity_metric=metric),
            index_directory=str(root / "indexes"), save_threshold=1,
        ))
        segments = n.SQLAlchemySegmentStore(n.SQLAlchemySegmentStoreParams(engine=engine))
        started = False
        try:
            await vectors.startup()
            started = True
            await segments.startup()
            partition = await segments.open_or_create_partition("events", n.SegmentStorePartitionConfig())
            collection = await vectors.open_or_create_collection(
                namespace="memory", name="events",
                config=n.VectorStoreCollectionConfig(
                    vector_dimensions=self.config["embedding_dimensions"],
                    similarity_metric=n.SimilarityMetric.COSINE,
                    indexed_properties_schema={
                        **n.EventMemory.expected_vector_store_collection_schema(),
                        "source_id": str, "request_id": str, "session_id": str,
                    },
                ),
            )
            # This is the simple tokenizer offered by the upstream manager.
            bm25 = n.BM25Reranker(n.BM25RerankerParams(
                tokenize=lambda text: re.sub(r"\W+", " ", text).lower().split()))
            reranker = n.RRFHybridReranker(n.RRFHybridRerankerParams(
                rerankers=[n.IdentityReranker(), bm25], k=self.config["rrf_k"]))
            deriver = n.WholeTextDeriver() if self.config["deriver"] == "whole_text" else n.SentenceTextDeriver()
            memory = n.EventMemory(n.EventMemoryParams(
                segment_store_partition=partition, vector_store_collection=collection,
                segmenter=n.PassthroughSegmenter(), deriver=deriver,
                embedder=_bridge(n, self.embedder, self.config["embedding_dimensions"]),
                reranker=reranker,
            ))
            yield n, memory, n.FormatOptions(date_style=None, time_style=None)
        finally:
            try:
                if started:
                    await vectors.shutdown()
                await segments.shutdown()
            finally:
                try:
                    # At the pinned revision shutdown() flushes indexes but
                    # leaves its internally created synchronous pool open.
                    vectors._sync_sqlalchemy_engine.dispose()
                finally:
                    await engine.dispose()

    async def _encode(self, user, sources):
        async with self._event_memory(user) as (n, memory, formatting):
            events = []
            for source in sources:
                event_id = uuid5(NAMESPACE_URL, "agent-memory/memmachine/" + user + "/" + source.id)
                # Upstream Event requires a datetime. This placeholder is an
                # internal storage value only; source authority keeps None and
                # all upstream text formatters suppress dates on this path.
                timestamp = datetime(1970, 1, 1, tzinfo=UTC) if source.timestamp is None else datetime.fromtimestamp(source.timestamp / 1000, UTC)
                events.append(n.Event(
                    uuid=event_id, timestamp=timestamp,
                    context=n.ProducerContext(producer=source.role),
                    blocks=[n.TextBlock(text=source.content)],
                    properties={"source_id": source.id, "request_id": source.request_id,
                                "session_id": source.session_id, "timestamp_known": source.timestamp is not None},
                ))
            await memory.forget_events([event.uuid for event in events])
            await memory.encode_events(events, format_options=formatting)

    def add(self, request: AddRequest) -> AddReceipt:
        self._check_identity()
        request = AddRequest.model_validate(request.model_dump())
        receipt = AddReceipt(request_id=request.request_id, user_id=request.user_id, session_id=request.session_id)
        with _index_lock(self._user_path(request.user_id) / "index.lock"):
            if self._store.completed(request):
                return receipt
            asyncio.run(self._encode(request.user_id, self._store.sources_for(request)))
            self._check_identity()
            self._store.commit(request, [])
        return receipt

    async def _query(self, request, sources):
        if self.config["expand_context"] and any(source["timestamp"] is None for source in sources.values()):
            raise ValueError("native timeline expansion requires explicit source timestamps")
        async with self._event_memory(request.user_id) as (n, memory, formatting):
            query = request.query
            if request.options:
                query += "\n" + "\n".join(request.options)
            limit = min(request.top_k, self.config["max_results"])
            result = await memory.query(
                query, vector_search_limit=limit * self.config["dedup_overfetch"],
                expand_context=self.config["expand_context"],
                property_filter=n.In("m.request_id", list(dict.fromkeys(
                    source["request_id"] for source in sources.values()))),
                format_options=formatting,
            )
            found, seen = [], set()
            for context in result.scored_segment_contexts:
                ids = []
                for segment in context.segments:
                    source_id = segment.properties.get("source_id")
                    if source_id in sources and source_id not in ids:
                        ids.append(source_id)
                if not ids or all(source_id in seen for source_id in ids):
                    continue
                # Hydrate exact sources, as the upstream long-term wrapper
                # hydrates episodes. Source IDs remain stable across retries.
                ids = [source_id for source_id in ids if source_id not in seen]
                parts = []
                for source_id in ids:
                    source = sources[source_id]
                    date = "unknown" if source["timestamp"] is None else datetime.fromtimestamp(source["timestamp"] / 1000, UTC).isoformat()
                    parts.append(f"[source={source_id}; role={source['role']}; timestamp={date}]\n{source['content']}")
                found.append(Evidence(
                    id="memmachine_" + hashlib.sha256(_json(ids).encode("utf-8")).hexdigest()[:24],
                    content="\n\n".join(parts), source_ids=tuple(ids),
                    score=float(context.score), kind="memmachine_event",
                ))
                seen.update(ids)
                if len(found) >= limit:
                    break
            return found

    def search(self, request: SearchRequest) -> list[Evidence]:
        self._check_identity()
        request = SearchRequest.model_validate(request.model_dump())
        with _index_lock(self._user_path(request.user_id) / "index.lock"):
            sources = {source["id"]: source for source in self._store.all_sources(request.user_id)}
            if not sources:
                return []
            result = asyncio.run(self._query(request, sources))
            self._check_identity()
            return result

    def close(self):
        if not self._closed:
            self._store.close()
            self._closed = True
