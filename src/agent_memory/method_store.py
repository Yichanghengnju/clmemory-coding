"""Persistence only for InvMem, ReFind and AMI; callers own Add/Search strategies.

Typical Add: check completed(request), derive from sources_for(request) without
any database lock, then commit(request, records). Every commit retains raw
messages. A kind='source' record attaches metadata/vector to unchanged raw
text; kind='raw' permits a method-rendered view. Both use id == source_ids[0]
and replace the default document without altering the persisted source message.
Identity is a frozen, strict JSON configuration, including method and revision.
"""

import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator

from .domain import AddRequest, Evidence, RequestConflict, Source
from .lexical import rank
from .memory import _canonical, _stable_id

# source_messages deliberately has the exact Storage column/constraint schema.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    identity TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS add_requests (
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    PRIMARY KEY (user_id, request_id)
);
CREATE TABLE IF NOT EXISTS source_messages (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    timestamp INTEGER CHECK (timestamp >= 0),
    PRIMARY KEY (user_id, id),
    UNIQUE (user_id, request_id, position),
    FOREIGN KEY (user_id, request_id)
        REFERENCES add_requests (user_id, request_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS method_source_request
    ON source_messages (user_id, request_id, id);
CREATE INDEX IF NOT EXISTS method_source_session
    ON source_messages (user_id, session_id, request_id, position);
CREATE TABLE IF NOT EXISTS method_records (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    text TEXT NOT NULL,
    kind TEXT NOT NULL,
    metadata TEXT NOT NULL,
    vector TEXT,
    PRIMARY KEY (user_id, id),
    UNIQUE (user_id, request_id, position),
    UNIQUE (user_id, request_id, id),
    FOREIGN KEY (user_id, request_id)
        REFERENCES add_requests (user_id, request_id)
);
CREATE INDEX IF NOT EXISTS method_record_kind ON method_records (user_id, kind, id);
CREATE TABLE IF NOT EXISTS method_record_sources (
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (user_id, record_id, position),
    UNIQUE (user_id, record_id, source_id),
    FOREIGN KEY (user_id, request_id, record_id)
        REFERENCES method_records (user_id, request_id, id),
    FOREIGN KEY (user_id, request_id, source_id)
        REFERENCES source_messages (user_id, request_id, id)
);
"""


def _json(value: object) -> str:
    """Reject lossy JSON coercions (tuple, non-string keys, custom objects)."""
    def check(item: object) -> None:
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float and math.isfinite(item):
            return
        if type(item) is list:
            for child in item:
                check(child)
            return
        if type(item) is dict and all(type(key) is str for key in item):
            for child in item.values():
                check(child)
            return
        raise ValueError("configuration/metadata must be strict finite JSON")
    check(value)
    return _canonical(value)


def _snapshot(request: AddRequest) -> AddRequest:
    return AddRequest.model_validate(request.model_dump())


def _payload_hash(request: AddRequest) -> str:
    return hashlib.sha256(_canonical(request.model_dump()).encode("utf-8")).hexdigest()


def sources_for(request: AddRequest) -> list[Source]:
    """Pure source construction, identical IDs to MemoryService, no database I/O."""
    request = _snapshot(request)
    return [
        Source(
            id=_stable_id("source", request.user_id, request.request_id, position),
            user_id=request.user_id, session_id=request.session_id,
            request_id=request.request_id, position=position, role=message.role,
            content=message.content, timestamp=message.timestamp,
        )
        for position, message in enumerate(request.messages)
    ]


def _vector(value: object) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ValueError("vector must be a nonempty list of finite numbers")
    result = []
    for number in value:
        if type(number) not in (int, float):
            raise ValueError("vector must contain finite numbers")
        try:
            number = float(number)
        except (ValueError, OverflowError) as error:
            raise ValueError("vector must contain finite numbers") from error
        if not math.isfinite(number):
            raise ValueError("vector must contain finite numbers")
        result.append(number)
    return result


def _unit(vector: list[float]) -> list[float]:
    # Scaling before the norm also handles very large/small finite embeddings.
    scale = max(abs(value) for value in vector)
    if not scale:
        return [0.0] * len(vector)
    scaled = [value / scale for value in vector]
    norm = math.hypot(*scaled)
    return [value / norm for value in scaled]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left, right = _vector(left), _vector(right)
    if len(left) != len(right):
        raise ValueError("vector dimensions differ")
    score = math.fsum(a * b for a, b in zip(_unit(left), _unit(right)))
    return max(-1.0, min(1.0, score))


def _limit(limit: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")


def cosine_rank(
    documents: list[dict], query_vector: list[float], limit: int
) -> list[dict]:
    """Rank a caller-selected corpus, ties by ID; omit missing/zero vectors.

    Dimension mismatches and nonfinite vectors are errors, not silent fallback.
    Negative similarities remain eligible. Results are copies with a score.
    """
    _limit(limit)
    query = _vector(query_vector)
    results = []
    for document in documents:
        if document.get("vector") is None:
            continue
        vector = _vector(document["vector"])
        if len(query) != len(vector):
            raise ValueError("vector dimensions differ")
        if not any(query) or not any(vector):
            continue
        results.append({**document, "score": cosine_similarity(query, vector)})
    return sorted(results, key=lambda item: (-item["score"], item["id"]))[:limit]


class MethodStore:
    sources_for = staticmethod(sources_for)

    def __init__(self, path: str | Path, identity: dict | str):
        if not ((type(identity) is dict and identity) or
                (type(identity) is str and identity.strip())):
            raise ValueError("identity must be a nonempty configuration dict or string")
        # Snapshot rather than retain the caller's mutable configuration object.
        self._identity = _json(identity)
        binding = _canonical({"method_store_version": 1, "identity": json.loads(self._identity)})
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            str(path), timeout=5.0, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            with self._transaction(write=True) as connection:
                connection.execute(_SCHEMA.split(";")[0])
                row = connection.execute(
                    "SELECT identity FROM memory_metadata WHERE singleton = 1"
                ).fetchone()
                if row is not None and row["identity"] != binding:
                    raise ValueError("database method/configuration identity differs")
                if row is None:
                    existing = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='add_requests'"
                    ).fetchone()
                    if existing and connection.execute("SELECT 1 FROM add_requests LIMIT 1").fetchone():
                        raise ValueError("database has completed requests without identity")
                    connection.execute("INSERT INTO memory_metadata VALUES (1, ?)", (binding,))
                for statement in _SCHEMA.split(";")[1:]:
                    if statement.strip():
                        connection.execute(statement)
        except BaseException:
            self._connection.close()
            raise

    @property
    def identity(self) -> dict | str:
        """A detached copy of the configuration bound at construction."""
        return json.loads(self._identity)

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("MethodStore is closed")
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @staticmethod
    def _completed(connection: sqlite3.Connection, request: AddRequest) -> bool:
        row = connection.execute(
            "SELECT payload_hash FROM add_requests WHERE user_id = ? AND request_id = ?",
            (request.user_id, request.request_id),
        ).fetchone()
        if row is None:
            return False
        if row["payload_hash"] != _payload_hash(request):
            raise RequestConflict("request_id already exists with a different payload")
        return True

    def completed(self, request: AddRequest) -> bool:
        request = _snapshot(request)
        with self._transaction() as connection:
            return self._completed(connection, request)

    def commit(self, request: AddRequest, records: list[dict]) -> None:
        """Atomically complete Add. Failed validation/writes leave nothing visible.

        Record fields: text, source_ids (ordered list/tuple), kind, metadata
        (dict, default {}), vector (finite list or None), and optional id.
        A retry of a completed payload leaves its first committed records intact.
        """
        request = _snapshot(request)
        if self.completed(request):
            return
        sources = sources_for(request)
        by_id = {source.id: source for source in sources}
        if not isinstance(records, list):
            raise ValueError("records must be a list of dicts")
        prepared = []
        seen = set()
        for position, record in enumerate(records):
            if not isinstance(record, dict) or set(record) - {
                "id", "text", "source_ids", "kind", "metadata", "vector"
            }:
                raise ValueError("invalid record fields")
            text, kind = record.get("text"), record.get("kind")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("record text must be nonempty")
            if not isinstance(kind, str) or not kind.strip():
                raise ValueError("record kind must be nonempty")
            refs = record.get("source_ids")
            if (not isinstance(refs, (list, tuple)) or not refs or
                    any(not isinstance(ref, str) or ref not in by_id for ref in refs) or
                    len(set(refs)) != len(refs)):
                raise ValueError("source_ids must be unique references within this request")
            refs = tuple(refs)
            record_id = record.get("id")
            if record_id is None:
                record_id = refs[0] if kind in ("source", "raw") else _stable_id(
                    "record", request.user_id, request.request_id, position
                )
            if not isinstance(record_id, str) or not record_id.strip():
                raise ValueError("record id must be nonempty")
            if record_id in seen:
                raise ValueError("duplicate record id")
            seen.add(record_id)
            if kind in ("source", "raw"):
                if len(refs) != 1 or record_id != refs[0]:
                    raise ValueError("raw/source record must retain its source id and self-reference")
                if kind == "source" and text != by_id[refs[0]].content:
                    raise ValueError("source record must retain its source text")
            elif record_id.startswith("source_"):
                raise ValueError("source IDs are reserved for kind='source' or 'raw'")
            metadata = record.get("metadata", {})
            if type(metadata) is not dict:
                raise ValueError("record metadata must be a dict")
            metadata_json = _json(metadata)
            vector = record.get("vector")
            vector_json = None if vector is None else _canonical(_vector(vector))
            prepared.append((record_id, position, text, kind, metadata_json, vector_json, refs))
        with self._transaction(write=True) as connection:
            # Serialize writers and recheck after any concurrent derivation.
            if self._completed(connection, request):
                return
            connection.execute(
                "INSERT INTO add_requests VALUES (?, ?, ?, ?)",
                (request.user_id, request.request_id, request.session_id, _payload_hash(request)),
            )
            connection.executemany(
                "INSERT INTO source_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(s.id, s.user_id, s.session_id, s.request_id, s.position,
                  s.role, s.content, s.timestamp) for s in sources],
            )
            for record_id, position, text, kind, metadata, vector, refs in prepared:
                connection.execute(
                    "INSERT INTO method_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (record_id, request.user_id, request.request_id, position,
                     text, kind, metadata, vector),
                )
                connection.executemany(
                    "INSERT INTO method_record_sources VALUES (?, ?, ?, ?, ?)",
                    [(request.user_id, request.request_id, record_id, ref, index)
                     for index, ref in enumerate(refs)],
                )

    def all_sources(self, user: str, session_id: str | None = None) -> list[dict]:
        with self._transaction() as connection:
            sql = "SELECT rowid AS seq, * FROM source_messages WHERE user_id = ?"
            params = [user]
            if session_id is not None:
                sql += " AND session_id = ?"
                params.append(session_id)
            return [dict(row) for row in connection.execute(
                sql + " ORDER BY rowid", params
            )]

    allsources = all_sources

    def source_lookup(self, user: str, ids: list[str] | tuple[str, ...]) -> list[dict]:
        """Resolve in requested order; raise ValueError for any missing user's ID."""
        if not isinstance(ids, (list, tuple)) or any(not isinstance(ref, str) for ref in ids):
            raise ValueError("ids must be a list/tuple of source IDs")
        with self._transaction() as connection:
            result = []
            for source_id in ids:
                row = connection.execute(
                    "SELECT rowid AS seq, * FROM source_messages WHERE user_id = ? AND id = ?",
                    (user, source_id),
                ).fetchone()
                if row is None:
                    raise ValueError(f"missing source: {source_id}")
                result.append(dict(row))
            return result

    def documents(self, user: str, kinds: list[str] | tuple[str, ...] | None = None) -> list[dict]:
        """One document per raw message plus derivatives, filtered before ranking.

        text/content are the stored literal text; metadata is separate and is
        never implicitly lexical-indexed. source is the raw row for source docs,
        otherwise None; sources contains the ordered, resolved raw references.
        Without a writer's kind='source' record, raw vector is None.
        """
        if kinds is not None and (not isinstance(kinds, (list, tuple, set, frozenset)) or
                                  any(not isinstance(kind, str) for kind in kinds)):
            raise ValueError("kinds must be a collection of kind strings or None")
        enabled = None if kinds is None else set(kinds)
        with self._transaction() as connection:
            sources = {row["id"]: dict(row) for row in connection.execute(
                "SELECT rowid AS seq, * FROM source_messages WHERE user_id = ? ORDER BY rowid", (user,)
            )}
            refs: dict[str, list[str]] = {}
            for row in connection.execute(
                "SELECT record_id, source_id FROM method_record_sources "
                "WHERE user_id = ? ORDER BY record_id, position", (user,)
            ):
                refs.setdefault(row["record_id"], []).append(row["source_id"])
            documents = {}
            if enabled is None or "source" in enabled:
                for source_id, source in sources.items():
                    documents[source_id] = {
                        "id": source_id, "user_id": user, "request_id": source["request_id"],
                        "session_id": source["session_id"], "seq": source["seq"],
                        "ordinal": source["seq"], "storedordinal": source["seq"],
                        "text": source["content"], "content": source["content"],
                        "source_ids": [source_id], "kind": "source",
                        "metadata": {**{key: source[key] for key in (
                            "session_id", "request_id", "position", "role", "timestamp"
                        )}, "ordinal": source["seq"], "roles": [source["role"]]},
                        "vector": None, "source": source, "sources": [source],
                    }
            for row in connection.execute(
                "SELECT rowid AS seq, * FROM method_records WHERE user_id = ? ORDER BY rowid", (user,)
            ):
                if enabled is not None and row["kind"] not in enabled:
                    continue
                source_ids = refs[row["id"]]
                resolved = [sources[source_id] for source_id in source_ids]
                first_source = resolved[0]
                metadata = {
                    "ordinal": row["seq"],
                    **json.loads(row["metadata"]),
                    "session_id": first_source["session_id"], "request_id": row["request_id"],
                    "position": first_source["position"], "role": first_source["role"],
                    "timestamp": first_source["timestamp"],
                    "roles": [source["role"] for source in resolved],
                }
                documents[row["id"]] = {
                    "id": row["id"], "user_id": user, "request_id": row["request_id"],
                    "session_id": first_source["session_id"], "seq": row["seq"],
                    "ordinal": row["seq"], "storedordinal": row["seq"],
                    "text": row["text"], "content": row["text"], "source_ids": source_ids,
                    "kind": row["kind"], "metadata": metadata,
                    "vector": None if row["vector"] is None else json.loads(row["vector"]),
                    "source": first_source if row["kind"] in ("source", "raw") else None,
                    "sources": resolved,
                }
            return sorted(documents.values(), key=lambda item: (item["seq"], item["id"]))

    def search_lexical(
        self, user: str, query: str, limit: int,
        kinds: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        _limit(limit)
        documents = self.documents(user, kinds)
        by_id = {document["id"]: document for document in documents}
        ranked = rank([
            (Evidence(d["id"], d["content"], tuple(d["source_ids"]), kind=d["kind"]), d["text"])
            for d in documents
        ], query, limit)
        return [{**by_id[item.id], "score": item.score} for item in ranked]

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "MethodStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
