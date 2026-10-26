"""SQLite storage for completed Add requests and their attributable evidence."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import Iterator

from .domain import AddRequest, Evidence, Fact, RequestConflict, Source

_SCHEMA = """
BEGIN IMMEDIATE;
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
CREATE TABLE IF NOT EXISTS facts (
    id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    text TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    effective_time TEXT NOT NULL,
    attribution TEXT NOT NULL CHECK (attribution IN ('user', 'assistant', 'unknown')),
    status TEXT NOT NULL CHECK (status IN ('current', 'historical', 'negated', 'uncertain')),
    supersedes TEXT NOT NULL,
    PRIMARY KEY (user_id, id),
    UNIQUE (user_id, request_id, position),
    FOREIGN KEY (user_id, request_id)
        REFERENCES add_requests (user_id, request_id)
);
CREATE TABLE IF NOT EXISTS fact_sources (
    user_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    position INTEGER NOT NULL CHECK (position >= 0),
    PRIMARY KEY (user_id, fact_id, position),
    UNIQUE (user_id, fact_id, source_id),
    FOREIGN KEY (user_id, fact_id) REFERENCES facts (user_id, id),
    FOREIGN KEY (user_id, source_id) REFERENCES source_messages (user_id, id)
);
COMMIT;
"""


class Storage:
    def __init__(self, db_path: str | Path):
        self._lock = RLock()
        self._closed = False
        self._connection = sqlite3.connect(
            str(db_path), timeout=5.0, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        try:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 5000")
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)
        except BaseException:
            self._connection.close()
            raise

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        # This lock protects one connection only, and is never held by extraction.
        with self._lock:
            if self._closed:
                raise RuntimeError("MemoryService is closed")
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield connection
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    def bind_identity(self, identity: str) -> None:
        with self._transaction(write=True) as connection:
            row = connection.execute(
                "SELECT identity FROM memory_metadata WHERE singleton = 1"
            ).fetchone()
            if row is not None:
                if row["identity"] != identity:
                    raise ValueError("database memory method or extractor strategy_identity differs")
                return
            # Existing successful rows without identity cannot be assigned a
            # configuration retrospectively: their extraction provenance is unknown.
            if connection.execute("SELECT 1 FROM add_requests LIMIT 1").fetchone():
                raise ValueError("database has completed requests without a configuration identity")
            connection.execute("INSERT INTO memory_metadata VALUES (1, ?)", (identity,))

    @staticmethod
    def _completed(
        connection: sqlite3.Connection, user_id: str, request_id: str, payload_hash: str
    ) -> bool:
        row = connection.execute(
            "SELECT payload_hash FROM add_requests WHERE user_id = ? AND request_id = ?",
            (user_id, request_id),
        ).fetchone()
        if row is None:
            return False
        if row["payload_hash"] != payload_hash:
            raise RequestConflict("request_id already exists with a different payload")
        return True

    def completed(self, user_id: str, request_id: str, payload_hash: str) -> bool:
        with self._transaction() as connection:
            return self._completed(connection, user_id, request_id, payload_hash)

    def commit_add(
        self,
        request: AddRequest,
        payload_hash: str,
        sources: list[Source],
        facts: list[tuple[str, Fact]],
    ) -> None:
        with self._transaction(write=True) as connection:
            # Another process may have committed while this caller was extracting.
            if self._completed(connection, request.user_id, request.request_id, payload_hash):
                return
            connection.execute(
                "INSERT INTO add_requests VALUES (?, ?, ?, ?)",
                (request.user_id, request.request_id, request.session_id, payload_hash),
            )
            connection.executemany(
                "INSERT INTO source_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        source.id, source.user_id, source.session_id, source.request_id,
                        source.position, source.role, source.content, source.timestamp,
                    )
                    for source in sources
                ],
            )
            for position, (fact_id, fact) in enumerate(facts):
                connection.execute(
                    "INSERT INTO facts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        fact_id, request.user_id, request.request_id, position,
                        fact.text, fact.subject, fact.predicate, fact.value,
                        fact.effective_time, fact.attribution, fact.status,
                        json.dumps(fact.supersedes, ensure_ascii=False),
                    ),
                )
                connection.executemany(
                    "INSERT INTO fact_sources VALUES (?, ?, ?, ?)",
                    [
                        (request.user_id, fact_id, source_id, index)
                        for index, source_id in enumerate(fact.source_ids)
                    ],
                )

    def documents(
        self, user_id: str, *, include_sources: bool, include_facts: bool
    ) -> list[tuple[Evidence, str]]:
        documents: list[tuple[Evidence, str]] = []
        # A single read snapshot keeps facts and their references consistent.
        with self._transaction() as connection:
            if include_sources:
                rows = connection.execute(
                    "SELECT * FROM source_messages WHERE user_id = ? ORDER BY id",
                    (user_id,),
                ).fetchall()
                for row in rows:
                    timestamp = "unknown" if row["timestamp"] is None else str(row["timestamp"])
                    metadata = json.dumps(
                        {
                            "role": row["role"], "timestamp": timestamp,
                            "session_id": row["session_id"],
                            "request_id": row["request_id"], "position": row["position"],
                        },
                        ensure_ascii=False, sort_keys=True,
                    )
                    content = f"{row['content']}\n[source {metadata}]"
                    documents.append(
                        (Evidence(row["id"], content, (row["id"],), kind="source"), row["content"])
                    )
            if include_facts:
                refs: dict[str, list[str]] = {}
                for row in connection.execute(
                    "SELECT fact_id, source_id FROM fact_sources "
                    "WHERE user_id = ? ORDER BY fact_id, position",
                    (user_id,),
                ):
                    refs.setdefault(row["fact_id"], []).append(row["source_id"])
                for row in connection.execute(
                    "SELECT * FROM facts WHERE user_id = ? ORDER BY id", (user_id,)
                ):
                    metadata = json.dumps(
                        {
                            "subject": row["subject"], "predicate": row["predicate"],
                            "value": row["value"], "effective_time": row["effective_time"],
                            "attribution": row["attribution"], "status": row["status"],
                            "supersedes": json.loads(row["supersedes"]),
                        },
                        ensure_ascii=False, sort_keys=True,
                    )
                    content = f"{row['text']}\n[fact {metadata}]"
                    search_text = " ".join(
                        row[field] for field in (
                            "text", "subject", "predicate", "value", "effective_time",
                            "attribution", "status",
                        )
                    )
                    documents.append(
                        (Evidence(row["id"], content, tuple(refs[row["id"]]), kind="fact"), search_text)
                    )
        return documents

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._connection.close()
                self._closed = True
