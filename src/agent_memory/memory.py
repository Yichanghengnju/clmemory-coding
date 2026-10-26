"""The shared Add/Search core for BM25 and extractor-backed fact memory."""

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from threading import Lock, RLock
from typing import Iterator
from weakref import WeakValueDictionary

from .domain import AddReceipt, AddRequest, Evidence, Extractor, Fact, SearchRequest, Source
from .lexical import rank
from .storage import Storage

_LOCK_GUARD = Lock()
_REQUEST_LOCKS: WeakValueDictionary[tuple[str, str, str], RLock] = WeakValueDictionary()


@contextmanager
def _request_lock(key: tuple[str, str, str]) -> Iterator[None]:
    # Share locks across service instances in this process. Weak references bound
    # the registry to requests that currently have callers, without evicting locks
    # while another caller is waiting on one.
    with _LOCK_GUARD:
        lock = _REQUEST_LOCKS.get(key)
        if lock is None:
            lock = RLock()
            _REQUEST_LOCKS[key] = lock
    with lock:
        yield


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _stable_id(kind: str, *parts: object) -> str:
    digest = hashlib.sha256(_canonical(parts).encode("utf-8")).hexdigest()
    return f"{kind}_{digest}"


def _validate_facts(facts: list[Fact], sources: list[Source]) -> None:
    if not isinstance(facts, list):
        raise ValueError("extractor must return a list of Fact objects")
    by_id = {source.id: source for source in sources}
    for fact in facts:
        if not isinstance(fact, Fact):
            raise ValueError("extractor returned a non-Fact value")
        for field in ("text", "subject", "predicate", "value", "effective_time"):
            value = getattr(fact, field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"fact {field} must be nonempty text")
        if fact.attribution not in ("user", "assistant", "unknown"):
            raise ValueError("invalid fact attribution")
        if fact.status not in ("current", "historical", "negated", "uncertain"):
            raise ValueError("invalid fact status")
        if (
            not isinstance(fact.source_ids, tuple)
            or not fact.source_ids
            or any(not isinstance(ref, str) or ref not in by_id for ref in fact.source_ids)
            or len(set(fact.source_ids)) != len(fact.source_ids)
        ):
            raise ValueError("fact source_ids must be unique references to this request's sources")
        if fact.attribution == "user" and not any(
            by_id[ref].role == "user" for ref in fact.source_ids
        ):
            raise ValueError("assistant-only sources cannot support user attribution")
        if not isinstance(fact.supersedes, tuple) or fact.supersedes:
            raise ValueError("nonempty supersedes is unsupported until relationships can be verified")


class MemoryService:
    def __init__(
        self,
        db_path: str | Path,
        *,
        method: str = "bm25",
        extractor: Extractor | None = None,
        include_sources: bool = True,
        include_facts: bool = True,
        storage_identity: dict | None = None,
    ):
        if method not in ("bm25", "fact"):
            raise ValueError("method must be 'bm25' or 'fact'")
        if method == "fact" and extractor is None:
            raise ValueError("method='fact' requires an extractor")
        self.method = method
        self.extractor = extractor
        self.include_sources = include_sources
        self.include_facts = include_facts
        self._identity = self._configuration_identity()
        self._storage = Storage(db_path)
        try:
            if storage_identity is not None and (not isinstance(storage_identity, dict) or not storage_identity):
                raise ValueError("storage_identity must be a nonempty Add component contract")
            self._storage.bind_identity(self._identity if storage_identity is None else _canonical(storage_identity))
        except BaseException:
            self._storage.close()
            raise
        self._database_key = (
            f":memory:{id(self)}" if str(db_path) == ":memory:"
            else os.path.normcase(str(Path(db_path).resolve()))
        )

    def _configuration_identity(self) -> str:
        strategy = None
        if self.method == "fact":
            if self.extractor is None or not hasattr(self.extractor, "strategy_identity"):
                raise ValueError("fact extractor requires explicit JSON strategy_identity")
            strategy = self.extractor.strategy_identity
            if strategy is None or strategy == {} or strategy == "":
                raise ValueError("fact extractor strategy_identity must not be empty")
        try:
            return _canonical({
                "method": self.method, "storage_version": 1, "strategy_identity": strategy
            })
        except (TypeError, ValueError) as error:
            raise ValueError("extractor strategy_identity must be finite JSON") from error

    def _check_configuration(self) -> None:
        if self._configuration_identity() != self._identity:
            raise ValueError("memory method or extractor strategy_identity changed after initialization")

    def add(self, request: AddRequest) -> AddReceipt:
        self._check_configuration()
        # Snapshot a mutable Pydantic request before hashing or calling user code.
        request = AddRequest.model_validate(request.model_dump())
        payload_hash = hashlib.sha256(
            _canonical(request.model_dump()).encode("utf-8")
        ).hexdigest()
        receipt = AddReceipt(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )
        key = (self._database_key, request.user_id, request.request_id)
        with _request_lock(key):
            if self._storage.completed(request.user_id, request.request_id, payload_hash):
                return receipt
            sources = [
                Source(
                    id=_stable_id("source", request.user_id, request.request_id, position),
                    user_id=request.user_id, session_id=request.session_id,
                    request_id=request.request_id, position=position, role=message.role,
                    content=message.content, timestamp=message.timestamp,
                )
                for position, message in enumerate(request.messages)
            ]
            facts: list[Fact] = []
            if self.method == "fact":
                assert self.extractor is not None
                # No database transaction/connection lock is held during this call.
                # The copy prevents an extractor from reordering/mutating our list.
                facts = self.extractor.extract(list(sources))
                _validate_facts(facts, sources)
            stored_facts = [
                (_stable_id("fact", request.user_id, request.request_id, position), fact)
                for position, fact in enumerate(facts)
            ]
            self._check_configuration()
            self._storage.commit_add(request, payload_hash, sources, stored_facts)
        return receipt

    def search(self, request: SearchRequest) -> list[Evidence]:
        self._check_configuration()
        request = SearchRequest.model_validate(request.model_dump())
        documents = self._storage.documents(
            request.user_id,
            include_sources=self.include_sources,
            include_facts=self.include_facts,
        )
        query = request.query
        if request.options:
            query += " " + " ".join(request.options)
        return rank(documents, query, request.top_k)

    def close(self) -> None:
        self._storage.close()
