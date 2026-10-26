"""Explicit, source-auditable Bailian v4 embeddings for new frozen experiments."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request, build_opener

from .tasks import TaskStore, _task

MODEL = "text-embedding-v4"
DIMENSIONS = {64, 128, 256, 512, 768, 1024, 1536, 2048}


def load_profile(value):
    profile = json.loads(Path(value).read_text(encoding="utf-8")) if isinstance(value, (str, Path)) else dict(value)
    if profile.get("model") != MODEL or profile.get("provider") != "bailian":
        raise ValueError("Require the explicit Bailian text-embedding-v4 profile")
    if set(profile) & {"api_key", "key", "token", "authorization"}:
        raise ValueError("Credentials must be supplied outside the profile")
    endpoint = urlsplit(profile["endpoint"])
    if (endpoint.scheme != "https" or not endpoint.hostname or endpoint.username or endpoint.password
            or endpoint.query or endpoint.fragment
            or not endpoint.hostname.endswith(".maas.aliyuncs.com")
            or endpoint.path != "/api/v1/services/embeddings/text-embedding/text-embedding"):
        raise ValueError("Require the verified official workspace DashScope endpoint")
    if type(profile.get("dimensions")) is not int or profile["dimensions"] not in DIMENSIONS:
        raise ValueError("Unsupported v4 dimensions")
    return profile


def specification(profile):
    p = load_profile(profile)
    return {"implementation": "bailian-v4-dashscope-dense-l2-v1",
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "provider": "bailian", "model": MODEL,
            "endpoint": p["endpoint"], "vendor_revision": "not_exposed_by_api",
            "dimensions": p["dimensions"], "output_type": "dense", "normalization": "l2",
            "document_text_type": "document", "query_text_type": "query",
            "document_prefix": "", "query_prefix": "", "instruct": None,
            "max_input_tokens": 8192, "oversize_policy": "reject_without_truncation"}


class BailianEmbeddings:
    """Same embed_documents/embed_query boundary as local embeddings.

    Exact successful batches are durably reused via the existing TaskStore.
    Credentials never enter the task inputs, attempt rows, identity, or errors.
    Transport-unknown and invalid returns remain held. At most three explicit
    sends handle known transient HTTP responses; no second retry layer exists.
    """

    def __init__(self, profile, *, user_confirmed=False, cache_dir=None, transport=None, timeout=120,
                 batch_size=10, max_attempts=3):
        if user_confirmed is not True:
            raise PermissionError("Bailian text-embedding-v4 requires explicit user confirmation for this use")
        self.profile = load_profile(profile)
        self.specification = specification(self.profile)
        digest = hashlib.sha256(json.dumps(self.specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.identity = "bailian-v4-dashscope-dense-l2-v1:" + digest
        self.dimensions = self.profile["dimensions"]
        if type(batch_size) is not int or not 1 <= batch_size <= 10:
            raise ValueError("batch_size must be 1..10")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("max_attempts must be 1..3")
        self.batch_size, self.max_attempts, self.timeout = batch_size, max_attempts, timeout
        key_file = self.profile.get("api_key_file")
        self._key = Path(key_file).expanduser().read_text(encoding="utf-8").strip() if key_file else os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not self._key:
            raise ValueError("Bailian credential is missing")
        root = Path(cache_dir or self.profile["cache_dir"]).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.store = TaskStore(root / "requests.sqlite3")
        self._transport = transport or self._send

    def _send(self, request, timeout):
        proxy = self.profile.get("proxy")
        opener = build_opener(ProxyHandler({"https": proxy})) if proxy else build_opener()
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def _vectors(self, value, count):
        rows = value["output"]["embeddings"]
        if (len(rows) != count or any(type(r.get("text_index")) is not int for r in rows)
                or sorted(r["text_index"] for r in rows) != list(range(count))):
            raise ValueError("Bailian result denominator/order differs")
        vectors = []
        for row in sorted(rows, key=lambda r: r["text_index"]):
            vector = row["embedding"]
            if (not isinstance(vector, list) or len(vector) != self.dimensions
                    or any(type(v) not in (int, float) or not math.isfinite(v) for v in vector)):
                raise ValueError("Bailian returned an invalid vector")
            norm = math.sqrt(math.fsum(v*v for v in vector))
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError("Bailian returned a zero vector")
            vectors.append([v/norm for v in vector])
        return vectors

    def _encode(self, texts, kind):
        payload = {"model": MODEL, "input": {"texts": texts},
                   "parameters": {"text_type": kind, "dimension": self.dimensions, "output_type": "dense"}}
        task = _task(self.store, "bailian_embedding", {"embedding_identity": self.identity, "payload": payload})
        saved = self.store.get_result(task["key"])
        if saved:
            return saved["result"]["vectors"]
        for _ in range(self.max_attempts):
            if len(self.store.list_attempts(task["id"])) >= self.max_attempts:
                raise RuntimeError("Bailian exact-input attempt budget exhausted")
            claimed = self.store.claim(task_id=task["id"])
            if not claimed:
                raise RuntimeError("Bailian exact input is already running or held; no automatic replay")
            owner = claimed["owner"]
            request = Request(self.profile["endpoint"], data=json.dumps(payload, ensure_ascii=False).encode(),
                              headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"})
            try:
                status, raw = self._transport(request, self.timeout)
            except Exception as error:
                # Do not persist exception args: transports can echo request headers.
                info = {"category": "transport_unknown", "exception_type": type(error).__name__}
                self.store.record_attempt(task["id"], owner, response=info, validation={"valid": False})
                self.store.fail(task["id"], owner, info, retry=False)
                raise RuntimeError("Bailian transport outcome is unknown; exact input held") from None
            try:
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("Response must be an object")
                raw_usage = value.get("usage", {})
                if not isinstance(raw_usage, dict):
                    raise ValueError("Invalid usage")
                usage = {k: v for k, v in raw_usage.items() if k in {
                    "input_tokens", "output_tokens", "prompt_tokens", "total_tokens"}
                    and type(v) is int and v >= 0}
            except (ValueError, TypeError, UnicodeError):
                self.store.record_attempt(task["id"], owner, response={"category": "invalid_json", "http_status": status},
                                          validation={"valid": False})
                self.store.fail(task["id"], owner, {"category": "invalid_json"}, retry=False)
                raise RuntimeError("Bailian returned invalid JSON; exact input held") from None
            request_id = value.get("request_id")
            request_id = request_id if isinstance(request_id, str) and re.fullmatch(r"[A-Za-z0-9-]{1,128}", request_id) else None
            metadata = {"http_status": status, "request_id": request_id, "model_requested": MODEL}
            if status != 200:
                info = {**metadata, "category": "http_error"}
                self.store.record_attempt(task["id"], owner, response=metadata, usage=usage, validation={"valid": False})
                attempts = len(self.store.list_attempts(task["id"]))
                retry = status in {408, 425, 429, 500, 502, 503, 504, 529} and attempts < self.max_attempts
                self.store.fail(task["id"], owner, info, retry=retry)
                if retry:
                    time.sleep(30 if status == 429 else min(2 ** attempts, 8))
                    continue
                raise RuntimeError(f"Bailian HTTP {status}; exact input held")
            try:
                vectors = self._vectors(value, len(texts))
            except (ValueError, KeyError, TypeError, OverflowError):
                self.store.record_attempt(task["id"], owner, response=metadata, usage=usage, validation={"valid": False, "category": "invalid_vectors"})
                self.store.fail(task["id"], owner, {"category": "invalid_vectors"}, retry=False)
                raise RuntimeError("Bailian returned invalid vectors; exact input held") from None
            result = {"vectors": vectors, "embedding_identity": self.identity, **metadata, "usage": usage}
            self.store.record_attempt(task["id"], owner, response=metadata, usage=usage)
            self.store.succeed(task["id"], owner, result)
            return vectors
        raise RuntimeError("Bailian request budget exhausted")

    def embed_documents(self, texts):
        if isinstance(texts, (str, bytes)):
            raise TypeError("Documents must be a sequence of strings")
        texts = list(texts)
        if any(not isinstance(v, str) or not v.strip() for v in texts):
            raise ValueError("Document strings must be nonempty")
        return [v for i in range(0, len(texts), self.batch_size)
                for v in self._encode(texts[i:i+self.batch_size], "document")]

    def embed_query(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query must be a nonempty string")
        return self._encode([text], "query")[0]
