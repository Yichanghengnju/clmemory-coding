"""Opt-in AMI observations; records belong to existing caller-owned results.

No text, queries, options, credentials or model configuration are accepted by
the observation helpers. IDs remain linkable data, not anonymous telemetry.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import json
import logging
from threading import Lock
import uuid

_ACTIVE = ContextVar("ami_evidence_trace", default=None)
_LOG = logging.getLogger(__name__)
METHOD = "active_memoryindex"
REVISION = "ami-observed-flow-v1"
_FIELDS = {
    "query_plan": {"round", "branch", "recall_status", "stop_policy"},
    "observed_candidates": {"boundary", "internal_top_k_truncation", "candidates", "independent_rerank"},
    "take_attempt": {"memory_id", "selection_score", "score_channel", "chars_left", "byte_units_left"},
    "take_rejected": {"memory_id", "reason"},
    "take_accepted": {"memory_id", "chars_left", "byte_units_left"},
    "neighbor_expansion": {"seed_id", "memory_id", "offset"},
    "selection_end": {"result_limit", "reason", "unvisited_ids", "ordering", "independent_assembly"},
    "method_return": {"items", "count"}, "http_method_result": {"items", "count"},
    "protocol_rejected": {"category", "requested_top_k", "returned_count"},
    "execution_exception": {"error_type"}, "http_exception": {"error_type"},
}


def current_trace():
    return _ACTIVE.get()


def _observation(function):
    def safe(self, *args, **kwargs):
        try:
            return function(self, *args, **kwargs)
        except Exception:
            self.data["capture_status"] = "incomplete"
            self.recorder.count("capture_failed")
    return safe


class TraceRecorder:
    """Explicit local execution option, excluded from method/cache identity.

    Runner traces are embedded in the existing stage result. HTTP users supply
    a writer that attaches to their existing result/attempt mechanism. No file,
    database, thread, exporter or network client is created by this recorder.
    """
    def __init__(self, *, enabled=False, run_id=None, writer=None):
        if type(enabled) is not bool:
            raise TypeError("enabled must be bool")
        if writer is not None and not callable(writer):
            raise TypeError("trace writer must be callable")
        self.enabled, self.run_id, self.writer = enabled, run_id, writer
        self._lock = Lock()
        self._counts = {"started": 0, "finished": 0, "write_failed_or_unknown": 0,
                        "capture_failed": 0}

    def count(self, name):
        with self._lock:
            self._counts[name] += 1

    @property
    def status(self):
        with self._lock:
            return dict(self._counts)

    @contextmanager
    def capture(self, *, result_key=None, run_id=None, task_id=None, attempt_id=None):
        if not self.enabled:
            yield None
            return
        trace = _Trace(self, {"run_id": run_id or self.run_id,
                              "result_key": result_key, "task_id": task_id,
                              "attempt_id": attempt_id})
        token = _ACTIVE.set(trace)
        try:
            yield trace
        except Exception as error:
            trace.event("execution_exception", error_type=type(error).__name__)
            raise  # tracing never turns a failed Search into a success
        finally:
            _ACTIVE.reset(token)


class _Trace:
    def __init__(self, recorder, binding):
        self.recorder = recorder
        self.data = {"revision": REVISION, "trace_id": str(uuid.uuid4()),
                     **binding, "method": METHOD, "events": [],
                     "text_recording": False, "semantic_support_verified": False,
                     "capture_status": "complete", "delivery": None}
        if self.data["result_key"] is None:
            self.data["result_key"] = self.data["trace_id"]
        self.finished = False
        recorder.count("started")

    def event(self, stage, **fields):
        try:
            if stage not in _FIELDS or set(fields) - _FIELDS[stage]:
                raise ValueError("unsupported trace fields")
            event = {"stage": stage, **fields}
            # Reject non-finite/unserializable observer data before it reaches
            # TaskStore. Only exception types, never exception messages, survive.
            event = json.loads(json.dumps(event, allow_nan=False))
            self.data["events"].append(event)
        except Exception:
            self.data["capture_status"] = "incomplete"
            self.recorder.count("capture_failed")

    def configure(self, identity, documents, top_k):
        try:
            self.data.update(
                method_version=identity["version"], upstream_commit=identity["upstream_commit"],
                config_id=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                top_k=top_k,
                snapshot={"existing_revision": None,
                          "boundary": "MethodStore.documents(user, kinds=[raw,fact])",
                          "observed_ids_sha256": hashlib.sha256(json.dumps(
                              [(d["id"], d.get("seq")) for d in documents]).encode()).hexdigest(),
                          "note": "observed manifest, not a versioned whole-memory snapshot"},
            )
        except Exception:
            self.data["capture_status"] = "incomplete"
            self.recorder.count("capture_failed")

    @_observation
    def candidates(self, ranked):
        candidates = []
        for rank, (document, score) in enumerate(ranked, 1):
            refs = list(document.get("source_ids") or [])
            resolved = {s["id"] for s in document.get("sources", [])}
            candidates.append({"memory_id": document["id"], "source_refs": refs,
                               "source_mapping": "known" if refs and set(refs) <= resolved else "missing",
                               "rank": rank, "raw_score": score,
                               "channel": "weighted_cosine_original_recall_0.5_0.5",
                               "round": 1, "branch": "raw_fact_union"})
        self.event("observed_candidates", boundary="ranked raw/fact documents before take()",
                   internal_top_k_truncation=False, candidates=candidates,
                   independent_rerank="not_applicable")

    @_observation
    def returned(self, hits, *, stage="method_return"):
        self.event(stage, items=[{"memory_id": h.id, "source_refs": list(h.source_ids),
                                            "rank": n, "score": h.score}
                                           for n, h in enumerate(hits, 1)], count=len(hits))

    def finish(self, *, delivery=None):
        if delivery is not None:
            self.data["delivery"] = delivery
        if not self.finished:
            self.finished = True
            self.recorder.count("finished")
            self.data["write_status"] = "attached_to_enclosing_stage_result"
            if self.recorder.writer is not None:
                self.data["write_status"] = "writer_pending"
                try:
                    self.recorder.writer(json.loads(json.dumps(self.data, allow_nan=False)))
                    self.data["write_status"] = "writer_returned"
                except Exception as error:
                    self.data["write_status"] = "failed_or_unknown"
                    self.recorder.count("write_failed_or_unknown")
                    try:
                        _LOG.warning("Evidence trace write failed (%s)", type(error).__name__)
                    except Exception:
                        pass  # the in-process failure counter remains observable
        return self.data


@contextmanager
def trace_scope(recorder, **binding):
    if recorder is None:
        yield None
    else:
        with recorder.capture(**binding) as trace:
            yield trace


class SearchTraceMiddleware:
    """ASGI send boundary for one explicitly selected AMI application."""
    def __init__(self, app, recorder):
        self.app, self.recorder = app, recorder

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") != "/search":
            await self.app(scope, receive, send)
            return
        with self.recorder.capture() as trace:
            scope.setdefault("state", {})["evidence_trace"] = trace
            status = None

            async def observed_send(message):
                nonlocal status
                await send(message)  # never catch or suppress transport errors
                if message["type"] == "http.response.start":
                    status = message["status"]
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    trace.finish(delivery={"http_status": status,
                                          "category": {200: "response_sent", 401: "authentication_rejected",
                                                       422: "request_validation_rejected", 409: "request_conflict",
                                                       503: "service_unavailable", 504: "service_timeout"}.get(status, "http_rejected"),
                                          "boundary": "ASGI final body send returned; not client acknowledgement"})
            try:
                await self.app(scope, receive, observed_send)
            except Exception as error:
                trace.event("http_exception", error_type=type(error).__name__)
                # Generic 500 is constructed outside user middleware by the
                # existing ServerErrorMiddleware; its handler finalizes below.
                raise


def finish_http_error(request, status):
    trace = getattr(request.state, "evidence_trace", None)
    if trace is not None:
        rejected = any(e["stage"] == "protocol_rejected" for e in trace.data["events"])
        trace.finish(delivery={"http_status": status,
                               "category": "protocol_rejected" if rejected else "service_error",
                               "boundary": "existing exception handler constructed response; send not observed"})


def diagnose(trace, *, support=None, source_rows=(), documents=(), returned=()):
    """Offline only: explicit, complete literal-support fixture annotations.

    Labels/quotes are never supplied to capture() or the method. This helper
    checks real source text and returned text, not merely matching IDs. It is
    not a semantic evaluator for arbitrary paraphrases or unknown annotations.
    """
    unknown = {"conclusion": "annotation_or_source_insufficient", "support": []}
    if not support or trace.get("capture_status") != "complete":
        return {**unknown, "flow_only": trace.get("events", []) if not support else []}
    if not all(any(e["stage"] == stage for e in trace.get("events", []))
               for stage in ("observed_candidates", "method_return")):
        return unknown  # absence of an observed boundary is not observed absence
    if not support.get("complete_literal_mapping"):
        return unknown
    sources = {s["id"]: s["content"] for s in source_rows}
    by_id = {d["id"]: d for d in documents}
    candidates = {c["memory_id"]: c for e in trace["events"] if e["stage"] == "observed_candidates"
                  for c in e["candidates"]}
    delivered = {h.id: h for h in returned}
    delivery = trace.get("delivery")
    if delivery is not None and delivery.get("http_status") != 200:
        delivered = {}
    results = []
    for item in support.get("necessary", []):
        sid, quote = item.get("source_id"), item.get("quote")
        if not quote or quote not in sources.get(sid, "") or not item.get("memory_ids"):
            return unknown
        if any(i not in by_id or quote not in by_id[i].get("text", "") or
               sid not in by_id[i].get("source_ids", []) for i in item["memory_ids"]):
            return unknown
        candidate_ids = set(item["memory_ids"]) & candidates.keys()
        if any(candidates[i]["source_mapping"] != "known" for i in candidate_ids):
            return unknown
        # An annotation pointing at the right ID but unsupported text is unknown.
        if any(quote not in by_id.get(i, {}).get("text", "") or
               sid not in candidates[i]["source_refs"] for i in candidate_ids):
            return unknown
        returned_ids = [i for i in item["memory_ids"] if i in delivered and
                        quote in delivered[i].content and sid in delivered[i].source_ids]
        outcome = ("necessary_evidence_delivered" if returned_ids else
                   "necessary_evidence_in_candidates_not_delivered" if candidate_ids else
                   "necessary_evidence_not_in_observed_candidates")
        results.append({"source_id": sid, "candidate_ids": sorted(candidate_ids),
                        "delivered_ids": returned_ids, "conclusion": outcome})
    if not results:
        return unknown
    kinds = {r["conclusion"] for r in results}
    return {"conclusion": next(iter(kinds)) if len(kinds) == 1 else "mixed_support_flow",
            "support": results, "scope": "annotated synthetic literal-support fixture only"}
