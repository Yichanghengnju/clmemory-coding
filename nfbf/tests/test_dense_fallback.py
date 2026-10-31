"""Graceful degradation of the dense channel: provider failure must never turn a
search into an error; it falls back to the BM25 order and backs off (2026-10-30)."""

import tempfile
import unittest
from pathlib import Path

from coding_memory.core import Memory

ROOT = Path(__file__).resolve().parents[1]
NFBF = json_config = __import__("json").loads(
    (ROOT / "configs/coding-submit-nf-bf-v0.json").read_text(encoding="utf-8")
)


class ExplodingEmbedder:
    """Simulates a rate-limited / down embedding provider."""

    def __init__(self):
        self.query_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        raise RuntimeError("Bailian HTTP 429 (simulated)")

    def embed_documents(self, texts):
        raise RuntimeError("Bailian HTTP 429 (simulated)")


class CountingEmbedder:
    def __init__(self, dims: int = 8):
        self.dims = dims
        self.query_calls = 0

    def _vec(self, text: str):
        v = [0.0] * self.dims
        for i, ch in enumerate(text.casefold()):
            v[i % self.dims] += (ord(ch) % 31) / 31.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.query_calls += 1
        return self._vec(text)


class DenseFallbackTests(unittest.TestCase):
    def _mem(self, **updates):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "m.sqlite3"
        cfg = dict(NFBF, enable_dense_rrf=True, embedding_calls=True, **updates)
        return Memory(path, cfg)

    def _add(self, mem, user, request_id, content, ts):
        mem.add(
            {
                "user_id": user,
                "request_id": request_id,
                "session_id": "s",
                "messages": [{"role": "user", "content": content, "timestamp": ts}],
            }
        )

    def test_embedder_failure_returns_bm25_results_and_logs(self):
        mem = self._mem()
        self._add(mem, "u", "r1", "def hello():\n    return 1\n", 1)
        self._add(mem, "u", "r2", "exportReporter writes CSV rows for dashboard.\n", 2)
        breaker = ExplodingEmbedder()
        mem.set_dense_embedder(breaker)
        trace = {}
        hits = mem.search({"user_id": "u", "query": "hello return", "top_k": 3}, trace=trace)
        self.assertTrue(hits, "search must still return BM25 results when dense fails")
        self.assertIn("dense_rrf_error", trace)
        self.assertEqual(trace["dense_rrf_error"]["type"], "RuntimeError")

    def test_cooldown_suppresses_retry(self):
        mem = self._mem(dense_error_cooldown_seconds=3600)
        self._add(mem, "u", "r1", "def hello():\n    return 1\n", 1)
        breaker = ExplodingEmbedder()
        mem.set_dense_embedder(breaker)
        mem.search({"user_id": "u", "query": "hello", "top_k": 3})
        first_calls = breaker.query_calls
        self.assertEqual(first_calls, 1)
        mem.search({"user_id": "u", "query": "hello again", "top_k": 3})
        self.assertEqual(breaker.query_calls, first_calls, "cooldown must suppress retry")

    def test_recovery_after_cooldown(self):
        mem = self._mem(dense_error_cooldown_seconds=1)
        self._add(mem, "u", "r1", "def hello():\n    return 1\n", 1)
        breaker = ExplodingEmbedder()
        mem.set_dense_embedder(breaker)
        mem.search({"user_id": "u", "query": "hello", "top_k": 3})
        self.assertEqual(breaker.query_calls, 1)
        import time as _time

        mem._dense_skip_until = _time.monotonic() - 0.1
        mem.search({"user_id": "u", "query": "hello two", "top_k": 3})
        self.assertEqual(breaker.query_calls, 2, "after cooldown dense must be retried")

    def test_pool_cap_preserves_tail(self):
        mem = self._mem(dense_rrf_pool_cap=1)
        good = CountingEmbedder()
        mem.set_dense_embedder(good)
        for i in range(6):
            self._add(mem, f"u{i}", f"r{i}", f"unique content alpha{i} beta{i} for search {i}\n", i)
        hits = mem.search({"user_id": "u0", "query": "alpha0 beta0", "top_k": 10})
        ids = [h["id"] for h in hits]
        self.assertTrue(hits)
        self.assertEqual(len(ids), len(set(ids)), "no duplicate rows")
        contents = "\n".join(h["content"] for h in hits)
        self.assertIn("alpha0", contents, "gold content must be present")
        total_rows = mem.search({"user_id": "u0", "query": "alpha0 beta0", "top_k": 100})
        self.assertGreaterEqual(len(total_rows), 1)

    def test_untouched_when_dense_off(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "m.sqlite3"
        cfg = dict(NFBF)
        self.assertFalse(cfg.get("enable_dense_rrf", False))
        mem = Memory(path, cfg)
        self._add(mem, "u", "r1", "def hello():\n    return 1\n", 1)
        mem.set_dense_embedder(ExplodingEmbedder())
        hits = mem.search({"user_id": "u", "query": "hello", "top_k": 3})
        self.assertTrue(hits)


if __name__ == "__main__":
    unittest.main()
