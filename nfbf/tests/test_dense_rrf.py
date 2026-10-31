"""Unit tests for dense RRF scaffolding (fake embedder; no live API)."""

import json
import tempfile
import unittest
from pathlib import Path

from coding_memory.core import Memory
from coding_memory.dense_rrf import cosine, dense_applies_for_profile, fuse_bm25_dense, rrf_fuse

ROOT = Path(__file__).resolve().parents[1]
NFBF = json.loads((ROOT / "configs/coding-submit-nf-bf-v0.json").read_text(encoding="utf-8"))


class FakeEmbedder:
    """Deterministic bag-of-char vectors; no network."""

    def __init__(self, dims: int = 8):
        self.dims = dims
        self.call_count = 0
        self.texts_embedded = 0

    def _vec(self, text: str):
        v = [0.0] * self.dims
        for i, ch in enumerate(text.casefold()):
            v[i % self.dims] += (ord(ch) % 31) / 31.0
        # crude L2
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        self.call_count += 1
        self.texts_embedded += len(texts)
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        self.call_count += 1
        self.texts_embedded += 1
        return self._vec(text)


class DenseRRFUnitTests(unittest.TestCase):
    def test_config_default_off(self):
        self.assertFalse(NFBF.get("enable_dense_rrf", False))
        self.assertFalse(NFBF.get("embedding_calls", True))

    def test_rrf_fuse_prefers_consensus(self):
        a = {"x": 1, "y": 2, "z": 3}
        b = {"y": 1, "x": 3, "z": 2}
        fused = rrf_fuse([a, b], k=60)
        ordered = sorted(fused, key=lambda i: (-fused[i], i))
        self.assertEqual(ordered[0], "y")

    def test_fuse_bm25_dense_reorders(self):
        bm25 = ["old", "new"]
        dense = {"new": 0.99, "old": 0.01}
        fused = fuse_bm25_dense(bm25, dense, rrf_k=60)
        self.assertEqual(fused[0][0], "new")

    def test_cosine_identical(self):
        v = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(cosine(v, v), 1.0)

    def test_memory_default_off_ignores_embedder(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "m.sqlite3"
        cfg = dict(NFBF)
        self.assertFalse(cfg.get("enable_dense_rrf", False))
        mem = Memory(path, cfg)
        mem.add(
            {
                "user_id": "u",
                "request_id": "r1",
                "session_id": "s",
                "messages": [{"role": "user", "content": "def hello():\n    return 1\n", "timestamp": 1}],
            }
        )
        # Inject a broken embedder; default-off must not call it.
        mem._dense_embedder = object()  # type: ignore[attr-defined]
        hits = mem.search({"user_id": "u", "query": "hello return", "top_k": 3})
        self.assertTrue(hits)

    def test_memory_dense_on_with_fake_embedder(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "m2.sqlite3"
        cfg = dict(NFBF, enable_dense_rrf=True, embedding_calls=True)
        mem = Memory(path, cfg)
        fake = FakeEmbedder()
        mem.set_dense_embedder(fake)
        mem.add(
            {
                "user_id": "u",
                "request_id": "r-beijing",
                "session_id": "s",
                "messages": [
                    {
                        "role": "user",
                        "content": "User lives in Beijing office tower A.",
                        "timestamp": 1,
                    }
                ],
            }
        )
        mem.add(
            {
                "user_id": "u",
                "request_id": "r-shanghai",
                "session_id": "s",
                "messages": [
                    {
                        "role": "user",
                        "content": "User moved to Shanghai campus building B.",
                        "timestamp": 2,
                    }
                ],
            }
        )
        hits = mem.search({"user_id": "u", "query": "Where does the user live now Shanghai", "top_k": 5})
        self.assertTrue(hits)
        self.assertGreaterEqual(fake.call_count, 1)
        blob = "\n".join(h["content"] for h in hits)
        self.assertTrue("Shanghai" in blob or "Beijing" in blob)


    def test_dense_applies_profiles_default_skips_bf(self):
        cfg = dict(NFBF, enable_dense_rrf=True)
        self.assertTrue(dense_applies_for_profile(cfg, "nf"))
        self.assertTrue(dense_applies_for_profile(cfg, "blend"))
        self.assertFalse(dense_applies_for_profile(cfg, "bf"))
        self.assertFalse(dense_applies_for_profile(dict(NFBF), "nf"))  # default off

    def test_bf_profile_skips_dense(self):
        """Runtime dense on + forced bf must not call embedder (anti BF-hurt gate)."""
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "m-bf.sqlite3"
        cfg = dict(
            NFBF,
            enable_dense_rrf=True,
            embedding_calls=True,
            search_profile="bf",
        )
        mem = Memory(path, cfg)
        fake = FakeEmbedder()
        mem.set_dense_embedder(fake)
        mem.add(
            {
                "user_id": "u",
                "request_id": "r1",
                "session_id": "s",
                "messages": [
                    {
                        "role": "user",
                        "content": "ValueError traceback in src/config.py strip BOM FAILED",
                        "timestamp": 1,
                    }
                ],
            }
        )
        hits = mem.search(
            {
                "user_id": "u",
                "query": "traceback ValueError BOM fix",
                "top_k": 3,
            }
        )
        self.assertTrue(hits)
        self.assertEqual(fake.call_count, 0)

    def test_nf_profile_runs_dense_when_enabled(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        path = Path(temp.name) / "m-nf.sqlite3"
        cfg = dict(
            NFBF,
            enable_dense_rrf=True,
            embedding_calls=True,
            search_profile="nf",
        )
        mem = Memory(path, cfg)
        fake = FakeEmbedder()
        mem.set_dense_embedder(fake)
        mem.add(
            {
                "user_id": "u",
                "request_id": "r1",
                "session_id": "s",
                "messages": [
                    {
                        "role": "user",
                        "content": "implement feature API endpoint for user profile",
                        "timestamp": 1,
                    }
                ],
            }
        )
        hits = mem.search(
            {
                "user_id": "u",
                "query": "implement feature API profile",
                "top_k": 3,
            }
        )
        self.assertTrue(hits)
        self.assertGreaterEqual(fake.call_count, 1)

    def test_config_apply_profiles_identity(self):
        self.assertFalse(NFBF.get("enable_dense_rrf", False))
        profiles = NFBF.get("dense_rrf_apply_profiles")
        self.assertIsNotNone(profiles)
        self.assertIn("nf", profiles)
        self.assertIn("blend", profiles)
        self.assertNotIn("bf", profiles)


if __name__ == "__main__":
    unittest.main()
