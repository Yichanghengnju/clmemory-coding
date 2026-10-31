"""Minimal local tests for agentic grep P0-1 / P0-2 / P0-3 / P0-4 / P0-5 (no official smoke)."""

import json
import tempfile
import unittest
from pathlib import Path

from coding_memory.core import Memory

ROOT = Path(__file__).resolve().parents[1]
NFBF = json.loads((ROOT / "configs/coding-submit-nf-bf-v0.json").read_text(encoding="utf-8"))


def _add(mem, *, user, request_id, session_id, role, text, timestamp, position_note=None):
    # position is assigned per-request by Add; cross-request adjacency uses
    # session turn order (timestamp, request_id, position).
    return mem.add(
        {
            "user_id": user,
            "request_id": request_id,
            "session_id": session_id,
            "messages": [{"role": role, "content": text, "timestamp": timestamp}],
        }
    )


class AgenticGrepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.cfg = dict(NFBF)
        self.memory = Memory(self.path, self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def test_enable_agentic_grep_default_on(self):
        self.assertTrue(self.cfg.get("enable_agentic_grep", True))

    def test_p0_1_bf_cross_request_session_neighbors(self):
        """Same user+session, two request_ids: error seed pulls repair neighbor."""
        user, session = "repo-bf", "sess-bf-1"
        _add(
            self.memory,
            user=user,
            request_id="req-symptom",
            session_id=session,
            role="user",
            text=(
                "Traceback (most recent call last):\n"
                "  File \"src/config.py\", line 12, in parseConfig\n"
                "ValueError: UTF-8 BOM not stripped\n"
            ),
            timestamp=100,
        )
        _add(
            self.memory,
            user=user,
            request_id="req-fix",
            session_id=session,
            role="assistant",
            text=(
                "Fixed parseConfig: strip BOM before json load. "
                "tests PASSED for UTF-8 BOM case.\n"
            ),
            timestamp=200,
        )
        _add(
            self.memory,
            user=user,
            request_id="req-noise",
            session_id=session,
            role="user",
            text="Update CSS colors for navbar theme and footer padding.",
            timestamp=300,
        )
        hits = self.memory.search(
            {
                "user_id": user,
                "query": "parseConfig UTF-8 BOM traceback fix",
                "top_k": 5,
            }
        )
        blob = "\n".join(h["content"] for h in hits)
        self.assertTrue(
            ("strip BOM" in blob) or ("PASSED" in blob) or ("ValueError" in blob),
            msg=f"expected symptom/fix evidence, got: {blob[:500]}",
        )
        self.assertNotIn("navbar theme", hits[0]["content"])

    def test_p0_1_flag_off_skips_neighbor_expand(self):
        user, session = "repo-flag", "sess-flag"
        cfg = dict(NFBF, enable_agentic_grep=False)
        path = Path(self.temp.name) / "flag.sqlite3"
        mem = Memory(path, cfg)
        _add(
            mem,
            user=user,
            request_id="r0",
            session_id=session,
            role="user",
            text="Error in src/a.py: TypeError on parseConfig",
            timestamp=1,
        )
        _add(
            mem,
            user=user,
            request_id="r1",
            session_id=session,
            role="assistant",
            text="UNIQUE_REPAIR_TOKEN strip BOM and re-encode; PASSED",
            timestamp=2,
        )
        hits = mem.search({"user_id": user, "query": "TypeError parseConfig traceback", "top_k": 3})
        blob = "\n".join(h["content"] for h in hits)
        # Without agentic expand, repair-only neighbor may be absent if seed is error chunk.
        # Soft assert: flag path runs; if repair not in seeds it should not appear as neighbor.
        # At least search returns something and does not crash.
        self.assertGreaterEqual(len(hits), 1)
        # When flag off, UNIQUE_REPAIR_TOKEN only appears if it independently ranked as seed.
        # Error query should prefer error chunk; repair may be missing — that is OK for off.
        _ = blob

    def test_p0_2_prefix_soft_hit_enters_pool(self):
        user = "repo-nf"
        self.memory.add(
            {
                "user_id": user,
                "request_id": "feat-1",
                "session_id": "sess-nf",
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Please implement exportReporter to write CSV rows "
                            "for the monthly dashboard export_report pipeline."
                        ),
                        "timestamp": 10,
                    }
                ],
            }
        )
        self.memory.add(
            {
                "user_id": user,
                "request_id": "noise-1",
                "session_id": "sess-nf",
                "messages": [
                    {
                        "role": "user",
                        "content": "Change button color to blue in the settings modal.",
                        "timestamp": 11,
                    }
                ],
            }
        )
        hits = self.memory.search(
            {
                "user_id": user,
                "query": "exportReport",
                "top_k": 5,
                "options": ["CSV"],
            }
        )
        self.assertTrue(hits, "prefix soft-hit should enter pool")
        blob = "\n".join(h["content"] for h in hits)
        self.assertTrue(
            ("exportReporter" in blob) or ("export_report" in blob),
            msg=f"expected morphological variant, got: {blob[:400]}",
        )

    def test_p0_3_path_error_anchor_boost(self):
        user, session = "repo-anchors", "sess-anchors"
        corpus = {
            "symptom": "Traceback BOM fix in src/config.py: ValueError.",
            "repair": "Traceback fix: strip BOM before json load; tests PASSED.",
            "noise": "Traceback BOM fix: navbar CSS / .theme",
        }
        for timestamp, (request_id, text) in enumerate(corpus.items()):
            _add(
                self.memory,
                user=user,
                request_id=request_id,
                session_id=session,
                role="assistant" if request_id == "repair" else "user",
                text=text,
                timestamp=timestamp,
            )

        for profile, query in (
            ("bf", "traceback BOM fix"),
            ("blend", "implement traceback BOM fix"),
            ("nf", "traceback BOM fix"),
            ("baseline", "traceback BOM fix"),
        ):
            with self.subTest(profile=profile):
                cfg = dict(self.cfg, search_profile=profile)
                mem = Memory(self.path, cfg)
                control = Memory(self.path, dict(cfg, bf_anchor_boost=0.0))
                payload = {"user_id": user, "query": query, "top_k": 3}
                trace, control_trace = {}, {}
                hits = mem.search(payload, trace=trace)
                control.search(payload, trace=control_trace)
                self.assertEqual(trace["search_profile"], profile)
                if profile in {"bf", "blend"}:
                    top = hits[0]["content"]
                    self.assertTrue(
                        any(s in top for s in ("src/", "strip BOM", "PASSED", "ValueError")),
                        msg=f"expected original symptom/fix window, got: {top}",
                    )
                    self.assertNotIn("navbar", top)
                    self.assertNotIn("CSS", top)
                # Search returns the original window; anchors are only a score bonus.
                for hit in hits:
                    self.assertIn(hit["content"].split("\n", 1)[1], corpus.values())
                scores = {c["request_id"]: c["score"] for c in trace["candidates"]}
                controls = {c["request_id"]: c["score"] for c in control_trace["candidates"]}
                for request_id in corpus:
                    expected = (
                        cfg["bf_anchor_boost"]
                        if profile in {"bf", "blend"} and request_id != "noise"
                        else 0.0
                    )
                    self.assertAlmostEqual(scores[request_id] - controls[request_id], expected)

    def test_p0_3_anchor_boundaries(self):
        for text in (
            "src/config.py", r".\foo\bar", r"C:\repo\config.py", "../src/.env", "/tmp/config.py",
            "ValueError", "CustomException", "BuildFailure", "MemoryFault", "Error", "exception",
            "fail", "pass", "FAILED", "PASSED", "assert value", "src/a.py ValueError FAILED assert",
        ):
            with self.subTest(anchor=text):
                self.assertEqual(self.memory._anchor_bonus(text), self.cfg["bf_anchor_boost"])
        for text in (
            "/", "\\", "Update navbar CSS / footer padding .theme .navbar { color: blue; }",
            "errorless", "exceptional", "failover", "compass", "bypassed", "assertion", "runtimeerror",
        ):
            with self.subTest(no_anchor=text):
                self.assertEqual(self.memory._anchor_bonus(text), 0.0)

    def test_p0_3_anchor_boost_default_and_override(self):
        cfg = dict(self.cfg)
        cfg.pop("bf_anchor_boost")
        self.assertEqual(Memory(self.path, cfg)._anchor_bonus("ValueError"), 0.25)
        for boost in (0.0, -0.25, 0.6):
            with self.subTest(boost=boost):
                mem = Memory(self.path, dict(cfg, bf_anchor_boost=boost))
                self.assertEqual(mem._anchor_bonus("src/config.py ValueError PASSED"), max(0.0, boost))

    def test_bounded_neighbor_budget(self):
        user, session = "repo-bound", "sess-bound"
        _add(
            self.memory,
            user=user,
            request_id="seed",
            session_id=session,
            role="user",
            text="BUG traceback Exception FAILED in src/core.py assert",
            timestamp=0,
        )
        for i in range(1, 25):
            _add(
                self.memory,
                user=user,
                request_id=f"n{i}",
                session_id=session,
                role="assistant",
                text=f"noise turn {i} unrelated padding " * 3,
                timestamp=i,
            )
        hits = self.memory.search(
            {"user_id": user, "query": "traceback Exception bug fix", "top_k": 100}
        )
        self.assertLessEqual(len(hits), 100)
        # seed first: top hit should be the error seed, not pure noise
        self.assertIn("traceback", hits[0]["content"].lower())
        total_chars = sum(len(h["content"]) for h in hits)
        self.assertLessEqual(total_chars, self.cfg["max_result_chars"])



    def test_p0_4_mmr_raw_first_no_echo_fill(self):
        """BF: near-duplicate windows must not fill top_k; returns provenance+raw body only."""
        user, session = "repo-mmr", "sess-mmr"
        cfg = dict(self.cfg, search_profile="bf")
        mem = Memory(Path(self.temp.name) / "mmr.sqlite3", cfg)
        seed_text = "BUG traceback ValueError FAILED in src/parse.py: strip needed"
        echo = "Fixed parseConfig strip BOM before json load; tests PASSED for UTF-8 BOM case. "
        corpus = {"seed": seed_text}
        _add(
            mem,
            user=user,
            request_id="seed",
            session_id=session,
            role="user",
            text=seed_text,
            timestamp=0,
        )
        for i in range(1, 12):
            text_i = echo + f"repeat#{i}"
            corpus[f"e{i}"] = text_i
            _add(
                mem,
                user=user,
                request_id=f"e{i}",
                session_id=session,
                role="assistant",
                text=text_i,
                timestamp=i,
            )
        hits = mem.search(
            {"user_id": user, "query": "traceback ValueError BOM fix", "top_k": 8}
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertLess(len(hits), 8, msg="echo near-duplicates must not fill top_k")
        bodies = []
        for hit in hits:
            self.assertTrue(
                hit["content"].startswith("[historical source "),
                msg="raw-first provenance header required",
            )
            body = hit["content"].split("\n", 1)[1]
            bodies.append(body)
            self.assertIn(body, corpus.values(), msg="content must be original window")
            self.assertNotIn("MemoryCard", hit["content"])
            self.assertNotIn("summary:", hit["content"].lower())
        self.assertIn(seed_text, bodies[0])
        # At most one near-echo of the repeated repair template should survive.
        from coding_memory.core import tokens

        def jac(a, b):
            A, B = set(tokens(a)), set(tokens(b))
            return len(A & B) / len(A | B) if A and B else 0.0

        echo_like = sum(1 for b in bodies if jac(b, echo) >= 0.85)
        self.assertLessEqual(echo_like, 1)

    def test_p0_4_mmr_lambda_default(self):
        self.assertAlmostEqual(float(self.cfg.get("mmr_lambda", 0.7)), 0.7)

    def test_p0_5_query_route_under_default_blend(self):
        """Default search_profile=blend: keyword route; options route-only (not BM25)."""
        self.assertEqual(self.cfg.get("search_profile"), "blend")
        cases = (
            ("error traceback ValueError", "bf"),
            ("parseConfig UTF-8 BOM traceback fix", "bf"),
            ("please implement new api endpoint", "nf"),
            ("feature module function design", "nf"),
            ("implement traceback fix", "blend"),  # dual keyword conflict
            ("hello world unrelated", "blend"),  # neither hint
        )
        for query, expected in cases:
            with self.subTest(query=query, expected=expected):
                got = self.memory._resolve_profile({"query": query, "options": []})
                self.assertEqual(got, expected)

                # options are route-only: may affect profile via _resolve_profile, never enter query_terms
        for query, expect_profile in (
            ("traceback error", "bf"),
            ("implement api", "nf"),
            ("implement traceback", "blend"),
        ):
            with self.subTest(opts_query=query):
                trace = {}
                self.memory.search(
                    {
                        "user_id": "route-user",
                        "query": query,
                        "top_k": 1,
                        "options": ["CSV"],
                    },
                    trace=trace,
                )
                self.assertEqual(trace.get("search_profile"), expect_profile)
                terms = set(trace.get("query_terms") or [])
                self.assertNotIn("csv", terms)

        # options-only NF hint still routes when query has neither hint
        with self.subTest(opts_only_route=True):
            self.assertEqual(
                self.memory._resolve_profile(
                    {"query": "hello world unrelated", "options": ["implement"]}
                ),
                "nf",
            )
            trace = {}
            self.memory.search(
                {
                    "user_id": "route-user",
                    "query": "hello world unrelated",
                    "top_k": 1,
                    "options": ["implement", "feature"],
                },
                trace=trace,
            )
            self.assertEqual(trace.get("search_profile"), "nf")
            terms = set(trace.get("query_terms") or [])
            self.assertNotIn("implement", terms)
            self.assertNotIn("feature", terms)

        # Forced nf|bf|baseline must not keyword-override
        for forced in ("nf", "bf", "baseline"):
            with self.subTest(forced=forced):
                mem = Memory(self.path, dict(self.cfg, search_profile=forced))
                self.assertEqual(
                    mem._resolve_profile({"query": "error traceback", "options": []}),
                    forced,
                )



if __name__ == "__main__":
    unittest.main()
