"""Behavior tests use temporary databases and a real HTTP subprocess."""

import copy
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from coding_memory.core import Memory, RequestConflict

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/baseline.json").read_text(encoding="utf-8"))


def request(user="repository-A", request_id="history-1", text="Fix parseConfig in src/config.py: reject malformed headers."):
    return {"user_id": user, "request_id": request_id, "session_id": "past-task-1",
            "messages": [{"role": "user", "content": text, "timestamp": None}]}


def query(user="repository-A", text="parseConfig malformed header", top_k=100):
    return {"user_id": user, "query": text, "top_k": top_k}


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "memory.sqlite3"
        self.memory = Memory(self.path, CONFIG)

    def tearDown(self):
        self.temp.cleanup()

    def test_atomic_idempotency_and_conflict(self):
        payload = request()
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(self.memory.add, [payload] * 8))
        self.assertTrue(all(r["success"] for r in results))
        with self.memory.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM requests").fetchone()[0], 1)
            self.assertEqual(db.execute("SELECT count(*) FROM sources").fetchone()[0], 1)
        with self.assertRaises(RequestConflict):
            self.memory.add(request(text="Different payload"))
        self.assertEqual(len(self.memory.search(query())), 1)

    def test_user_isolation_includes_ranking_statistics(self):
        self.memory.add(request())
        before = self.memory.search(query())
        for i in range(10):
            self.memory.add(request(user="repository-A-extra", request_id=str(i), text="parseConfig " * 50))
        self.assertEqual(before, self.memory.search(query()))
        self.assertEqual(self.memory.search(query(user="repository")), [])

    def test_raw_chunk_provenance_is_exact_and_complete(self):
        text = ("def parseConfig(value):\n    return '原文 不改写'\n" * 150) + "TAIL"
        self.memory.add(request(text=text))
        with self.memory.connect() as db:
            source = db.execute("SELECT * FROM sources").fetchone()
            rows = db.execute("SELECT * FROM chunks ORDER BY start_char").fetchall()
        self.assertEqual(source["content"], text)
        covered = set()
        for row in rows:
            covered.update(range(row["start_char"], row["end_char"]))
        self.assertEqual(len(covered), len(text))
        for hit in self.memory.search(query()):
            header, body = hit["content"].split("\n", 1)
            metadata = json.loads(header[len("[historical source "):-1])
            self.assertEqual(body, text[metadata["start_char"]:metadata["end_char"]])
            self.assertEqual(metadata["role"], "user")
            self.assertIsNone(metadata["timestamp"])

    def test_reopen_and_search_change_reuse_sources(self):
        self.memory.add(request())
        before = self.memory.search(query())
        other = Memory(self.path, dict(CONFIG, search_version="diagnostic-only"))
        self.assertEqual(before, other.search(query()))
        with self.assertRaises(ValueError):
            Memory(self.path, dict(CONFIG, add_version="incompatible"))

    def test_unrelated_noise_does_not_outrank_symbol(self):
        self.memory.add(request())
        self.memory.add(request(request_id="noise", text="Update CSS colors for navbar theme and footer."))
        hits = self.memory.search(query())
        self.assertEqual(len(hits), 1)
        self.assertIn("src/config.py", hits[0]["content"])
        self.assertEqual(self.memory.search(query(text="UnseenUniqueSymbol987")), [])
        self.assertEqual(self.memory.search(query(top_k=0)), [])

    def test_concurrent_distinct_requests_are_retained(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(self.memory.add, [request(request_id=str(i), text=f"parseConfig case {i}") for i in range(12)]))
        self.assertEqual(len(self.memory.search(query())), 12)
        self.assertEqual(len(self.memory.search(query(top_k=3))), 3)


class HTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.key = secrets.token_urlsafe(36)
        (cls.directory / "api.key").write_text(cls.key)
        cls.env = dict(os.environ, CODING_CONFIG=str(ROOT / "configs/baseline.json"),
                       CODING_KEY_FILE=str(cls.directory / "api.key"),
                       CODING_DB=str(cls.directory / "memory.sqlite3"), PYTHONPATH=str(ROOT))
        cls.log = (cls.directory / "server.log").open("wb")
        cls.start_server()

    @classmethod
    def start_server(cls):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            cls.port = sock.getsockname()[1]
        cls.base = f"http://127.0.0.1:{cls.port}"
        cls.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "coding_memory.api:create_app", "--factory",
             "--host", "127.0.0.1", "--port", str(cls.port), "--no-access-log"],
            cwd=ROOT, env=cls.env, stdout=cls.log, stderr=cls.log)
        for _ in range(100):
            try:
                with urllib.request.urlopen(cls.base + "/health", timeout=1):
                    return
            except (OSError, urllib.error.URLError):
                if cls.process.poll() is not None:
                    raise RuntimeError("HTTP test server exited")
                time.sleep(0.05)
        raise RuntimeError("HTTP test server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.process.terminate()
        cls.process.wait(timeout=10)
        cls.log.close()
        cls.temp.cleanup()

    def call(self, route, payload, auth=True):
        headers = {"Content-Type": "application/json"}
        if auth:
            headers["Authorization"] = "Bearer " + self.key
        req = urllib.request.Request(self.base + route, json.dumps(payload).encode(), headers)
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    def test_http_authorization_and_validation(self):
        self.assertEqual(self.call("/add", request(), auth=False)[0], 401)
        bad = request()
        bad["gold"] = "must not be accepted"
        code, body = self.call("/add", bad)
        self.assertEqual(code, 422)
        self.assertNotIn("must not be accepted", json.dumps(body))
        self.assertEqual(self.call("/search", dict(query(), top_k=True))[0], 422)
        self.assertEqual(self.call("/search", dict(query(), top_k=101))[0], 422)

    def test_http_immediate_visibility_and_conflict(self):
        payload = request(user="http-user")
        code, receipt = self.call("/add", payload)
        self.assertEqual(code, 200)
        self.assertEqual(receipt, dict(success=True, **{k: payload[k] for k in ("user_id", "request_id", "session_id")}))
        code, body = self.call("/search", query(user="http-user"))
        self.assertEqual(code, 200)
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(set(body["data"][0]), {"id", "content", "score"})
        self.assertEqual(self.call("/add", request(user="http-user", text="changed"))[0], 409)
        self.assertEqual(self.call("/search", query(user="http-user-other"))[1], {"data": []})

    def test_http_restart_preserves_results(self):
        self.call("/add", request(user="restart-user"))
        before = self.call("/search", query(user="restart-user"))[1]
        self.__class__.process.terminate()
        self.__class__.process.wait(timeout=10)
        self.__class__.start_server()
        after = self.call("/search", query(user="restart-user"))[1]
        self.assertEqual(before, after)
        self.assertEqual(self.call("/add", request(user="restart-user"))[0], 200)


if __name__ == "__main__":
    unittest.main()
