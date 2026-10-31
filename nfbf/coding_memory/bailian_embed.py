"""Lightweight Bailian text-embedding-v4 client for Coding dense RRF.

Credentials: DASHSCOPE_API_KEY env or profile api_key_file. Never log the key.
No generative LLM. Optional in-memory call counter for probes.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Sequence
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

MODEL = "text-embedding-v4"
DIMENSIONS = {64, 128, 256, 512, 768, 1024, 1536, 2048}
DEFAULT_ENDPOINT = (
    "https://ws-12vkr9d2p0p0yxhq.cn-beijing.maas.aliyuncs.com"
    "/api/v1/services/embeddings/text-embedding/text-embedding"
)


def load_embedding_profile(value) -> dict:
    if isinstance(value, (str, Path)):
        profile = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        profile = dict(value)
    if profile.get("model") != MODEL or profile.get("provider") != "bailian":
        raise ValueError("Require Bailian text-embedding-v4 profile")
    if set(profile) & {"api_key", "key", "token", "authorization"}:
        raise ValueError("Credentials must be supplied outside the profile")
    dims = profile.get("dimensions")
    if type(dims) is not int or dims not in DIMENSIONS:
        raise ValueError("Unsupported v4 dimensions")
    return profile


def resolve_api_key(profile: dict) -> str:
    key_file = profile.get("api_key_file")
    if key_file:
        path = Path(str(key_file)).expanduser()
        if path.is_file():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    env = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if env:
        return env
    raise DenseEmbedCredentialError(
        "Bailian credential missing: set DASHSCOPE_API_KEY or profile api_key_file"
    )


class DenseEmbedCredentialError(RuntimeError):
    pass


class BailianEmbedder:
    """Minimal embed_documents / embed_query surface for dense RRF."""

    def __init__(
        self,
        profile,
        *,
        user_confirmed: bool = False,
        timeout: float = 60.0,
        batch_size: int = 10,
        transport=None,
    ):
        if user_confirmed is not True:
            raise PermissionError(
                "Bailian text-embedding-v4 requires explicit user confirmation"
            )
        self.profile = load_embedding_profile(profile)
        self.dimensions = self.profile["dimensions"]
        self.endpoint = self.profile.get("endpoint") or DEFAULT_ENDPOINT
        self.timeout = timeout
        self.batch_size = max(1, min(int(batch_size), 10))
        self._key = resolve_api_key(self.profile)
        self._transport = transport or self._send
        self.call_count = 0
        self.texts_embedded = 0

    def _send(self, request: Request, timeout: float):
        proxy = self.profile.get("proxy")
        opener = build_opener(ProxyHandler({"https": proxy})) if proxy else build_opener()
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def _l2(self, vector: Sequence[float]) -> list[float]:
        if len(vector) != self.dimensions:
            raise ValueError("Bailian returned unexpected dimensions")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in vector):
            raise ValueError("Bailian returned non-finite vector")
        norm = math.sqrt(math.fsum(v * v for v in vector))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Bailian returned a zero vector")
        return [float(v) / norm for v in vector]

    def _encode(self, texts: list[str], kind: str) -> list[list[float]]:
        payload = {
            "model": MODEL,
            "input": {"texts": texts},
            "parameters": {
                "text_type": kind,
                "dimension": self.dimensions,
                "output_type": "dense",
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": "Bearer " + self._key,
                "Content-Type": "application/json",
            },
        )
        status, raw = self._transport(request, self.timeout)
        self.call_count += 1
        self.texts_embedded += len(texts)
        if status != 200:
            raise RuntimeError(f"Bailian HTTP {status}")
        value = json.loads(raw)
        rows = value["output"]["embeddings"]
        if len(rows) != len(texts):
            raise ValueError("Bailian result count differs")
        ordered = sorted(rows, key=lambda r: r["text_index"])
        return [self._l2(row["embedding"]) for row in ordered]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if isinstance(texts, (str, bytes)):
            raise TypeError("Documents must be a sequence of strings")
        texts = list(texts)
        if any(not isinstance(v, str) or not v.strip() for v in texts):
            raise ValueError("Document strings must be nonempty")
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            out.extend(self._encode(texts[i : i + self.batch_size], "document"))
        return out

    def embed_query(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Query must be a nonempty string")
        return self._encode([text], "query")[0]


def profile_from_coding_config(config: dict) -> dict:
    """Build a minimal Bailian profile from coding submit config keys."""
    return {
        "provider": "bailian",
        "model": str(config.get("embedding_model", MODEL)),
        "endpoint": config.get("embedding_endpoint") or DEFAULT_ENDPOINT,
        "dimensions": int(config.get("embedding_dimensions", 1024)),
        "api_key_file": config.get("embedding_api_key_file")
        or "~/.config/agentmemory/credentials/bailian-apikey-7885504.key",
        "proxy": config.get("embedding_proxy"),
    }
