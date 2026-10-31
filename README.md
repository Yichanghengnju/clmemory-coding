# clmemory-coding v1.0

**A source-preserving, event-only MemMachine adaptation for the Agent Memory
Leaderboard Coding track.** It directly uses MemMachine's native
event storage and BM25/dense/RRF components, with separate delivery policies.
Read [the attribution](THIRD_PARTY_NOTICES.md) and [exact changes and limits](docs/provenance.md).

Add stores original messages and their source roles/timestamps. Search returns
source-bound evidence, never a final answer or code patch. Add uses zero
generative LLM calls. Search applies a final `gpt-4o-mini` selection pass for
relevance, entity/time/state and safety/privacy after every retrieval channel.
Embeddings are `text-embedding-v4`, 1024 dimensions.
No official score is claimed.

## Install

Use a dedicated Python 3.12+ environment on Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[test]' -r requirements-native.txt
.venv/bin/python -m pytest -q
```

The native dependency is pinned to upstream MemMachine commit
`d5a39a8e774e977fb1e6c22efb03555c4e51e05e` (Apache-2.0).
Synthetic tests use a deterministic test embedder and make no external calls.
The deployment command always requires the real embedding profile; it has no
synthetic or alternate-model fallback.

## Configure and run

Create a private `embedding.local.json` from `embedding.example.json`; set the
official DashScope workspace endpoint, external `api_key_file`, and a private
cache directory. Create a separate random service-key file for AML; it is not
the DashScope key or the platform-issued Eval Key. Never commit any key or local
profile. The `.example` file is a template and does not contain a usable endpoint.
Create a private filter profile from `filter.example.json` with the model-provider
key, private cache and an OS lock path shared between both track services. The
CLI requires the final filter; it never silently switches to unfiltered results.

```sh
.venv/bin/python -m agent_memory.submission --track coding \
  --db run/coding/memory.sqlite3 --api-key-file run/coding/api.key \
  --embedding-profile embedding.local.json --filter-profile filter.local.json \
  --manifest-dir run/coding \
  --host 0.0.0.0 --port 8000
```

This repository is the independent Coding submission package. Its common
Add/Search implementation is reused from [clmemory@41ebcfb](https://github.com/Yichanghengnju/clmemory/tree/41ebcfb4467a037d519fa1e61a8e2a685769f195).
We disclose that shared origin; this is not a claim of a second independently
invented retrieval algorithm. Coding uses its own database, vector indexes,
embedding cache, filter cache, service key and application version
`v1.0-coding`. Textual research data, trajectories, scores and learned strategies
are not packaged here. The common implementation still accepts both profile
names for source compatibility; this repository registers only Coding.

The filter cache is also track-specific. Filter model calls serialize through
the shared OS lock. The initial review configuration permits 200 physical mini
requests per track, including failures; successful exact inputs are cached.
That is a smoke/access-review budget, not a Full-evaluation capacity claim.
Use one worker per track; native index and source delivery use per-user
cross-process locks. Initial AML settings: Add concurrency=1, Search concurrency=1.
Maximum accepted top_k is 100; fewer unique evidence items may be returned.
An unknown user returns an empty data array.

Routes: unauthenticated `GET /health`, authenticated `POST /add` and `POST /search`.
Authentication accepts `Authorization: Token`, `Authorization: Bearer`, or
`X-Api-Key`. An identical successful Add is reusable; changed payload under the
same user/request ID returns 409. Successful Add is persisted and searchable
before acknowledgement. A version manifest is written at startup, without keys.

Deploy behind HTTPS for encrypted transport where available. If direct HTTP is
used for initial access review, disclose that in the application; the public
listener still requires a dedicated service key. Capacity is constrained by
host storage and embedding quota; no Full-scale throughput is claimed from
synthetic checks. Known transient embedding responses have at most three
physical attempts; uncertain transport is held and requires operator review.
Do not erase its cache or retry under a fresh identity to bypass that hold.

The final filter preserves the order and exact body of each retained item. It
records all keep/drop reasons in its private ledger. Common credential patterns
are blocked before the selection-model request. The model separately checks
relevance and unauthorized disclosure/injection, with explicit controls for
actual facts, historical questions and questions about assistant suggestions.
It is a fallible classifier, not a general privacy or safety guarantee. Detection
of common secrets does not detect every personal datum and does not retroactively
change what was stored or embedded during Add. Benign medical/personal facts and
ordinary coding commands are not blanket-denied. Uncertain relevance may be kept.
Classifier failures, invalid decisions, quota exhaustion and oversized inputs
return service errors; they never become a successful empty retrieval. The
single selection request has a 90,000 UTF-8-byte payload bound and 8,192-token
output bound, and requires the declared schema without source rewriting.

The platform controls source chunking, answering and scoring. There are no
benchmark data, labels, judge prompts, research runners, or private logs here.

Contact: Chang-Heng Yi, Nanjing University / Cllab. Apache-2.0; see LICENSE and NOTICE.


## nfbf submission recipes (v0)

The current smoke-passed submission recipes live in [`nfbf/`](nfbf/README.md):
a self-contained nf/bf blend search package (agentic grep, guarded dense-RRF
channel, verbatim-source delivery) with both submit configs and unit tests.
Version selection (2026-10-31, pre-registered paired panel over a frozen
314-case retrieval set): dense channel −0.96pp hit@10 — **nfbf-agentic is the
recommended submission face**; the dense variant is published as the guarded
optional channel. File hashes in `source-export-nfbf.json`.
