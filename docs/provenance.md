# Method provenance and claim boundaries

clmemory-coding v1.0 is the Coding package of clmemory, an **engineering adaptation and delivery extension of
MemMachine**, submitted as such. It is not presented as a new retrieval core.

| Component | Actual origin | What this release changes |
| --- | --- | --- |
| Event encoding, segment/vector storage, USearch | Direct MemMachine runtime | Explicit event-only setup; no semantic-profile, summary or retrieval-agent model |
| Dense candidate retrieval, BM25 and RRF | Direct MemMachine runtime | RRF k=60; 16-result head; an independent deeper retrieval supplies only later slots |
| Immutable original-message authority | clmemory adapter | Stable IDs, complete user ownership binding, committed-request visibility, source hydration |
| Add/Search web interface | clmemory adapter using FastAPI | Official synchronous schema, authentication, replay/conflict behavior, response-count enforcement |
| Embedding client | clmemory adapter using DashScope | text-embedding-v4, 1024 dimensions, document/query roles, L2 normalization, durable exact-batch cache and bounded known-failure attempts |
| Coding evidence delivery | clmemory delivery policy | Preserve first 32; shape ranks 33..100 by explicit source file headers, cap retained tail at 2/file, backfill siblings deterministically |
| Source rendering | clmemory adapter; role/time context is also present upstream | Exact original body with explicit source/role/time header; unknown dates remain unknown |
| Final result filter | clmemory selection policy using GPT-4o-mini | After all channels, source-preserving relevance/entity/time/state and safety/privacy selection; private per-item decisions, pre-model common-credential guard, no unfiltered fallback on failure |

The native head is obtained with top_k=min(request top_k,16), which generates
up to 64 vector candidates through overfetch=4. If slots remain, a second native
retrieval at the requested bound (up to 100; at most 400 vector candidates)
supplies the tail. The second retrieval reuses the identical cached query
embedding and never replaces or reranks the initial head. This describes the
release integration precisely; it is not a claim that all retrieval uses a
16-candidate cap. Coding first-32 preservation applies after head/tail assembly.

Coding file shaping applies only to an explicit `[repo/path:start-end]` source
header. When absent or ambiguous, the item passes through unchanged. No file
identity or benchmark annotation is inferred. Sibling backfill can reintroduce
demoted items; the 2/file cap applies to retained tail items, not globally.

This is a first reproducible service integration. It has no official score and
no claim of superiority over MemMachine. Synthetic contract tests establish
engineering behavior only. Prior local experiments used other snapshots,
models and panels; their scores are deliberately not transferred to this
release. Textual and Coding have separate source stores, vector indexes,
embedding caches and API credentials. A result in one is not evidence in the other.

Known limits: the final filter makes a fallible LLM-based evidence-selection
judgment; it does not materialize a persistent resolved current-state database.
There is no new learned reranker, no answer
or patch generation; source messages are not rewritten. The release does not
claim component-level novelty for BM25/dense/RRF, context windows or deduplication.
The final filter is also a standard model-based selection design, not a claim
of a new filtering algorithm or an experimentally established safety benefit.

`source-export.json` records the precise allowlisted in-house source snapshot.
All project-specific original files in this release are licensed Apache-2.0;
upstream components retain their original authorship and license.

## Shared code and independent Coding registration

This Coding package reuses the exact common runtime source from
[clmemory commit 41ebcfb4467a037d519fa1e61a8e2a685769f195](https://github.com/Yichanghengnju/clmemory/tree/41ebcfb4467a037d519fa1e61a8e2a685769f195).
The separate repository records the Coding deployment, documentation and
registration boundary. It is not presented as an unrelated new method.
Only the Coding profile is registered from this repository. No Textual datasets,
experiment results, teacher trajectories or learned skills are transferred.
