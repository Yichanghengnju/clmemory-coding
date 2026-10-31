# nfbf submission recipes (v0, 2026-10-31)

Verbatim source of the two smoke-passed Coding-track submission recipes
(`clmemory-nfbf-agentic-v0-coding` / `version_28077d6d9063` and
`clmemory-nfbf-dense-v0-coding` / `version_5b5460258cf1`). This is a
self-contained successor package: it does not modify or depend on the v1.0
export under `src/agent_memory/`.

- `coding_memory/` — the service package (verbatim internal files):
  `core.py` (nf/bf blend search, agentic grep, dense-v0 guarded RRF with
  fallback + 60s cooldown + candidate cap 96, verbatim-source delivery),
  `dense_rrf.py`, `bailian_embed.py` (stdlib-only DashScope text-embedding-v4 client).
- `configs/` — the two submit configs. They differ only in the dense channel
  (`enable_dense_rrf` + guard keys). `embedding_api_key_file` is scrubbed to a
  placeholder in this publication; deployments must provide their own key file.
- `tests/` — unit tests incl. the dense-fallback suite (5/5) and agentic-grep
  fixtures. Run from the repository root with `nfbf/` importable as
  `coding_memory` (e.g. `python -m pytest nfbf/tests -q` with `nfbf` on
  `sys.path`, or copy the package out).

## Identity

- Open-source Methods track identity: embeddings = text-embedding-v4 (1024d),
  LLM components = gpt-4o-mini, rerank unrestricted.
- Version selection (2026-10-31): a pre-registered paired panel over the frozen
  314-case retrieval set (both configs verbatim, same denominator) measured the
  dense channel at −0.96pp hit@10 (18W/21L/275T) — dense NOT adopted;
  **nfbf-agentic is the recommended submission face**. The dense recipe is kept
  published as the guarded optional channel.
- The v1.0 baseline (`version_60c7d218454f`) remains tagged `v1.0` below.

## Provenance boundary

`coding_memory/` files are published as-run (no research data, no credentials,
no logs). File hashes are recorded in `source-export-nfbf.json`.
