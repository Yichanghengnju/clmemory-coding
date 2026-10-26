# Third-party sources and attribution

## MemMachine: direct runtime reuse

- Original project and authors: [MemMachine / MemMachine contributors](https://github.com/MemMachine/MemMachine).
- Frozen source: [`d5a39a8e774e977fb1e6c22efb03555c4e51e05e`](https://github.com/MemMachine/MemMachine/tree/d5a39a8e774e977fb1e6c22efb03555c4e51e05e).
- License: Apache-2.0; the original license text is retained under `licenses/`.
- We directly call upstream EventMemory, WholeTextDeriver, PassthroughSegmenter,
  SQLAlchemySegmentStore, SQLiteVectorStore, USearchVectorSearchEngine,
  IdentityReranker, BM25Reranker and RRFHybridReranker.
- These components are **not independently reimplemented** and are **not claimed
  as clmemory inventions**. The two pinned Python packages are installed from
  upstream; their source is not vendored in this repository.
- Our adapter and all changes are identified in [provenance](docs/provenance.md).
- This release does not reproduce an identified MemMachine leaderboard deployment
  or claim MemMachine's published scores. No paper is asserted as the source of
  the particular pinned event-only configuration; the cited code and documentation
  are its implementation reference.

## Other methods studied

ReFind, ActiveMemoryIndex, MemFusion, M-Flow and other public memory systems were
studied during development. Their implementations, copied code, judge prompts,
datasets, labels and experimental results are not included in this minimal
release. General techniques (BM25, dense retrieval, RRF, neighbor expansion and
deduplication) are established techniques, not new clmemory algorithms.

## Dependency licenses

The service also imports FastAPI (MIT), Pydantic (MIT) and Uvicorn (BSD-3-Clause),
and the transitive dependencies installed by the pinned MemMachine packages.
Their original distributions and license notices remain authoritative. This
repository's license does not relicense upstream dependencies or data.
