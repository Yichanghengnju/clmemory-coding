"""Standalone, track-scoped deployment entry point for clmemory v1.0."""
import argparse
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path

from .api import create_app
from .bailian_embeddings import BailianEmbeddings, load_profile
from .delivery import SubmissionMemory


def code_manifest():
    root = Path(__file__).parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob('*.py'))}


def build_app(*, db_path, api_key, embedder, track, manifest_dir=None, final_filter=None):
    if not api_key:
        raise ValueError("A nonempty memory system key is required")
    memory = SubmissionMemory(db_path, embedder=embedder, track=track)
    if final_filter is not None:
        from .final_filter import FilteredMemory
        memory = FilteredMemory(memory, final_filter)
    app = create_app(db_path, service=memory, api_key=api_key)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            memory.close()
    app.router.lifespan_context = lifespan
    manifest = {"system": "clmemory", "version": "v1.0-" + track, "track": track,
                "source_files_sha256": code_manifest(), "method": memory.identity,
                "llm_calls_in_add": 0,
                "llm_calls_in_search": "one cached final-filter batch if nonempty" if final_filter else 0,
                "final_filter_enabled": final_filter is not None, "credentials_in_manifest": False,
                "embedding": getattr(embedder, "specification", embedder.identity)}
    app.state.memory_service = memory
    app.state.version_manifest = manifest
    if manifest_dir is not None:
        path = Path(manifest_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'version-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track', required=True, choices=['textual', 'coding'])
    parser.add_argument('--db', required=True)
    parser.add_argument('--api-key-file', required=True)
    parser.add_argument('--embedding-profile', required=True)
    parser.add_argument('--manifest-dir', required=True)
    parser.add_argument('--filter-profile', required=True,
                        help='Private GPT-4o-mini relevance + safety/privacy final-filter profile')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    profile = load_profile(args.embedding_profile)
    if profile['dimensions'] != 1024:
        parser.error('This release is frozen to text-embedding-v4 / 1024 dimensions')
    embedder = BailianEmbeddings(profile, user_confirmed=True)
    from .final_filter import FinalEvidenceFilter
    final_filter = FinalEvidenceFilter(json.loads(Path(args.filter_profile).read_text()), track=args.track)
    app = build_app(db_path=args.db, api_key=Path(args.api_key_file).read_text().strip(),
                    embedder=embedder, track=args.track, manifest_dir=args.manifest_dir, final_filter=final_filter)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, workers=1, limit_concurrency=16,
                timeout_keep_alive=5, log_level='info')


if __name__ == '__main__':
    main()
