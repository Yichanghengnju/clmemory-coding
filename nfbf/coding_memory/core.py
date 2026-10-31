"""Immutable sources and per-user BM25. No models, solver, or benchmark labels."""

import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .dense_rrf import (
    DenseRRFNotAvailable,
    dense_applies_for_profile,
    dense_scores_for_chunks,
    fuse_bm25_dense,
    require_dense_enabled,
)


class RequestConflict(Exception):
    pass


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def tokens(text):
    # Preserve full identifiers and add camelCase/snake_case components.
    result = []
    for word in re.findall(r"[^\W]+", text, re.UNICODE):
        result.append(word.casefold())
        pieces = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word).replace("_", " ").split()
        if len(pieces) > 1:
            result.extend(p.casefold() for p in pieces)
    return result


def windows(text, size, overlap):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            newline = text.rfind("\n", start + size * 3 // 4, end)
            if newline >= 0:
                end = newline + 1
        yield start, end
        if end == len(text):
            break
        start = end - overlap



BF_HINT = re.compile(r"error|traceback|exception|fail|bug|fix|regress|stack", re.I)
NF_HINT = re.compile(r"feature|implement|api|endpoint|add\b|new\b|endpoint|module|function", re.I)
# Paths need a named segment after an explicit root, or multiple relative
# segments. Bare separators and CSS selectors such as .navbar are not anchors.
PATH_RE = re.compile(
    r"(?<![\w./\\])(?:"
    r"(?:[A-Za-z]:[\\/]|\.{1,2}[\\/]|[\\/])(?:[\w.-]+[\\/])*[\w.-]+"
    r"|[\w.-]+(?:[\\/][\w.-]+)+)"
)
ERROR_RE = re.compile(r"\b(?:[A-Z][A-Za-z0-9]*(?:Error|Exception|Failure|Fault)|(?i:Error|Exception))\b")
OUTCOME_RE = re.compile(r"(?i:\b(?:fail(?:ed)?|pass(?:ed)?|assert)\b)")
ANCHOR_HINT = re.compile("|".join(pattern.pattern for pattern in (PATH_RE, ERROR_RE, OUTCOME_RE)))


def shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Memory:
    def __init__(self, db_path, config):
        self.path = Path(db_path)
        self.config = dict(config)
        self._dense_embedder = None
        self._dense_vector_cache = {}
        self._dense_skip_until = 0.0
        self._symbol_side_tables = {}
        if not 0 <= config["overlap_chars"] < config["chunk_chars"] // 2:
            raise ValueError("Invalid chunk overlap")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS requests (
                    user_id TEXT NOT NULL, request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    session_id TEXT NOT NULL, PRIMARY KEY(user_id,request_id));
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, user_id TEXT NOT NULL, request_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, position INTEGER NOT NULL, role TEXT NOT NULL,
                    timestamp INTEGER, content TEXT NOT NULL, content_sha256 TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS sources_user ON sources(user_id);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
                    user_id TEXT NOT NULL, start_char INTEGER NOT NULL, end_char INTEGER NOT NULL,
                    terms TEXT NOT NULL, length INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS chunks_user ON chunks(user_id);
            """)
            identity = canonical({k: config[k] for k in ("add_version", "chunk_chars", "overlap_chars")})
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('add_identity',?)", (identity,))
            if db.execute("SELECT value FROM metadata WHERE key='add_identity'").fetchone()[0] != identity:
                raise ValueError("Incompatible Add identity; use a new database")
            db.commit()

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
        finally:
            db.close()

    def add(self, payload):
        identity = digest(payload)
        user, request = payload["user_id"], payload["request_id"]
        receipt = {k: payload[k] for k in ("request_id", "user_id", "session_id")}
        receipt["success"] = True
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT payload_hash FROM requests WHERE user_id=? AND request_id=?", (user, request)
            ).fetchone()
            if previous:
                if previous[0] != identity:
                    raise RequestConflict("Same user/request_id with different payload")
                return receipt
            for position, message in enumerate(payload["messages"]):
                source_id = "src_" + digest([user, request, position])
                content = message["content"]
                db.execute("INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?)", (
                    source_id, user, request, payload["session_id"], position, message["role"],
                    message.get("timestamp"), content, hashlib.sha256(content.encode("utf-8")).hexdigest(),
                ))
                for start, end in windows(content, self.config["chunk_chars"], self.config["overlap_chars"]):
                    terms = Counter(tokens(content[start:end]))
                    db.execute("INSERT INTO chunks VALUES (?,?,?,?,?,?,?)", (
                        "mem_" + digest([source_id, start, end]), source_id, user,
                        start, end, canonical(terms), sum(terms.values()),
                    ))
            db.execute("INSERT INTO requests VALUES (?,?,?,?)", (user, request, identity, payload["session_id"]))
            db.commit()
        return receipt

    def _resolve_profile(self, payload) -> str:
        profile = str(self.config.get("search_profile", "baseline")).lower()
        if profile in {"nf", "bf", "baseline"}:
            return profile
        # blend / auto: keyword route, else blend
        text = payload.get("query", "")
        opts = payload.get("options") or []
        blob = " ".join([text, *opts])
        bf = bool(BF_HINT.search(blob))
        nf = bool(NF_HINT.search(blob))
        if bf and not nf:
            return "bf"
        if nf and not bf:
            return "nf"
        return "blend"

    def _prefix_soft_hits(self, query_terms, terms) -> int:
        """Count query terms (len>=prefix_min_len) that share a prefix with a chunk term.

        Soft hits enter the candidate pool even without exact BM25 term overlap.
        Exact matches are excluded here (they already contribute via BM25).
        """
        min_len = int(self.config.get("prefix_min_len", 4))
        prefixes = Counter(t[:min_len] for t in terms if len(t) >= min_len)
        return sum(
            1 for q in query_terms
            if len(q) >= min_len and prefixes.get(q[:min_len], 0) > int(q in terms)
        )

    def _prefix_bonus(self, query_terms, terms) -> float:
        boost = float(self.config.get("prefix_boost", 0.0))
        if boost <= 0:
            return 0.0
        return boost * self._prefix_soft_hits(query_terms, terms)

    def _anchor_bonus(self, body: str) -> float:
        boost = float(self.config.get("bf_anchor_boost", 0.25))
        if boost <= 0:
            return 0.0
        return boost if ANCHOR_HINT.search(body) else 0.0

    def _chunk_jaccard(self, row_a, row_b, source_bodies) -> float:
        body_a = source_bodies[row_a["source_id"]][row_a["start_char"] : row_a["end_char"]]
        body_b = source_bodies[row_b["source_id"]][row_b["start_char"] : row_b["end_char"]]
        a, b = set(tokens(body_a)), set(tokens(body_b))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def _mmr_similarity(self, items, source_bodies):
        """Tokenize each candidate once for this selection, without changing sets."""
        term_sets = {
            row["id"]: set(tokens(source_bodies[row["source_id"]][row["start_char"]:row["end_char"]]))
            for item in items for row in (item[1],)
        }

        def similarity(row_a, row_b):
            a, b = term_sets[row_a["id"]], term_sets[row_b["id"]]
            if not a or not b:
                return 0.0
            intersection = len(a & b)
            return intersection / (len(a) + len(b) - intersection)

        return similarity

    def _mmr_select(self, ranked, source_bodies, top_k):
        # Maintain exactly the same maximum redundancy as the original full
        # selected-set scan, updating it once when a new item is selected.
        lam = float(self.config.get("mmr_lambda", 0.7))
        echo_sim = float(self.config.get("mmr_echo_sim", 0.75))
        selected = []
        remaining = list(ranked)
        similarity = self._mmr_similarity(remaining, source_bodies)
        redundancy = {row["id"]: 0.0 for _, row in remaining}
        while remaining and len(selected) < top_k:
            best_i, best_score = None, None
            for i, (score, row) in enumerate(remaining):
                red = redundancy[row["id"]]
                if not selected:
                    mmr = score
                else:
                    if echo_sim > 0 and red >= echo_sim:
                        continue
                    mmr = lam * score - (1.0 - lam) * red * (score if score else 1.0)
                if best_score is None or mmr > best_score:
                    best_score, best_i = mmr, i
            if best_i is None:
                break
            chosen = remaining.pop(best_i)
            selected.append(chosen)
            for _, row in remaining:
                redundancy[row["id"]] = max(redundancy[row["id"]], similarity(row, chosen[1]))
        return selected

    def _mmr_select_expanded(self, expanded, source_bodies, top_k):
        """P0-4: after neighbor expand, keep distinct seeds then MMR-pick others.

        Seeds are preferred but still subject to mmr_echo_sim so repeated seed
        windows cannot occupy every top_k slot. Remaining slots use soft MMR
        (mmr_lambda) with the same hard echo ceiling.
        """
        lam = float(self.config.get("mmr_lambda", 0.7))
        echo_sim = float(self.config.get("mmr_echo_sim", 0.75))
        seeds = [item for item in expanded if item[2] == "seed"]
        others = [item for item in expanded if item[2] != "seed"]
        selected = []
        similarity = self._mmr_similarity(expanded, source_bodies)
        for item in seeds:
            if len(selected) >= top_k:
                break
            score, row, kind = item
            if selected and echo_sim > 0:
                red = max(similarity(row, srow) for _, srow, _ in selected)
                if red >= echo_sim:
                    continue
            selected.append(item)
        # Always retain the top seed even if somehow skipped (should not happen).
        if seeds and not any(item[2] == "seed" for item in selected):
            selected.insert(0, seeds[0])
            selected = selected[:top_k]
        remaining = list(others)
        redundancy = {
            row["id"]: max((similarity(row, srow) for _, srow, _ in selected), default=0.0)
            for _, row, _ in remaining
        }
        while remaining and len(selected) < top_k:
            best_i, best_score = None, None
            for i, (score, row, kind) in enumerate(remaining):
                red = redundancy[row["id"]]
                if echo_sim > 0 and red >= echo_sim:
                    continue
                mmr = lam * score - (1.0 - lam) * red * (score if score else 1.0)
                if best_score is None or mmr > best_score:
                    best_score, best_i = mmr, i
            if best_i is None:
                break
            chosen = remaining.pop(best_i)
            selected.append(chosen)
            for _, row, _ in remaining:
                redundancy[row["id"]] = max(redundancy[row["id"]], similarity(row, chosen[1]))
        return selected


    def set_dense_embedder(self, embedder):
        """Inject embedder (tests/probes). Live path may lazy-build BailianEmbedder."""
        self._dense_embedder = embedder

    def _get_dense_embedder(self):
        if self._dense_embedder is not None:
            return self._dense_embedder
        if not require_dense_enabled(self.config):
            return None
        try:
            from .bailian_embed import BailianEmbedder, profile_from_coding_config
        except ImportError as exc:  # pragma: no cover
            raise DenseRRFNotAvailable("bailian_embed unavailable") from exc
        profile_path = self.config.get("embedding_profile")
        if profile_path:
            profile = profile_path
        else:
            profile = profile_from_coding_config(self.config)
        try:
            self._dense_embedder = BailianEmbedder(profile, user_confirmed=True)
        except Exception as exc:
            raise DenseRRFNotAvailable(
                "dense RRF requested but embedder could not start: " + type(exc).__name__
            ) from exc
        return self._dense_embedder

    def _apply_dense_rrf(self, ranked, source_bodies, query_text):
        """Re-rank BM25 list via RRF with dense cosine. Mutates order/scores only.

        Only the head (dense_rrf_pool_cap, default 96) is sent to the embedder;
        the untouched tail is appended back in its original BM25 order so the
        result never loses rows and never embeds the whole corpus.
        """
        if not ranked:
            return ranked
        embedder = self._get_dense_embedder()
        if embedder is None:
            raise DenseRRFNotAvailable("enable_dense_rrf set but no embedder")
        cap = max(1, int(self.config.get("dense_rrf_pool_cap", 96)))
        head = ranked[:cap]
        tail = ranked[cap:]
        chunk_texts = {}
        for _score, row in head:
            body = source_bodies[row["source_id"]][row["start_char"] : row["end_char"]]
            if not body.strip():
                body = " "
            chunk_texts[row["id"]] = body
        q_vec = embedder.embed_query(query_text)
        dense = dense_scores_for_chunks(
            q_vec,
            chunk_texts,
            embedder.embed_documents,
            vector_cache=self._dense_vector_cache,
        )
        head_ids = [row["id"] for _s, row in head]
        row_by_id = {row["id"]: (score, row) for score, row in head}
        fused = fuse_bm25_dense(
            head_ids,
            dense,
            rrf_k=int(self.config.get("dense_rrf_k", 60)),
            bm25_weight=float(self.config.get("dense_rrf_bm25_weight", 1.0)),
            dense_weight=float(self.config.get("dense_rrf_dense_weight", 1.0)),
        )
        out = []
        seen = set()
        for doc_id, rrf_score in fused:
            if doc_id not in row_by_id:
                continue
            seen.add(doc_id)
            _old, row = row_by_id[doc_id]
            out.append((rrf_score, row))
        for score, row in tail:
            if row["id"] not in seen:
                out.append((score, row))
        return out

    def _apply_dense_rrf_guarded(self, ranked, source_bodies, query_text, trace=None):
        """Dense RRF with graceful degradation: any embedder failure returns the
        BM25 ranking unchanged and suppresses further dense attempts for
        dense_error_cooldown_seconds, so a rate-limited provider can never turn
        a search into an error."""
        now = time.monotonic()
        if now < self._dense_skip_until:
            return ranked
        try:
            return self._apply_dense_rrf(ranked, source_bodies, query_text)
        except Exception as exc:  # noqa: BLE001 - degradation must not re-raise
            cooldown = max(1.0, float(self.config.get("dense_error_cooldown_seconds", 60)))
            self._dense_skip_until = now + cooldown
            if trace is not None:
                trace["dense_rrf_error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:200],
                    "cooldown_seconds": cooldown,
                }
            return ranked

    def search(self, payload, *, trace=None):
        profile = self._resolve_profile(payload)
        # options / route_hints: route-only via _resolve_profile; do NOT join into BM25
        # query_text. Optional payload["query_options"] may append to query (default off).
        query_text = payload["query"]
        extra = payload.get("query_options") or []
        if extra:
            query_text = query_text + " " + " ".join(str(x) for x in extra)
        all_query_terms = list(dict.fromkeys(tokens(query_text)))
        query_terms = all_query_terms[: self.config["max_query_terms"]]
        if trace is not None:
            trace.update(
                {
                    "query": payload,
                    "query_terms": query_terms,
                    "query_terms_omitted": all_query_terms[len(query_terms) :],
                    "config": self.config,
                    "search_profile": profile,
                    "candidates": [],
                    "returned": [],
                }
            )
        if not query_terms or payload["top_k"] == 0:
            return []
        with self.connect() as db:
            rows = db.execute(
                """
                SELECT c.*, s.request_id, s.session_id, s.position, s.role, s.timestamp
                FROM chunks c JOIN sources s ON s.id=c.source_id WHERE c.user_id=?
            """,
                (payload["user_id"],),
            ).fetchall()
            source_rows = db.execute(
                "SELECT id, content, session_id, request_id, position, timestamp FROM sources WHERE user_id=?",
                (payload["user_id"],),
            ).fetchall()
        if not rows:
            return []
        source_bodies = {r["id"]: r["content"] for r in source_rows}
        source_meta = {
            r["id"]: {
                "session_id": r["session_id"],
                "request_id": r["request_id"],
                "position": r["position"],
            }
            for r in source_rows
        }
        documents = [(r, json.loads(r["terms"])) for r in rows]
        df = Counter(term for _, terms in documents for term in query_terms if term in terms)
        avgdl = max(1, sum(r["length"] for r in rows) / len(rows))
        k1, b = self.config["bm25_k1"], self.config["bm25_b"]
        ranked = []
        for row, terms in documents:
            score = 0.0
            for term in query_terms:
                tf = terms.get(term, 0)
                if tf:
                    idf = math.log(1 + (len(rows) - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * row["length"] / avgdl))
            if profile in {"nf", "blend"}:
                # P0-2: prefix soft-hit (>=4) contributes score and thus enters pool
                # even when no exact BM25 term matched.
                score += self._prefix_bonus(query_terms, terms)
            if profile in {"bf", "blend"}:
                body = source_bodies[row["source_id"]][row["start_char"] : row["end_char"]]
                score += self._anchor_bonus(body)
            if score > 0:
                ranked.append((score, row))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        if profile in {"bf", "blend"} and ranked:
            pool = ranked[: max(payload["top_k"] * 4, payload["top_k"])]
            ranked = self._mmr_select(pool, source_bodies, max(payload["top_k"] * 2, payload["top_k"]))
            ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        if trace is not None:
            trace.update({"corpus_chunks": len(rows), "avgdl": avgdl, "document_frequency": dict(df)})
            ranks = {row["id"]: i + 1 for i, (_, row) in enumerate(ranked)}
            scores = {row["id"]: score for score, row in ranked}
            trace["candidates"] = [
                {
                    "id": row["id"],
                    "source_id": row["source_id"],
                    "request_id": row["request_id"],
                    "start_char": row["start_char"],
                    "end_char": row["end_char"],
                    "score": scores.get(row["id"], 0.0),
                    "positive_rank": ranks.get(row["id"]),
                    "matched_terms": {q: terms[q] for q in query_terms if q in terms},
                }
                for row, terms in documents
            ]

        # Explicit experimental side table, disabled unless a new configuration
        # enables it. All refs come from this user's immutable source spans.
        if self.config.get("enable_symbol_side_table", False):
            from .coding_symbols import SymbolSideTable, apply_symbol_side_table
            # Sources/spans are immutable. The complete id tuple invalidates on
            # Add without mixing users or relying on only a corpus row count.
            symbol_identity = (payload["user_id"], tuple(sorted(row["id"] for row, _terms in documents)))
            table = self._symbol_side_tables.get(symbol_identity)
            if table is None:
                table = SymbolSideTable(documents, source_bodies)
                self._symbol_side_tables = {
                    key: value for key, value in self._symbol_side_tables.items()
                    if key[0] != payload["user_id"]
                }
                if len(self._symbol_side_tables) >= 4:
                    self._symbol_side_tables.pop(next(iter(self._symbol_side_tables)))
                self._symbol_side_tables[symbol_identity] = table
            ranked, symbol_trace = apply_symbol_side_table(
                ranked, documents, source_bodies, query_text,
                candidate_cap=int(self.config.get("symbol_candidate_cap", 96)), table=table,
            )
            if trace is not None:
                trace["symbol_side_table"] = symbol_trace

        # Optional dense RRF bypass (default off). BF profile skips dense
        # (A/B: NF+ / BF-); override via dense_rrf_apply_profiles. Guarded:
        # embedder failure degrades to the BM25 order, never an error.
        if dense_applies_for_profile(self.config, profile):
            ranked = self._apply_dense_rrf_guarded(ranked, source_bodies, query_text, trace)

        # Agentic grep (P0-1): narrow BM25/prefix/anchor seed -> one bounded
        # session-level neighbor expand. No LLM, no unbounded corpus sweep.
        enable_agentic = bool(self.config.get("enable_agentic_grep", True))
        if profile == "baseline":
            enable_agentic = False
        radius = 0
        if enable_agentic:
            if profile == "nf":
                radius = int(self.config.get("neighbor_radius_nf", 0))
            elif profile == "bf":
                radius = int(self.config.get("neighbor_radius_bf", 0))
            elif profile == "blend":
                radius = int(self.config.get("neighbor_radius_bf", 2))
        neighbor_budget = int(self.config.get("neighbor_budget", 0)) if enable_agentic else 0

        seeds = []
        for score, row in ranked:
            seeds.append((score, row, "seed"))
            if len(seeds) >= payload["top_k"]:
                break

        expanded = list(seeds)
        if radius > 0 and neighbor_budget > 0:
            # Session turn order within this user: (session_id, turn) — NOT request_id.
            # Cross-Add symptom↔fix in the same session can link via adjacent turns.
            session_turns = {}
            for src in source_rows:
                sid = src["session_id"]
                session_turns.setdefault(sid, []).append(src)
            turn_of = {}
            chunks_by_source = {}
            for row, _terms in documents:
                chunks_by_source.setdefault(row["source_id"], []).append(row)
            for sid, srcs in session_turns.items():
                srcs.sort(
                    key=lambda r: (
                        r["timestamp"] is None,
                        r["timestamp"] if r["timestamp"] is not None else 0,
                        r["request_id"],
                        r["position"],
                        r["id"],
                    )
                )
                for idx, src in enumerate(srcs):
                    turn_of[src["id"]] = (sid, idx)
            used_ids = {row["id"] for _, row, _ in expanded}
            added = 0
            for score, row, _kind in list(seeds):
                loc = turn_of.get(row["source_id"])
                if loc is None:
                    continue
                sid, turn = loc
                ordered = session_turns[sid]
                for delta in range(-radius, radius + 1):
                    if delta == 0:
                        continue
                    nturn = turn + delta
                    if nturn < 0 or nturn >= len(ordered):
                        continue
                    nsrc = ordered[nturn]
                    for nrow in chunks_by_source.get(nsrc["id"], []):
                        if nrow["id"] in used_ids:
                            continue
                        expanded.append((score * 0.5, nrow, "neighbor"))
                        used_ids.add(nrow["id"])
                        added += 1
                        if added >= neighbor_budget:
                            break
                    if added >= neighbor_budget:
                        break
                if added >= neighbor_budget:
                    break
            # Keep seeds first, then neighbors; trim later by top_k / chars
            expanded.sort(key=lambda item: (0 if item[2] == "seed" else 1, -item[0], item[1]["id"]))

        # P0-4 BF/blend: post-expand MMR keeps seeds, de-echoes neighbors (raw-first packaging below).
        if profile in {"bf", "blend"} and len(expanded) > 1:
            expanded = self._mmr_select_expanded(expanded, source_bodies, payload["top_k"])

        hits, used = [], 0
        for score, row, _kind in expanded:
            source = {
                k: row[k]
                for k in (
                    "source_id",
                    "request_id",
                    "session_id",
                    "position",
                    "role",
                    "timestamp",
                    "start_char",
                    "end_char",
                )
            }
            body = source_bodies[row["source_id"]][row["start_char"] : row["end_char"]]
            content = "[historical source " + canonical(source) + "]\n" + body
            if used + len(content) > self.config["max_result_chars"]:
                if _kind == "neighbor":
                    continue
                break
            hits.append({"id": row["id"], "content": content, "score": score})
            used += len(content)
            if len(hits) == payload["top_k"]:
                break
        if trace is not None:
            trace.update(
                {
                    "returned": hits,
                    "returned_chars": used,
                    "positive_candidates": len(ranked),
                    "returned_ids": [h["id"] for h in hits],
                }
            )
        return hits
