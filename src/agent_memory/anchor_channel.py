"""Deterministic anchor-append Search channel (submission recipe, 2026-10-24 user decision).

Port of anchor_spec_v1 from the frozen diagnostic cards (configs/grep-anchor-diag-20261024.json;
tools/grep_anchor_gap_audit_20261024.py extract_anchors), kept byte-equivalent in behavior:
names / dates / numbers / quoted spans from the query, case-insensitive substring matching.
The channel APPENDS verbatim sources into the ranked evidence list (append-not-displace,
E4a lesson); it never removes, rewrites or reranks existing evidence, and calls no model.

This module is the canonical in-repo implementation. The tools/ copies stay as-frozen for
the diagnostic cards' script-sha receipts; any behavioral divergence must bump the spec id.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime

from .domain import Evidence

ANCHOR_SPEC = "v1"

MONTHS = {"january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"}
WEEKDAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
STOP_SINGLE = set("""question what who where when why how did do does is are was were the in on at my i he she
they we you it please tell according which a an and but or if after before during between last next this that
these those there their his her its our your from with about for to of by as so not no yes also some all any
both each many much more most other another same different new old first second third then now here hi hello
okay ok thanks thank hey yeah let us make get got give given take took come came go went see saw say said know
known think thought want wanted need needed""".split())

DATE_RXS = [
    re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b"),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?\b", re.IGNORECASE),
    re.compile(r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\b", re.IGNORECASE),
    re.compile(r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE),
    re.compile(r"\b(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),
]
NUM_RE = re.compile(r"\b\d+\b")
QUOTED_RE = re.compile(r'"([^"\n]{3,120})"')
NAME_RE = re.compile(r"[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*")


def normalized(text):
    return re.sub(r"\s+", " ", text or "").strip()


def strip_preamble(q):
    q = q or ""
    q = re.sub(r"^Question date:.*?\n", "", q)
    q = re.sub(r"^\s*Question:\s*", "", q)
    return q.strip()


def is_sentence_start(text, pos):
    if pos == 0:
        return True
    i = pos - 1
    while i >= 0 and text[i] in " \t":
        i -= 1
    if i < 0:
        return True
    return text[i] in ".!?\n:"


def extract_anchors(question):
    text = strip_preamble(question)
    dates, date_spans = [], []
    for rx in DATE_RXS:
        for m in rx.finditer(text):
            dates.append(m.group(0))
            date_spans.append(m.span())

    def in_date(s, e):
        return any(s >= a and e <= b for a, b in date_spans)

    nums = []
    for m in NUM_RE.finditer(text):
        if in_date(*m.span()):
            continue
        v = m.group(0)
        if len(v) == 4 and v[:2] in ("19", "20"):
            continue
        nums.append(v)
    quoted = [m.group(1) for m in QUOTED_RE.finditer(text)]
    names = []
    for m in NAME_RE.finditer(text):
        cand = m.group(0)
        first = cand.split()[0].lower()
        if first in STOP_SINGLE or first in MONTHS or first in WEEKDAYS:
            continue
        if is_sentence_start(text, m.start()) and len(cand.split()) == 1:
            continue
        names.append(cand)

    def dedupe(xs):
        seen, out = set(), []
        for x in xs:
            k = x.lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    return {"names": dedupe(names), "dates": dedupe(dates),
            "numbers": dedupe(nums), "quoted": dedupe(quoted)}


def anchor_list(anchors):
    return [a for v in anchors.values() for a in v]


def _anchor_timestamp(ts):
    return "unknown" if ts is None else datetime.fromtimestamp(ts / 1000, UTC).isoformat()


def _render_source(source):
    date = _anchor_timestamp(source.get("timestamp"))
    return f"[source={source['id']}; role={source['role']}; timestamp={date}]\n{source['content']}"


def append_evidence(hits, sources, query, *, m=8):
    """Append up to m verbatim sources missed by the ranked evidence, anchor-ranked.

    sources: iterable of {id, content, role, timestamp} in stable store order.
    Rank = (anchor hit count desc, store order asc) over sources whose normalized
    text was not already delivered inside `hits`. Evidence formatting matches the
    native single-source hydration exactly; score = anchor hit count.
    """
    if m <= 0 or not sources:
        return []
    low = [a.lower() for a in anchor_list(extract_anchors(query))]
    if not low:
        return []
    delivered = {sid for hit in hits for sid in hit.source_ids}
    by_id = {}
    for order, source in enumerate(sources):
        by_id.setdefault(source["id"], source)
    delivered_texts = set()
    for sid in delivered:
        source = by_id.get(sid)
        if source is not None:
            text = normalized(source.get("content"))
            if text:
                delivered_texts.add(text.lower())
    seen_text = set(delivered_texts)
    scored = []
    for order, source in enumerate(sources):
        if source["id"] in delivered:
            continue
        text = normalized(source.get("content"))
        key = text.lower()
        if not text or key in seen_text:
            continue
        count = sum(1 for a in low if a in key)
        if count:
            seen_text.add(key)
            scored.append((-count, order, source))
    scored.sort(key=lambda t: (t[0], t[1]))
    out = []
    for count, _, source in scored[:m]:
        out.append(Evidence(
            id="memmachine_anchor_" + hashlib.sha256(
                json.dumps([source["id"]], ensure_ascii=False, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
            content=_render_source(source), source_ids=(source["id"],),
            score=float(-count),  # -count keeps the list order recoverable from scores
            kind="anchor_append",
        ))
    return out
