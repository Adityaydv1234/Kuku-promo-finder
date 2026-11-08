"""Schemas for LLM output, and deterministic clip building + validation.

The LLM only names beat IDs. Everything time-related happens here, in code:
  1. resolve beat IDs -> times
  2. never cut mid-line: widen to cover any dialogue line the boundary falls inside
  3. snap to the camera shot on screen (if within SNAP_TOL s) so the cut lands on a shot change
  4. check duration, hook position, cliffhanger position, ordering, quotes
Invalid recommendations are rejected with reasons (and retried once upstream),
never silently repaired.
"""
from __future__ import annotations

import difflib
import re

from pydantic import BaseModel, Field

from .parser import Beat, Episode, fmt_ts

SNAP_TOL = 2.0


# ---------------- LLM response schemas ----------------
class SegmentRef(BaseModel):
    start_beat: str
    end_beat: str


class Candidate(BaseModel):
    title: str = ""
    segments: list[SegmentRef] = Field(min_length=1, max_length=3)
    hook_beat: str
    hook: str
    hook_quote: str = ""
    cliffhanger_beat: str
    cliffhanger: str
    cliffhanger_quote: str = ""
    why: str


class DiscoveryResponse(BaseModel):
    promos: list[Candidate]


class Criterion(BaseModel):
    score: int = Field(ge=1, le=10)
    beats: list[str] = Field(min_length=1)   # clip-local beat IDs (L01, L02...) that support the score
    quote: str = Field(min_length=2)         # short exact phrase from those beats
    reason: str = ""


class Judgement(BaseModel):
    promo_id: str
    title: str = ""                 # written by the blind judge: can only describe what is in the clip
    unanswered_question: str = ""   # the specific open question the clip ends on ("" = none)
    hook_strength: Criterion
    cliffhanger_strength: Criterion
    standalone_clarity: Criterion
    emotional_impact: Criterion
    confusing_for_new_viewer: list[str] = []
    verdict: str = ""


class JudgeResponse(BaseModel):
    judgements: list[Judgement]


# ---------------- validated output ----------------
class Segment(BaseModel):
    episode: int
    start: float
    end: float
    start_ts: str
    end_ts: str
    start_beat: str
    end_beat: str


class Promo(BaseModel):
    promo_id: str
    title: str                      # blind judge's title (falls back to discovery title)
    discovery_title: str = ""       # title proposed by the discovery model (may spoil later events)
    episodes: list[int]
    segments: list[Segment]
    duration: float
    hook: str
    hook_beat: str
    hook_quote: str
    hook_offset: float              # seconds from clip start to hook beat
    cliffhanger: str
    cliffhanger_beat: str
    cliffhanger_quote: str
    why: str
    adjustments: list[str] = []     # deterministic boundary changes (mid-line / shot snapping)
    warnings: list[str] = []
    source: str = ""                # model that proposed it
    scores: dict[str, int] = {}
    evidence: dict[str, str] = {}
    confusing_for_new_viewer: list[str] = []
    verdict: str = ""
    unanswered_question: str = ""
    score: float = 0.0
    evidence_ok: bool = True        # False only when evidence is unsupported (invented or outside the clip)
    evidence_beats: dict[str, list[str]] = {}   # criterion -> real beat IDs cited by the judge
    evidence_status: dict[str, str] = {}        # criterion -> verified | wording_drift | other_beat | unsupported: ...
    judge_failed: bool = False      # no valid judgement after bounded retries: excluded from ranking
    judge_error: str = ""
    judge_raw: str = ""             # last raw judge response, kept for manual review


class BeatIndex:
    def __init__(self, episodes: list[Episode]):
        self.by_id: dict[str, Beat] = {}
        self.order: dict[str, int] = {}
        self.eps = {e.number: e for e in episodes}
        for e in episodes:
            for i, b in enumerate(e.beats):
                self.by_id[b.id] = b
                self.order[b.id] = i

    def clip_beats(self, seg: Segment) -> list[Beat]:
        return [b for b in self.eps[seg.episode].beats if seg.start - 1e-6 <= b.start < seg.end - 1e-6]


# ---------------- text helpers ----------------
def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", s.lower().replace("\u2019", "'"))).strip()


def quote_in(quote: str, text: str) -> bool:
    q, t = norm(quote), norm(text)
    if not q:
        return True
    if q in t:
        return True
    # tolerate tiny paraphrase / punctuation drift
    sm = difflib.SequenceMatcher(None, q, t)
    m = sm.find_longest_match(0, len(q), 0, len(t))
    return m.size >= 0.85 * len(q)


def beat_line(b: Beat, offset: float = 0.0, with_id: bool = True) -> str:
    who = f" {b.speaker}" + (f" ({b.emotion})" if b.emotion else "") if b.type == "DIALOGUE" else ""
    tag = f"{b.id} " if with_id else ""
    return f"[{tag}{fmt_ts(b.start - offset)}-{fmt_ts(b.end - offset)}] {b.type}{who}: {b.text}"


# ---------------- boundary logic ----------------
def _bounds(ep: Episode, start: float, end: float) -> tuple[float, float, list[str]]:
    notes = []
    dialog = [b for b in ep.beats if b.type == "DIALOGUE"]
    cams = [b for b in ep.beats if b.type == "CAMERA"]
    # never start or end inside a spoken line
    for b in dialog:
        if b.start < start < b.end:
            notes.append(f"start moved {fmt_ts(start)}->{fmt_ts(b.start)} to include full line {b.id}")
            start = b.start
    changed = True
    while changed:
        changed = False
        for b in dialog:
            if b.start < end < b.end:
                notes.append(f"end moved {fmt_ts(end)}->{fmt_ts(b.end)} to finish line {b.id}")
                end, changed = b.end, True
    # snap to the camera shot on screen, so cuts land on shot changes
    for c in cams:
        if c.start < start < c.end and start - c.start <= SNAP_TOL:
            notes.append(f"start snapped {fmt_ts(start)}->{fmt_ts(c.start)} to shot {c.id}")
            start = c.start
            break
    for c in cams:
        if c.start < end < c.end and c.end - end <= SNAP_TOL:
            notes.append(f"end snapped {fmt_ts(end)}->{fmt_ts(c.end)} to shot {c.id}")
            end = c.end
            break
    return round(start, 2), round(min(end, ep.duration), 2), notes


def build_promo(cand: Candidate, idx: BeatIndex, promo_id: str, cfg: dict, source: str = "") -> tuple[Promo | None, list[str]]:
    errs: list[str] = []
    ids = [cand.hook_beat, cand.cliffhanger_beat] + [x for s in cand.segments for x in (s.start_beat, s.end_beat)]
    missing = [i for i in ids if i not in idx.by_id]
    if missing:
        return None, [f"unknown beat id(s): {', '.join(sorted(set(missing)))}"]

    segs: list[Segment] = []
    notes: list[str] = []
    for k, s in enumerate(cand.segments, 1):
        a, b = idx.by_id[s.start_beat], idx.by_id[s.end_beat]
        if a.episode != b.episode:
            errs.append(f"segment {k}: start and end beats are in different episodes")
            continue
        if idx.order[a.id] > idx.order[b.id] or a.start > b.end:
            errs.append(f"segment {k}: start beat {a.id} comes after end beat {b.id}")
            continue
        st, en, n = _bounds(idx.eps[a.episode], a.start, max(a.end, b.end))
        notes += [f"seg{k}: {x}" for x in n]
        segs.append(Segment(episode=a.episode, start=st, end=en, start_ts=fmt_ts(st), end_ts=fmt_ts(en),
                            start_beat=a.id, end_beat=b.id))
    if errs:
        return None, errs
    keyed = [(s.episode, s.start, s.end) for s in segs]
    if len(set(keyed)) != len(keyed):
        errs.append("duplicated segment")
    for p, q in zip(segs, segs[1:]):
        if (q.episode, q.start) < (p.episode, p.end):
            errs.append(f"segments out of order or overlapping: {p.start_ts}-{p.end_ts} then {q.start_ts}-{q.end_ts}")

    duration = round(sum(s.end - s.start for s in segs), 2)
    lo, hi = cfg.get("min_duration", 20), cfg.get("max_duration", 60)
    if not lo <= duration <= hi:
        msg = f"duration {duration:.1f}s outside {lo}-{hi}s"
        if len(segs) == 1:  # give the model concrete, checkable fixes for its retry
            s = segs[0]
            ok = [b for b in idx.eps[s.episode].beats if b.type == "DIALOGUE" and lo + 1 <= b.end - s.start <= hi - 2]
            msg += (f". The clip starts at {s.start_ts}, so it must end between {fmt_ts(s.start + lo)} and "
                    f"{fmt_ts(s.start + hi)}")
            if ok:
                msg += "; end beats that fit: " + ", ".join(f"{b.id} (ends {fmt_ts(b.end)})" for b in ok[-4:])
            msg += ". Set cliffhanger_beat to match, or use 2 segments (hook + payoff) to skip the middle."
        errs.append(msg)

    hook, cliff = idx.by_id[cand.hook_beat], idx.by_id[cand.cliffhanger_beat]
    first, last = segs[0], segs[-1]
    hook_offset = round(hook.start - first.start, 2)
    if hook.episode != first.episode or not (first.start - 1e-6 <= hook.start < first.end):
        errs.append(f"hook beat {hook.id} is not inside the first segment")
    elif hook_offset > cfg.get("hook_window", 3.0):
        errs.append(f"hook beat {hook.id} starts {hook_offset:.1f}s into the clip (must be within {cfg.get('hook_window', 3.0)}s)")
    if cliff.episode != last.episode or not (last.start <= cliff.start < last.end + 1e-6):
        errs.append(f"cliffhanger beat {cliff.id} is not inside the last segment")
    elif last.end - cliff.end > cfg.get("cliffhanger_window", 8.0):
        errs.append(f"cliffhanger beat {cliff.id} ends {last.end - cliff.end:.1f}s before the clip ends (resolution risk)")

    warnings = []
    for label, q, beat in (("hook", cand.hook_quote, hook), ("cliffhanger", cand.cliffhanger_quote, cliff)):
        if q and not quote_in(q, beat.text):
            in_clip = any(quote_in(q, b.text) for s in segs for b in idx.clip_beats(s))
            if in_clip:
                warnings.append(f"{label} quote found in the clip but not in cited beat {beat.id}")
            else:
                errs.append(f"{label} quote not found in screenplay: '{q[:80]}'")
    if errs:
        return None, errs
    return Promo(promo_id=promo_id, title=cand.title, discovery_title=cand.title, episodes=sorted({s.episode for s in segs}), segments=segs,
                 duration=duration, hook=cand.hook, hook_beat=hook.id, hook_quote=cand.hook_quote,
                 hook_offset=hook_offset, cliffhanger=cand.cliffhanger, cliffhanger_beat=cliff.id,
                 cliffhanger_quote=cand.cliffhanger_quote, why=cand.why, adjustments=notes,
                 warnings=warnings, source=source), []


def clip_transcript(p: Promo, idx: BeatIndex, with_ids: bool = False) -> str:
    out, t0 = [], 0.0
    for k, s in enumerate(p.segments, 1):
        if len(p.segments) > 1:
            out.append(f"--- segment {k} (clip time {fmt_ts(t0)}) ---")
        for b in idx.clip_beats(s):
            out.append(beat_line(b, offset=s.start - t0, with_id=with_ids))
        t0 += s.end - s.start
    return "\n".join(out)


def clip_lines(p: Promo, idx: BeatIndex) -> tuple[str, dict[str, Beat]]:
    """Judge-facing transcript with clip-local IDs (L01, L02...). Real IDs like E03-B0047 would reveal
    the episode number to the blind judge, so they are mapped back in code after judging."""
    out, local, t0, k = [], {}, 0.0, 0
    for n, s in enumerate(p.segments, 1):
        if len(p.segments) > 1:
            out.append(f"--- segment {n} (clip time {fmt_ts(t0)}) ---")
        for b in idx.clip_beats(s):
            k += 1
            lid = f"L{k:02d}"
            local[lid] = b
            line = beat_line(b, offset=s.start - t0, with_id=False)
            out.append(f"[{lid} " + line[1:])
        t0 += s.end - s.start
    return "\n".join(out), local


def verify_evidence(quote: str, cited: list[str], local: dict[str, Beat]) -> str:
    """Classify one piece of judge evidence. Only 'unsupported...' is treated as a failure.
    verified       quote is in the cited beats (case/punctuation ignored)
    wording_drift  >=85% of the quote matches the cited beats contiguously (minor misquote)
    other_beat     quote is real clip text, but in beats other than the ones cited
    unsupported    beat IDs not in the clip, or quote not found anywhere in the clip"""
    bad_ids = [c for c in cited if c not in local]
    if bad_ids:
        return f"unsupported: beat id(s) {', '.join(bad_ids)} not in clip"
    order = list(local)
    cited_text = " ".join(local[c].text for c in sorted(set(cited), key=order.index))
    if norm(quote) and norm(quote) in norm(cited_text):
        return "verified"
    if quote_in(quote, cited_text):
        return "wording_drift"
    all_text = " ".join(b.text for b in local.values())
    dialogue = " ".join(b.text for b in local.values() if b.type == "DIALOGUE")   # lines split across beats
    if any(norm(quote) in norm(t) or quote_in(quote, t) for t in (all_text, dialogue)):
        return "other_beat"
    return "unsupported: quote not found in clip"
