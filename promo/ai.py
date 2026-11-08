"""The two LLM steps.

discover(): per episode, the model reads the full timestamped beat list (plus short
            series context) and proposes promos as BEAT IDs. Code validates them.
judge():    a separate, blind call. The judge sees ONLY the clip transcript, never the
            show name, loglines or other episodes, to simulate a scrolling viewer who
            has never seen the show. This is what makes 'standalone clarity' meaningful.
"""
from __future__ import annotations

import re

from pydantic import ValidationError

from .clips import (BeatIndex, Candidate, Judgement, Promo, beat_line, build_promo, clip_lines,
                    verify_evidence)
from .llm import LLMClient, extract_json
from .parser import Episode

DISCOVER_SYSTEM = """You are a senior promo editor for vertical microdrama apps (ReelShort / DramaBox style).
You cut 20-60 second promos from a show's own footage, to post on TikTok, Instagram and YouTube Shorts and pull in NEW viewers.

A great promo:
1. HOOK: opens on something that stops a scrolling thumb within the first 3 seconds: a slap, a betrayal, a reveal, a humiliation, a shocking line. No slow establishing shots first.
2. CLIFFHANGER: ends on an unresolved moment (a threat, a reveal about to land, a question, a shocked face). Cut BEFORE the answer or resolution.
3. STANDALONE: someone who has never seen the show understands who is wronging whom within seconds. Avoid clips that depend on unexplained backstory, names or earlier events.

Rules (your output is checked by code; violations are rejected):
- Refer to moments ONLY by the beat IDs given (e.g. E07-B0012). Never invent IDs, dialogue, characters or events.
- Each promo has 1 segment (preferred: one continuous stretch) or at most 3 segments from this episode, in time order, non-overlapping.
- start_beat of the first segment must be the hook beat itself or the CAMERA beat on screen when the hook happens. The hook beat must start within 3 seconds of the clip start.
- end_beat of the last segment should be the cliffhanger beat or a reaction shot at most a few seconds after it.
- Total length (from start_beat start to end_beat end, summed over segments) must be 20-60 seconds. Use the timestamps; beats of different types overlap in time, so do not add beat durations together.
- hook_quote / cliffhanger_quote: copy the exact words from that beat's text (dialogue or action), or "" if not useful.
- Prefer quality over quantity. If an episode has only one strong moment, return one promo.

Return ONLY JSON:
{"promos": [{"title": "3-6 word label",
  "segments": [{"start_beat": "E01-B0010", "end_beat": "E01-B0035"}],
  "hook_beat": "E01-B0011", "hook": "what happens in the first 3 seconds", "hook_quote": "...",
  "cliffhanger_beat": "E01-B0034", "cliffhanger": "the unresolved moment it cuts on", "cliffhanger_quote": "...",
  "why": "one line on why this works for a cold viewer"}]}"""

JUDGE_SYSTEM = """You are a skeptical short-form video ad reviewer. You are shown the screenplay transcript of short promo clips
(timestamps are relative to the clip start). You know NOTHING about the show beyond what is in each clip. Judge each clip as a
viewer scrolling TikTok who has never seen the series.

Score each criterion 1-10 (5 = mediocre, 8 = clearly strong, 10 = exceptional; use the full range, most clips are NOT 8+):
- hook_strength: would the first 3 seconds stop a scroll?
- cliffhanger_strength: does the clip END on a SPECIFIC unanswered narrative question: an imminent revelation, an undecided choice,
  an interrupted confrontation, a threat about to be carried out? First write that question in "unanswered_question"
  (e.g. "Will she accept the stranger's offer?"). Things that are NOT cliffhangers by themselves: a dramatic reaction, a revenge
  payoff, a completed insult, an emotional resolution, a determined look, a music swell, a cut to black. If you cannot name a
  specific question the clip leaves open, set "unanswered_question" to "" and cliffhanger_strength must be 4 or lower.
- standalone_clarity: can a brand-new viewer follow who the people are and what is at stake? Penalise unexplained names, references to earlier events, or confusing jumps.
- emotional_impact: intensity of the feeling (outrage, shock, longing, triumph).

Every line of a clip starts with a clip-local beat ID like [L07 ...]. For EVERY criterion give:
  "beats": the ID(s) of the line(s) that support the score, "quote": a short exact phrase copied from those lines
  (if a sentence continues across two lines, cite both IDs), "reason": why. Do not invent content.
All four criteria (hook_strength, cliffhanger_strength, standalone_clarity, emotional_impact) are REQUIRED for every clip,
and unanswered_question is a separate top-level field, not a replacement for cliffhanger_strength.
List anything a new viewer would not understand in confusing_for_new_viewer.
"title": 3-6 words describing ONLY what happens inside this clip (no events that are not shown).

Return ONLY JSON: {"judgements": [{"promo_id": "...", "title": "...", "unanswered_question": "Will ...?",
 "hook_strength": {"score": 7, "beats": ["L01"], "quote": "...", "reason": "..."},
 "cliffhanger_strength": {"score": 6, "beats": ["L18", "L19"], "quote": "...", "reason": "..."},
 "standalone_clarity": {"score": 5, "beats": ["L04"], "quote": "...", "reason": "..."},
 "emotional_impact": {"score": 8, "beats": ["L09"], "quote": "...", "reason": "..."},
 "confusing_for_new_viewer": ["..."], "verdict": "one sentence"}]}"""


def episode_prompt(ep: Episode, episodes: list[Episode], n_promos: int, context_eps: int = 5) -> str:
    prev = [e for e in episodes if ep.number - context_eps <= e.number < ep.number]
    ctx = "\n".join(f"Ep {e.number}: {e.logline}" for e in prev) or "(this is the first episode)"
    beats = "\n".join(beat_line(b) for b in ep.beats)
    return (f"SERIES SO FAR (for your understanding only; the promo must not rely on it):\n{ctx}\n\n"
            f"EPISODE {ep.number} LOGLINE: {ep.logline}\nCONTINUITY: {ep.continuity}\n"
            f"EPISODE LENGTH: {ep.duration:.1f}s\n\nBEATS [id start-end] TYPE: text\n{beats}\n\n"
            f"Find up to {n_promos} of the strongest promos in episode {ep.number}.")


def _parse_discovery(text: str) -> tuple[list[Candidate], list[str]]:
    try:
        data = extract_json(text)
    except Exception as e:
        return [], [f"response is not valid JSON: {e}"]
    if isinstance(data, list):
        data = {"promos": data}
    good, errs = [], []
    for i, raw in enumerate(data.get("promos", []) if isinstance(data, dict) else []):
        try:
            good.append(Candidate.model_validate(raw))
        except ValidationError as e:
            errs.append(f"promo {i + 1} schema error: {e.errors()[0]['loc']} {e.errors()[0]['msg']}")
    return good, errs


def discover(ep: Episode, episodes: list[Episode], idx: BeatIndex, client: LLMClient, cfg: dict,
             raw_log: list, log=print) -> tuple[list[Promo], list[dict]]:
    prompt = episode_prompt(ep, episodes, cfg.get("promos_per_episode", 3), cfg.get("context_episodes", 5))
    text = client.complete(DISCOVER_SYSTEM, prompt)
    raw_log.append({"step": "discover", "episode": ep.number, "attempt": 1, "model": client.name, "response": text})
    cands, schema_errs = _parse_discovery(text)
    accepted, rejected, failed = [], [], []
    for c in cands:
        p, errs = build_promo(c, idx, f"E{ep.number:02d}-P{len(accepted) + 1}", cfg, client.name)
        if p and p.episodes == [ep.number]:
            accepted.append(p)
        else:
            errs = errs or ["promo uses beats outside this episode"]
            failed.append((c, errs))
    for e in schema_errs:
        rejected.append({"episode": ep.number, "attempt": 1, "errors": [e], "candidate": None})

    if (failed or schema_errs or not cands) and cfg.get("retry_invalid", True):
        problems = "\n".join(f"- promo '{c.title}': " + "; ".join(errs) for c, errs in failed)
        problems += "".join(f"\n- {e}" for e in schema_errs) or ""
        if not cands and not schema_errs:
            problems = "- no promos returned"
        retry_prompt = (prompt + "\n\nYOUR PREVIOUS ANSWER:\n" + text + "\n\nTHESE PROMOS WERE REJECTED BY THE VALIDATOR:\n"
                        + problems + "\n\nReturn JSON with corrected versions of ONLY the rejected promos (or better "
                        "replacements). Do NOT resend a rejected promo unchanged: change its beat IDs using the "
                        "timestamps and the suggestions above.")
        log(f"  E{ep.number:02d}: {len(failed) + len(schema_errs)} invalid, retrying once")
        text2 = client.complete(DISCOVER_SYSTEM, retry_prompt)
        raw_log.append({"step": "discover", "episode": ep.number, "attempt": 2, "model": client.name, "response": text2})
        cands2, schema2 = _parse_discovery(text2)
        for c in cands2:
            p, errs = build_promo(c, idx, f"E{ep.number:02d}-P{len(accepted) + 1}", cfg, client.name)
            if p and p.episodes == [ep.number]:
                p.warnings.append("accepted on retry")
                accepted.append(p)
            else:
                rejected.append({"episode": ep.number, "attempt": 2, "errors": errs or ["outside episode"],
                                 "candidate": c.model_dump()})
        for e in schema2:
            rejected.append({"episode": ep.number, "attempt": 2, "errors": [e], "candidate": None})
    for c, errs in failed:
        rejected.append({"episode": ep.number, "attempt": 1, "errors": errs, "candidate": c.model_dump()})
    return accepted, rejected


def _parse_judgements(text: str, expected: list[str]) -> tuple[dict[str, Judgement], dict[str, str]]:
    """Validate each clip's judgement on its own: one malformed item must not discard the others."""
    ok, errs = {}, {}
    try:
        data = extract_json(text)
    except Exception as e:
        return {}, {pid: f"response is not valid JSON ({str(e)[:80]}); possibly truncated" for pid in expected}
    items = data.get("judgements", []) if isinstance(data, dict) else data if isinstance(data, list) else []
    for raw in items if isinstance(items, list) else []:
        if not isinstance(raw, dict):
            continue
        pid = str(raw.get("promo_id", ""))
        if pid not in expected or pid in ok:
            continue
        try:
            ok[pid] = Judgement.model_validate(raw)
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in e.errors()[:6])
            errs[pid] = f"{problems} (keys returned: {sorted(raw)})"
    for pid in expected:
        if pid not in ok and pid not in errs:
            errs[pid] = "no judgement returned for this clip"
    return ok, errs


def judge(promos: list[Promo], idx: BeatIndex, client: LLMClient, raw_log: list, label: str, retries: int = 2) -> None:
    """Blind-score promos in place, one call per batch. Malformed items are re-asked (with the validator's
    errors) up to `retries` times. Still invalid -> judge_failed=True, no scores, raw response kept."""
    if not promos:
        return
    views = {p.promo_id: clip_lines(p, idx) for p in promos}
    pending, feedback, text, errs = list(promos), "", "", {}
    for attempt in range(1, retries + 2):
        blocks = [f"=== CLIP {p.promo_id} ({p.duration:.0f}s) ===\n{views[p.promo_id][0]}" for p in pending]
        text = client.complete(JUDGE_SYSTEM, "Score each clip.\n\n" + "\n\n".join(blocks) + feedback)
        raw_log.append({"step": "judge", "batch": label, "attempt": attempt, "model": client.name, "response": text})
        ok, errs = _parse_judgements(text, [p.promo_id for p in pending])
        for p in pending:
            if p.promo_id in ok:
                apply_judgement(p, ok[p.promo_id], views[p.promo_id][1])
                if attempt > 1:
                    p.warnings.append(f"judge output valid on attempt {attempt}")
        pending = [p for p in pending if p.promo_id in errs]
        if not pending:
            return
        feedback = ("\n\nYOUR PREVIOUS ANSWER:\n" + text[:6000] + "\n\nIT FAILED VALIDATION:\n"
                    + "\n".join(f"- {pid}: {e}" for pid, e in errs.items())
                    + "\n\nReturn JSON for ONLY the clips above, with ALL required fields for each.")
    for p in pending:
        mark_judge_failed(p, errs.get(p.promo_id, "unknown"), text)


def mark_judge_failed(p: Promo, error: str, raw: str = "") -> None:
    p.judge_failed, p.judge_error, p.judge_raw = True, error[:1000], raw[:4000]
    p.scores, p.evidence, p.score = {}, {}, 0.0
    p.warnings.append(f"JUDGE FAILED (excluded from ranking, manual review): {error[:200]}")


# questions that fit any clip do not count as a specific cliffhanger
GENERIC_Q = re.compile(r"(what|how) (will|would|does|is going to) (happen|happens)( next)?|what happens next|what will (she|he|they) do( next)?|"
                       r"how will (this|it|things) (end|turn out|unfold|play out)|what will happen( next)?")


# "How will X react?" / "What will X do now?" restate a reaction, not a specific open plot question
REACTION_Q = re.compile(r"^(how|what) (will|would|is going to) [\w' ]{1,40}? (react|respond|do|feel|handle|cope)\b")


def apply_judgement(p: Promo, j: Judgement, local: dict) -> None:
    crit = ["hook_strength", "cliffhanger_strength", "standalone_clarity", "emotional_impact"]
    p.scores = {c: getattr(j, c).score for c in crit}
    p.confusing_for_new_viewer = j.confusing_for_new_viewer
    p.verdict = j.verdict
    p.unanswered_question = j.unanswered_question.strip()
    if j.title.strip():
        p.title = j.title.strip()
    # rubric rule enforced in code, not left to the model: no specific open question -> cliffhanger capped at 4
    q = p.unanswered_question.lower().strip(" ?.")
    missing = not q or q in {"none", "n/a", "na"}
    generic = missing or bool(GENERIC_Q.fullmatch(q)) or bool(REACTION_Q.match(q))
    if generic and p.scores["cliffhanger_strength"] > 4:
        why = "no" if missing else "only a generic/reaction"
        p.warnings.append(f"cliffhanger capped {p.scores['cliffhanger_strength']}->4: judge named {why} unanswered question")
        p.scores["cliffhanger_strength"] = 4
    # evidence: beat IDs must be in the clip and contain the quote
    unsupported, notes = [], []
    for c in crit:
        cr = getattr(j, c)
        status = verify_evidence(cr.quote, cr.beats, local)
        p.evidence_status[c] = status
        p.evidence_beats[c] = [local[b].id for b in cr.beats if b in local]
        p.evidence[c] = f'"{cr.quote}" [{", ".join(p.evidence_beats[c]) or ", ".join(cr.beats)}] {cr.reason}'.strip()
        if status.startswith("unsupported"):
            unsupported.append(f"{c} ({status[13:]}: '{cr.quote[:50]}')")
        elif status == "other_beat":
            notes.append(f"{c} quote is real but cites the wrong beat")
        elif status == "wording_drift":
            notes.append(f"{c} quote slightly misquoted")
    p.evidence_ok = not unsupported
    if unsupported:
        p.warnings.append("judge evidence UNSUPPORTED (penalised): " + "; ".join(unsupported))
    if notes:
        p.warnings.append("evidence note (no penalty): " + "; ".join(notes))
