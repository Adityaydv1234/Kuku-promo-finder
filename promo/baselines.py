"""Non-LLM baselines.

heuristic: keyword rules (slap, betray, shocked...). Runs offline with no API key,
           so the whole pipeline can be tested end to end, and it is a 'rules only'
           comparison point for the LLM.
random:    random valid 20-60s windows. Used in the blind human review to check that
           the AI picks are actually better than chance.
"""
from __future__ import annotations

import random
import re

from .clips import BeatIndex, Candidate, Judgement, Promo, SegmentRef, build_promo
from .parser import Episode

INTENSE = re.compile(r"slap|strike|struck|hit|shov|grab|betray|shock|gasp|furious|fury|rage|scream|cry|tears|sob|cruel|"
                     r"cold|threat|reveal|kneel|blood|kiss|dare|duke|marr|divorce|liar|lie|kill|die|dead|poison|"
                     r"secret|truth|never|how could|disgrace|humiliat|stunned|horrif|trembl|desperate|menac|smirk|gloat",
                     re.I)


def intensity(b) -> int:
    s = len(INTENSE.findall(b.text)) + 2 * len(INTENSE.findall(b.emotion))
    if b.type == "SFX" and re.search(r"slap|crash|scream|gasp|thud|shatter", b.text, re.I):
        s += 3
    if b.type == "DIALOGUE" and b.text.rstrip().endswith(("?", "!", "...", "\u2026")):
        s += 1
    return s


def heuristic_candidates(ep: Episode, n: int = 3) -> list[Candidate]:
    beats = ep.beats
    scored = sorted((b for b in beats if b.type != "CAMERA"), key=lambda b: (-intensity(b), b.start))
    out, used = [], []
    for h in scored:
        if len(out) >= n or intensity(h) == 0:
            break
        if any(abs(h.start - u) < 30 for u in used):
            continue
        cam = next((c for c in beats if c.type == "CAMERA" and c.start <= h.start < c.end), h)
        ends = [b for b in beats if b.type == "DIALOGUE" and 25 <= b.end - cam.start <= 55]
        if not ends:
            continue
        cl = max(ends, key=lambda b: (intensity(b), -b.end))
        out.append(Candidate(title=f"rules pick @{h.start:.0f}s", segments=[SegmentRef(start_beat=cam.id, end_beat=cl.id)],
                             hook_beat=h.id, hook=h.text[:120], cliffhanger_beat=cl.id, cliffhanger=cl.text[:120],
                             why="keyword-intensity rule (baseline, no AI)"))
        used.append(h.start)
    return out


def heuristic_discover(ep: Episode, idx: BeatIndex, cfg: dict) -> tuple[list[Promo], list[dict]]:
    acc, rej = [], []
    for c in heuristic_candidates(ep, cfg.get("promos_per_episode", 3)):
        p, errs = build_promo(c, idx, f"E{ep.number:02d}-P{len(acc) + 1}", cfg, "heuristic")
        if p:
            acc.append(p)
        else:
            rej.append({"episode": ep.number, "attempt": 1, "errors": errs, "candidate": c.model_dump()})
    return acc, rej


def heuristic_judge(promos: list[Promo], idx: BeatIndex) -> None:
    from .ai import apply_judgement
    from .clips import clip_lines
    for p in promos:
        _, local = clip_lines(p, idx)
        lids = list(local)
        first = [k for k in lids if local[k].start - p.segments[0].start <= 3] or lids[:1]
        last = [k for k in lids if p.segments[-1].end - local[k].end <= 6] or lids[-1:]
        names = {w for k in lids if local[k].type == "DIALOGUE" for w in re.findall(r"\b[A-Z][a-z]{2,}\b", local[k].text)}

        def crit(score, ks, why):
            k = next((k for k in ks if local[k].text), ks[0])
            return {"score": max(1, min(10, score)), "beats": [k], "quote": local[k].text[:40] or "-", "reason": why}
        q = next((local[k].text for k in reversed(last) if local[k].type == "DIALOGUE" and local[k].text.endswith("?")), "")
        apply_judgement(p, Judgement(
            promo_id=p.promo_id,
            hook_strength=crit(3 + sum(intensity(local[k]) for k in first), first, "keyword count, first 3s"),
            cliffhanger_strength=crit(3 + sum(intensity(local[k]) for k in last), last, "keyword count, last 6s"),
            standalone_clarity=crit(9 - len(names), lids, "fewer named references = clearer"),
            emotional_impact=crit(2 + sum(intensity(local[k]) for k in lids) // 3, lids, "keyword density"),
            unanswered_question=q, verdict="heuristic score, not an AI judgement"), local)


def random_clips(episodes: list[Episode], idx: BeatIndex, cfg: dict, n: int, seed: int = 7) -> list[Promo]:
    rng = random.Random(seed)
    relaxed = dict(cfg, hook_window=1e9, cliffhanger_window=1e9)
    out, tries = [], 0
    while len(out) < n and tries < n * 50:
        tries += 1
        ep = rng.choice(episodes)
        cams = [b for b in ep.beats if b.type == "CAMERA" and b.start < ep.duration - 25]
        if not cams:
            continue
        a = rng.choice(cams)
        target = rng.uniform(25, 50)
        b = min((x for x in ep.beats if x.end > a.start + 20), key=lambda x: abs(x.end - a.start - target), default=None)
        if b is None:
            continue
        c = Candidate(title="random", segments=[SegmentRef(start_beat=a.id, end_beat=b.id)], hook_beat=a.id,
                      hook="(random baseline)", cliffhanger_beat=b.id, cliffhanger="(random baseline)", why="random baseline")
        p, errs = build_promo(c, idx, f"R{len(out) + 1:02d}", relaxed, "random")
        if p and not any(o.segments[0].episode == p.segments[0].episode and abs(o.segments[0].start - p.segments[0].start) < 20 for o in out):
            out.append(p)
    return out
