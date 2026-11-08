"""Explainable ranking.

score = weighted average of the judge's 1-10 scores (weights in config.yaml)
        minus a fixed penalty if the judge's evidence quotes are not in the clip.
Then:
  dedupe:    drop a clip if it shares its hook or cliffhanger beat with a higher-ranked clip,
             or overlaps a higher-ranked clip in the same episode by more than `overlap_threshold`
             of the shorter clip.
  diversity: within the top N, at most `max_per_episode_top` clips per episode. Clips pushed out
             are not deleted, they just move below the top N. Weak clips are never promoted.
"""
from __future__ import annotations

from .clips import Promo

DEFAULT_WEIGHTS = {"hook_strength": 0.30, "cliffhanger_strength": 0.30, "standalone_clarity": 0.25, "emotional_impact": 0.15}


def score(p: Promo, weights: dict, evidence_penalty: float) -> float:
    if not p.scores:
        return 0.0
    tw = sum(weights.values()) or 1.0
    s = sum(weights[k] * p.scores.get(k, 0) for k in weights) / tw
    return round(s - (0 if p.evidence_ok else evidence_penalty), 3)


def overlap(a: Promo, b: Promo) -> float:
    inter = 0.0
    for x in a.segments:
        for y in b.segments:
            if x.episode == y.episode:
                inter += max(0.0, min(x.end, y.end) - max(x.start, y.start))
    return inter / max(1e-6, min(a.duration, b.duration))


def rank(promos: list[Promo], cfg: dict) -> tuple[list[Promo], list[dict]]:
    weights = cfg.get("weights", DEFAULT_WEIGHTS)
    pen = cfg.get("evidence_penalty", 0.5)
    promos = [p for p in promos if not p.judge_failed and p.scores]   # a missing judgement is never a score of 0
    for p in promos:
        p.score = score(p, weights, pen)
    order = sorted(promos, key=lambda p: (-p.score, -p.scores.get("hook_strength", 0), p.promo_id))
    kept, dropped = [], []
    for p in order:
        dup = next((k for k in kept if k.hook_beat == p.hook_beat or k.cliffhanger_beat == p.cliffhanger_beat
                    or overlap(p, k) > cfg.get("overlap_threshold", 0.5)), None)
        if dup:
            dropped.append({"promo_id": p.promo_id, "duplicate_of": dup.promo_id, "score": p.score,
                            "overlap": round(overlap(p, dup), 2)})
        else:
            kept.append(p)
    top_n, cap = cfg.get("top_n", 10), cfg.get("max_per_episode_top", 2)
    top, rest, per_ep = [], [], {}
    for p in kept:
        e = p.episodes[0]
        if len(top) < top_n and per_ep.get(e, 0) < cap:
            top.append(p)
            per_ep[e] = per_ep.get(e, 0) + 1
        else:
            rest.append(p)
    return top + rest, dropped


def ablation(promos: list[Promo], cfg: dict) -> dict:
    """Re-rank with standalone_clarity weight = 0. No extra API calls."""
    base, _ = rank([p.model_copy(deep=True) for p in promos], cfg)
    w = dict(cfg.get("weights", DEFAULT_WEIGHTS), standalone_clarity=0.0)
    alt, _ = rank([p.model_copy(deep=True) for p in promos], dict(cfg, weights=w))
    n = cfg.get("top_n", 10)
    b_ids, a_ids = [p.promo_id for p in base[:n]], [p.promo_id for p in alt[:n]]
    sc = {p.promo_id: p.scores.get("standalone_clarity", 0) for p in promos if p.scores}
    avg = lambda ids: round(sum(sc[i] for i in ids) / max(1, len(ids)), 2)
    return {
        "description": "Top-N with vs without the standalone_clarity weight",
        "top_with_standalone": b_ids,
        "top_without_standalone": a_ids,
        "entered_without_standalone": [i for i in a_ids if i not in b_ids],
        "left_without_standalone": [i for i in b_ids if i not in a_ids],
        "avg_standalone_score_with": avg(b_ids),
        "avg_standalone_score_without": avg(a_ids),
    }
