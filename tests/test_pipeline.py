"""Run: KUKU_PACK=path/to/pack.zip pytest -q   (tests use the real screenplay pack)"""
import json
import os
from pathlib import Path

import pytest

from promo.ai import discover, judge
from promo.clips import BeatIndex, Candidate, SegmentRef, build_promo, clip_lines
from promo.rank import rank
from promo.parser import load_episodes, parse_report

PACK = os.environ.get("KUKU_PACK", "")
pytestmark = pytest.mark.skipif(not Path(PACK).exists(), reason="set KUKU_PACK to the screenplay zip/folder")
CFG = {"min_duration": 20, "max_duration": 60, "hook_window": 3.0, "cliffhanger_window": 8.0, "promos_per_episode": 3}


@pytest.fixture(scope="module")
def eps():
    return load_episodes(PACK)


@pytest.fixture(scope="module")
def idx(eps):
    return BeatIndex(eps)


def cand(a, b, hook, cliff, hq="", cq=""):
    return Candidate(title="t", segments=[SegmentRef(start_beat=a, end_beat=b)], hook_beat=hook, hook="h",
                     hook_quote=hq, cliffhanger_beat=cliff, cliffhanger="c", cliffhanger_quote=cq, why="w")


def test_parse_complete(eps):
    rep = parse_report(eps)
    assert rep["episodes"] == 50 and rep["episode_numbers"] == list(range(1, 51))
    assert rep["total_beats"] == 3727 and not rep["issues"]
    b = eps[0].beats[0]
    assert b.id == "E01-B0001" and b.type == "CAMERA" and b.start == 0.0 and b.end == 4.8


def test_dialogue_fields(eps):
    d = next(b for b in eps[0].beats if b.type == "DIALOGUE")
    assert d.speaker == "MATRIARCH" and d.emotion == "Cold, Commanding" and d.text == "You should turn around, Juliana."
    assert (d.start, d.end) == (11.3, 13.1)


def test_valid_clip_and_snapping(eps, idx):
    # Ep1: "Gerald?" reveal through "Because he is marrying me."
    hook = next(b for b in eps[0].beats if b.text == "Gerald?")
    gold = next(b for b in eps[0].beats if b.text == "Because he is marrying me.")
    end = next(b for b in eps[0].beats if b.start >= hook.start + 25 and b.type == "DIALOGUE")
    p, errs = build_promo(cand(hook.id, end.id, hook.id, end.id, hq="Gerald?"), idx, "T1", CFG)
    assert p, errs
    assert p.segments[0].start <= hook.start and p.hook_offset <= 3
    assert 20 <= p.duration <= 60
    # a clip must never end in the middle of a spoken line
    for b in idx.clip_beats(p.segments[0]):
        if b.type == "DIALOGUE":
            assert b.end <= p.segments[0].end + 1e-6
    assert gold.start >= p.segments[0].start


def test_rejects_bad_inputs(eps, idx):
    b = eps[0].beats
    assert "unknown beat" in build_promo(cand("E01-B9999", b[30].id, b[0].id, b[30].id), idx, "x", CFG)[1][0]
    _, errs = build_promo(cand(b[0].id, b[5].id, b[0].id, b[5].id), idx, "x", CFG)  # ~11 s
    assert any("duration" in e for e in errs)
    late_hook = next(x for x in b if x.start > 10 and x.type == "DIALOGUE")
    end = next(x for x in b if x.end > 35)
    _, errs = build_promo(cand(b[0].id, end.id, late_hook.id, end.id), idx, "x", CFG)
    assert any("hook beat" in e for e in errs)
    _, errs = build_promo(cand(b[0].id, end.id, b[0].id, end.id, hq="a line nobody ever said"), idx, "x", CFG)
    assert any("not found in screenplay" in e for e in errs)


class FakeClient:
    """Stands in for an LLM: returns canned JSON, records prompts."""
    name = "fake:model"
    calls = cache_hits = 0

    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def complete(self, system, prompt):
        self.prompts.append(prompt)
        return self.replies.pop(0)


def gerald_clip(eps, idx, pid="T1"):
    b = eps[0].beats
    hook = next(x for x in b if x.text == "Gerald?")
    end = next(x for x in b if x.start >= hook.start + 25 and x.type == "DIALOGUE")
    p, errs = build_promo(cand(hook.id, end.id, hook.id, end.id), idx, pid, CFG)
    assert p, errs
    _, local = clip_lines(p, idx)
    lid = next(k for k, v in local.items() if v.text == "Gerald?")
    return p, local, lid


def crit(lid, score=8, quote="Gerald?"):
    return {"score": score, "beats": [lid], "quote": quote, "reason": "test"}


def judgement(pid, lid, q="Who is the woman holding his hand?", **over):
    j = {"promo_id": pid, "title": "Bride Sees Groom With Another", "unanswered_question": q,
         **{c: crit(lid) for c in ("hook_strength", "cliffhanger_strength", "standalone_clarity", "emotional_impact")}}
    j.update(over)
    return j


def test_discover_retry_and_blind_judge(eps, idx):
    b = eps[0].beats
    hook = next(x for x in b if x.text == "Gerald?")
    end = next(x for x in b if x.start >= hook.start + 25 and x.type == "DIALOGUE")
    good = cand(hook.id, end.id, hook.id, end.id).model_dump()
    bad = cand(b[0].id, b[5].id, b[0].id, b[5].id).model_dump()
    fc = FakeClient([json.dumps({"promos": [good, bad]}), json.dumps({"promos": []})])
    raw = []
    acc, rej = discover(eps[0], eps, idx, fc, CFG, raw, log=lambda *_: None)
    assert len(acc) == 1 and len(rej) == 1 and len(fc.prompts) == 2
    assert "REJECTED BY THE VALIDATOR" in fc.prompts[1]
    _, _, lid = gerald_clip(eps, idx)
    jc = FakeClient([json.dumps({"judgements": [judgement(acc[0].promo_id, lid)]})])
    judge(acc, idx, jc, raw, "E01")
    assert acc[0].scores["hook_strength"] == 8 and acc[0].evidence_ok
    assert acc[0].evidence_status["hook_strength"] == "verified"
    # blind: no show name, loglines, or real beat IDs (they reveal the episode number)
    assert "Broken Soldier" not in jc.prompts[0] and "LOGLINE" not in jc.prompts[0] and "E01-B" not in jc.prompts[0]


def test_cliffhanger_cap_and_blind_title(eps, idx):
    p, _, lid = gerald_clip(eps, idx)
    j = judgement("T1", lid, q="", cliffhanger_strength=crit(lid, 9))
    judge([p], idx, FakeClient([json.dumps({"judgements": [j]})]), [], "x")
    assert p.scores["cliffhanger_strength"] == 4 and any("capped" in w for w in p.warnings)
    assert p.title == "Bride Sees Groom With Another" and p.discovery_title == "t"
    p2, _, _ = gerald_clip(eps, idx)
    j2 = judgement("T1", lid, cliffhanger_strength=crit(lid, 9))
    judge([p2], idx, FakeClient([json.dumps({"judgements": [j2]})]), [], "x")
    assert p2.scores["cliffhanger_strength"] == 9


def test_generic_question_capped(eps, idx):
    p, _, lid = gerald_clip(eps, idx)
    j = judgement("T1", lid, q="What happens next?", cliffhanger_strength=crit(lid, 9))
    judge([p], idx, FakeClient([json.dumps({"judgements": [j]})]), [], "x")
    assert p.scores["cliffhanger_strength"] == 4 and any("generic" in w for w in p.warnings)


# ---------- regression: Gemini omitted cliffhanger_strength on Episode 2 (batch of 2) ----------
def test_judge_missing_field_retried_per_item(eps, idx):
    p1, _, lid = gerald_clip(eps, idx, "E02-P1")
    p2, _, _ = gerald_clip(eps, idx, "E02-P2")
    broken = judgement("E02-P1", lid)
    del broken["cliffhanger_strength"]                    # the exact failure seen on Episode 2
    fc = FakeClient([json.dumps({"judgements": [broken, judgement("E02-P2", lid)]}),
                     json.dumps({"judgements": [judgement("E02-P1", lid)]})])
    judge([p1, p2], idx, fc, [], "E02")
    assert len(fc.prompts) == 2
    assert "cliffhanger_strength" in fc.prompts[1] and "Field required" in fc.prompts[1]   # useful feedback
    assert "=== CLIP E02-P2" not in fc.prompts[1]          # only the failed clip is re-asked
    assert p2.scores and not p2.judge_failed               # the valid item is not thrown away
    assert p1.scores and not p1.judge_failed and any("attempt 2" in w for w in p1.warnings)


def test_judge_failure_is_never_a_zero_score(eps, idx):
    p1, _, lid = gerald_clip(eps, idx, "E02-P1")
    good, _, _ = gerald_clip(eps, idx, "E09-P1")
    broken = judgement("E02-P1", lid)
    del broken["cliffhanger_strength"]
    fc = FakeClient([json.dumps({"judgements": [broken]})] * 3)
    judge([p1], idx, fc, [], "E02", retries=2)
    assert len(fc.prompts) == 3                            # bounded: 1 try + 2 retries
    assert p1.judge_failed and not p1.scores and "cliffhanger_strength" in p1.judge_error and p1.judge_raw
    judge([good], idx, FakeClient([json.dumps({"judgements": [judgement("E09-P1", lid)]})]), [], "x")
    ranked, _ = rank([p1, good], CFG)
    assert [p.promo_id for p in ranked] == ["E09-P1"]       # failed clip excluded, not ranked at 0.0


def test_evidence_classification(eps, idx):
    p, local, lid = gerald_clip(eps, idx)
    d = [k for k in local if local[k].type == "DIALOGUE"]
    a, b2 = d[0], d[1]
    cases = {
        "hook_strength": crit(lid, quote="GERALD"),                                          # case/punct only
        "cliffhanger_strength": {"score": 7, "beats": [a, b2], "quote": local[a].text + " " + local[b2].text,
                                 "reason": ""},                                               # spans 2 cited beats
        "standalone_clarity": {"score": 7, "beats": [lid], "quote": local[b2].text, "reason": ""},  # real, wrong beat
        "emotional_impact": crit(lid, quote="I will burn this cathedral to the ground"),     # invented
    }
    judge([p], idx, FakeClient([json.dumps({"judgements": [judgement("T1", lid, **cases)]})]), [], "x")
    st = p.evidence_status
    assert st["hook_strength"] == "verified" and st["cliffhanger_strength"] == "verified"
    assert st["standalone_clarity"] == "other_beat"
    assert st["emotional_impact"].startswith("unsupported") and not p.evidence_ok


def test_out_of_clip_beat_id_unsupported(eps, idx):
    p, _, lid = gerald_clip(eps, idx)
    j = judgement("T1", lid, emotional_impact={"score": 8, "beats": ["L99"], "quote": "Gerald?", "reason": ""})
    judge([p], idx, FakeClient([json.dumps({"judgements": [j]})]), [], "x")
    assert p.evidence_status["emotional_impact"].startswith("unsupported: beat id") and not p.evidence_ok


def test_reaction_questions_capped_specific_kept():
    from promo.ai import GENERIC_Q, REACTION_Q
    capped = ["How will Juliana react to Gerald marrying another woman at their wedding?",
              "What will Juliana do now that she has hardened her resolve against Gerald and Clarissa?"]
    kept = ["Why is the stranger protecting Juliana and what will happen next?",
            "Will Juliana accept being humiliated or fight back?",
            "How will Gerald and Clarissa retaliate against Juliana's public humiliation?"]
    is_generic = lambda q: bool(GENERIC_Q.fullmatch(q.lower().strip(" ?."))) or bool(REACTION_Q.match(q.lower().strip(" ?.")))
    assert all(is_generic(q) for q in capped)
    assert not any(is_generic(q) for q in kept)
