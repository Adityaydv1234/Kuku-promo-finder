"""Runs the whole thing and writes every artifact to one output folder.

parse -> discover (LLM or heuristic) -> validate -> blind judge -> rank -> ablation -> review sheet
"""
from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path

import yaml

from .ai import discover, judge, mark_judge_failed
from .baselines import heuristic_discover, heuristic_judge, random_clips
from .clips import BeatIndex, Promo, clip_transcript
from .llm import LLMClient
from .parser import Episode, fmt_ts, parse_report
from .rank import ablation, rank


def load_config(path: str | Path = "config.yaml") -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text()) if p.exists() else {}


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=False), "utf-8")


def seg_str(p: Promo) -> str:
    return " | ".join(f"Ep{s.episode} {s.start_ts}-{s.end_ts}" for s in p.segments)


def run(episodes: list[Episode], cfg: dict, outdir: str | Path, client: LLMClient | None = None,
        progress=None, log=print) -> dict:
    """client=None runs the offline heuristic baseline (no API key needed)."""
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    logf = (out / "run_log.txt").open("w", encoding="utf-8")

    def L(msg):
        log(msg)
        logf.write(msg + "\n")
        logf.flush()

    only = cfg.get("episodes")  # optional subset, e.g. [1,2,3]
    eps = [e for e in episodes if not only or e.number in only]
    idx = BeatIndex(episodes)
    model = client.name if client else "heuristic (no LLM)"
    L(f"model: {model} | episodes: {len(eps)}")
    rep = parse_report(episodes)
    dump(out / "parse_report.json", rep)
    if rep["issues"]:
        L(f"parse issues in {len(rep['issues'])} episodes, see parse_report.json")

    raw: list = []
    accepted: list[Promo] = []
    rejected: list[dict] = []
    total = len(eps) * 2
    for i, ep in enumerate(eps):
        if progress:
            progress(i / total, f"Finding promos in episode {ep.number}")
        try:
            if client:
                acc, rej = discover(ep, episodes, idx, client, cfg, raw, log=L)
            else:
                acc, rej = heuristic_discover(ep, idx, cfg)
        except Exception as e:  # one bad episode must not kill the run
            L(f"  E{ep.number:02d}: discovery FAILED: {e}")
            rej = [{"episode": ep.number, "attempt": 0, "errors": [f"call failed: {e}"], "candidate": None}]
            acc = []
        L(f"E{ep.number:02d}: {len(acc)} valid, {len(rej)} rejected")
        accepted += acc
        rejected += rej

    by_ep: dict[int, list[Promo]] = {}
    for p in accepted:
        by_ep.setdefault(p.episodes[0], []).append(p)
    for k, (e, ps) in enumerate(sorted(by_ep.items())):
        if progress:
            progress((len(eps) + k) / total, f"Blind-judging episode {e}")
        try:
            judge(ps, idx, client, raw, f"E{e:02d}", cfg.get("judge_retries", 2)) if client else heuristic_judge(ps, idx)
        except Exception as ex:
            L(f"  E{e:02d}: judge call FAILED: {ex}")
            for p in ps:
                if not p.scores:
                    mark_judge_failed(p, f"judge call failed: {ex}")
        nf = sum(p.judge_failed for p in ps)
        if nf:
            L(f"  E{e:02d}: {nf} clip(s) judge_failed after retries (kept for manual review)")

    judge_failed = [p for p in accepted if p.judge_failed]
    ranked, dupes = rank(accepted, cfg)
    abl = ablation(accepted, cfg) if accepted else {}

    # ---- random baseline for the blind human review (+ judge sanity check) ----
    rand = random_clips(eps, idx, cfg, cfg.get("review_n", 10), seed=cfg.get("seed", 7))
    if rand:
        try:
            judge(rand, idx, client, raw, "random-baseline", cfg.get("judge_retries", 2)) if client else heuristic_judge(rand, idx)
        except Exception as ex:
            L(f"random-baseline judge failed: {ex}")
        rank(rand, cfg)  # fills .score

    # ---------------- write artifacts ----------------
    (out / "raw").mkdir(exist_ok=True)
    with (out / "raw" / "llm_responses.jsonl").open("w", encoding="utf-8") as f:
        for r in raw:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dump(out / "candidates_validated.json", [p.model_dump() for p in sorted(accepted, key=lambda p: p.promo_id)])
    dump(out / "candidates_rejected.json", rejected)
    dump(out / "duplicates_removed.json", dupes)
    dump(out / "judge_failed.json", [p.model_dump() for p in judge_failed])
    dump(out / "ablation.json", abl)
    ranked_rows = []
    for r, p in enumerate(ranked, 1):
        d = p.model_dump()
        d["rank"] = r
        d["transcript"] = clip_transcript(p, idx, with_ids=True)
        ranked_rows.append(d)
    dump(out / "promos_ranked.json", ranked_rows)
    write_csv(out / "promos_ranked.csv", ranked)
    write_review_sheet(out / "eval", ranked[: cfg.get("review_n", 10)], rand, idx, cfg.get("seed", 7))
    write_markdown(out / "top_promos.md", ranked[: cfg.get("top_n", 10)], idx, model, judge_failed)

    rs = [p.score for p in rand if p.scores]
    ts = [p.score for p in ranked[: cfg.get("top_n", 10)] if p.scores]
    summary = {
        "model": model,
        "episodes_processed": len(eps),
        "promos_validated": len(accepted),
        "promos_rejected": len(rejected),
        "duplicates_removed": len(dupes),
        "judge_failed": len(judge_failed),
        "promos_ranked": len(ranked),
        "top_n_avg_judge_score": round(sum(ts) / len(ts), 2) if ts else None,
        "random_baseline_avg_judge_score": round(sum(rs) / len(rs), 2) if rs else None,
        "api_calls": client.calls if client else 0,
        "cache_hits": client.cache_hits if client else 0,
        "seconds": round(time.time() - t0, 1),
        "config": {k: v for k, v in cfg.items() if k != "api_key"},
    }
    dump(out / "summary.json", summary)
    L(f"done: {len(accepted)} valid, {len(rejected)} rejected, {len(judge_failed)} judge_failed, "
      f"{len(dupes)} duplicates, {summary['seconds']}s")
    logf.close()
    if progress:
        progress(1.0, "Done")
    return {"summary": summary, "ranked": ranked, "rejected": rejected, "ablation": abl, "idx": idx}


CRIT = ["hook_strength", "cliffhanger_strength", "standalone_clarity", "emotional_impact"]


def write_csv(path: Path, ranked: list[Promo]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "promo_id", "title", "segments", "duration_s", "hook", "hook_beat", "cliffhanger",
                    "cliffhanger_beat", "unanswered_question", "why", *CRIT, "score", "evidence_ok", "warnings"])
        for r, p in enumerate(ranked, 1):
            w.writerow([r, p.promo_id, p.title, seg_str(p), p.duration, p.hook, p.hook_beat, p.cliffhanger,
                        p.cliffhanger_beat, p.unanswered_question, p.why, *[p.scores.get(c, "") for c in CRIT], p.score,
                        p.evidence_ok, "; ".join(p.warnings)])


def write_review_sheet(folder: Path, ai: list[Promo], rand: list[Promo], idx: BeatIndex, seed: int) -> None:
    """Blind sheet: AI top-N and random clips shuffled, labelled only C01..C20."""
    folder.mkdir(parents=True, exist_ok=True)
    items = [("ai", p) for p in ai] + [("random", p) for p in rand]
    random.Random(seed).shuffle(items)
    key = {}
    with (folder / "review_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip", "segments", "duration_s", "transcript",
                    "hook_stops_scroll(0/1)", "ends_unresolved(0/1)", "standalone_clear(0/1)",
                    "timestamps_correct(0/1)", "would_post(0/1)", "notes"])
        for i, (src, p) in enumerate(items, 1):
            cid = f"C{i:02d}"
            key[cid] = {"source": src, "promo_id": p.promo_id}
            w.writerow([cid, seg_str(p), p.duration, clip_transcript(p, idx), "", "", "", "", "", ""])
    dump(folder / "review_key.json", key)


def write_markdown(path: Path, top: list[Promo], idx: BeatIndex, model: str, failed: list[Promo] = ()) -> None:
    lines = [f"# Top promos\n\nModel: {model}\n"]
    for r, p in enumerate(top, 1):
        lines += [f"## {r}. {p.title or p.promo_id}  ({p.promo_id}, score {p.score})",
                  f"- **Segments:** {seg_str(p)}  ({p.duration:.1f}s)",
                  f"- **Hook** ({p.hook_beat}, +{p.hook_offset:.1f}s): {p.hook}",
                  f"- **Cliffhanger** ({p.cliffhanger_beat}): {p.cliffhanger}",
                  f"- **Unanswered question (blind judge):** {p.unanswered_question or 'none named'}",
                  f"- **Cliffhanger judged:** {p.evidence.get('cliffhanger_strength', '')}",
                  *([f"- **Discovery title (may spoil):** {p.discovery_title}"] if p.discovery_title != p.title else []),
                  f"- **Why:** {p.why}",
                  "- **Scores:** " + ", ".join(f"{k} {v}" for k, v in p.scores.items()),
                  *( [f"- **Warnings:** {'; '.join(p.warnings)}"] if p.warnings else []), ""]
    if failed:
        lines.append("## Judge failures (not ranked, review manually)")
        lines += [f"- {p.promo_id} {seg_str(p)}: {p.judge_error[:200]}" for p in failed]
    path.write_text("\n".join(lines), "utf-8")


__all__ = ["run", "load_config", "fmt_ts"]
