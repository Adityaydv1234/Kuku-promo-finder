"""Compare two runs promo by promo: rank, title, cliffhanger score, question and explanation.
    python compare_runs.py outputs/<old_run> outputs/<new_run>
Writes comparison.md into the new run folder."""
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to cp1252
import json
import sys
from pathlib import Path

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
load = lambda d: {p["promo_id"]: p for p in json.loads((d / "promos_ranked.json").read_text("utf-8"))}
old, new = load(old_dir), load(new_dir)
jf = new_dir / "judge_failed.json"
failed = {p["promo_id"]: p for p in json.loads(jf.read_text("utf-8"))} if jf.exists() else {}
out = [f"# {old_dir.name} vs {new_dir.name}\n",
       "| promo | rank old > new | score old > new | cliffhanger old > new | title old > new |", "|---|---|---|---|---|"]
for pid in sorted(set(old) | set(new) | set(failed), key=lambda i: new.get(i, old.get(i, {"rank": 999}))["rank"]):
    o, n = old.get(pid, {}), new.get(pid, {})
    g = lambda p, k: p.get(k, "-") if p else "-"
    c = lambda p: p.get("scores", {}).get("cliffhanger_strength", "-") if p else "-"
    out.append(f"| {pid} | {g(o, 'rank')} > {g(n, 'rank')} | {g(o, 'score')} > {g(n, 'score')} | {c(o)} > {c(n)} | "
               f"{g(o, 'title')} > {g(n, 'title')} |")
out.append("\n## Cliffhanger explanations\n")
for pid, n in sorted(new.items(), key=lambda kv: kv[1]["rank"]):
    o = old.get(pid, {})
    out += [f"### {pid}  ({n['segments'][0]['start_ts']}-{n['segments'][-1]['end_ts']})",
            f"- old ({o.get('scores', {}).get('cliffhanger_strength', '-')}): {o.get('evidence', {}).get('cliffhanger_strength', '-')}",
            f"- new ({n['scores'].get('cliffhanger_strength', '-')}): {n['evidence'].get('cliffhanger_strength', '-')}",
            f"- unanswered question: {n.get('unanswered_question') or 'none named'}",
            *[f"- warning: {w}" for w in n.get("warnings", [])], ""]
if failed:
    out.append("## Judge failures (not ranked)\n")
    out += [f"- {pid}: {p['judge_error'][:200]}" for pid, p in failed.items()]
text = "\n".join(out)
(new_dir / "comparison.md").write_text(text, "utf-8")
print(text)
