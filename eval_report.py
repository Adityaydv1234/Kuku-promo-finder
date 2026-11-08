"""After you fill in eval/review_sheet.csv by hand (blind), run:
    python eval_report.py outputs/<run>/eval
Prints AI-top-N vs random-clip pass rates for each rubric item and lists every failure.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to cp1252
import csv
import json
import sys
from pathlib import Path

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "eval")
key = json.loads((folder / "review_key.json").read_text())
rows = list(csv.DictReader((folder / "review_sheet.csv").open(encoding="utf-8")))
cols = [c for c in rows[0] if c.endswith("(0/1)")]
groups = {"ai": [], "random": []}
for r in rows:
    if all(r[c].strip() in ("0", "1") for c in cols):
        groups[key[r["clip"]]["source"]].append(r)
if not groups["ai"]:
    sys.exit("No completed rows yet: fill the 0/1 columns in review_sheet.csv first.")

lines = ["| rubric item | AI top picks | random clips |", "|---|---|---|"]
for c in cols:
    rate = lambda g: f"{sum(int(r[c]) for r in g)}/{len(g)}" if g else "n/a"
    lines.append(f"| {c.replace('(0/1)', '')} | {rate(groups['ai'])} | {rate(groups['random'])} |")
lines += ["", "Failures in AI picks:"]
for r in groups["ai"]:
    bad = [c.replace("(0/1)", "") for c in cols if r[c].strip() == "0"]
    if bad:
        lines.append(f"- {key[r['clip']]['promo_id']} {r['segments']}: failed {', '.join(bad)}. {r['notes']}")
report = "\n".join(lines)
print(report)
(folder / "eval_results.md").write_text(report, encoding="utf-8")
