# Promo Finder (Kuku AI Generalist assignment, Problem 1)

Reads timestamped microdrama screenplays and returns a ranked shortlist of 20 to 60 second promo clips, each with exact cut points, the hook, the cliffhanger and why it works. It is an editorial decision-support tool: it shortlists, an editor decides.

## How it works

```
screenplay .txt ──► parser (code) ──► discover (LLM) ──► validate (code) ──► blind judge (LLM) ──► rank (code)
                      beats with IDs     picks beat IDs     times, rules        scores clip text     weights, dedupe
```

| Step | Who does it | Why |
|---|---|---|
| Parse every beat (CAMERA, ACTION, SOUND FX, DIALOGUE) into JSON with stable IDs like `E01-B0042` | Code | Deterministic, checkable. 3,727 / 3,727 beats parsed on the pack, 0 issues. |
| Find promo moments | LLM, per episode | Needs narrative judgment. The model returns **beat IDs only**, never timestamps. |
| Turn IDs into cut points | Code | Never cuts mid-line, snaps to the camera shot on screen (within 2 s), so a model cannot invent a timestamp. |
| Validate | Code | 20 to 60 s, hook beat starts within 3 s, cliffhanger within the last 8 s, segments ordered and non-overlapping, quoted lines must exist in the script. Failures get one retry with the validator's reasons, then are rejected and logged. Nothing is silently repaired. |
| Score | A separate, **blind** LLM call | The judge sees only the clip's text: no show name, no loglines. That is what makes "standalone clarity" real. Separate call, so the model does not grade its own picks. Every score must quote the clip; unverifiable evidence is penalised. |
| Rank | Code | `score = 0.30 hook + 0.30 cliffhanger + 0.25 standalone + 0.15 emotion` (configurable). Drops clips sharing a hook/cliffhanger beat or overlapping more than 50 %. Max 2 per episode in the top 10. |

Note: beats overlap in time (a CAMERA shot and a line of DIALOGUE run together), so duration is always `end - start` of the clip, never a sum of beats.

## Quick start

```bash
pip install -r requirements.txt

# offline, no key: keyword-rules baseline (checks the whole pipeline end to end)
python run.py --input pack.zip --provider heuristic

# with a free Gemini key from https://aistudio.google.com  ("Get API key")
export GEMINI_API_KEY=...            # Windows: set GEMINI_API_KEY=...
python run.py --input pack.zip --episodes 1-3     # quick test
python run.py --input pack.zip                    # full show (about 100 calls, roughly 10 min on free tier)

# the app
streamlit run app.py
```

### Swap the model
Change `provider` / `model` in `config.yaml`, pass `--provider groq --model llama-3.3-70b-versatile`, or pick in the app sidebar. Supported: `gemini`, `groq`, `anthropic`, `openai` (any OpenAI-compatible endpoint via `OPENAI_BASE_URL`), `heuristic`. Keys are read from `GEMINI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`. Nothing outside `promo/llm.py` knows which provider is used.

Responses are cached in `.cache/` (keyed by model + prompt), so re-runs are free and reproducible. 429 / 5xx errors are retried with backoff. If the default model name is not available on your key, use any current Flash model listed in AI Studio.

## Outputs (`outputs/<run>/`)

| File | What it is |
|---|---|
| `top_promos.md` | Human-readable top 10 |
| `promos_ranked.json` / `.csv` | Every valid promo, ranked, with segments, hook, cliffhanger, why, scores, evidence, warnings, clip transcript |
| `raw/llm_responses.jsonl` | Unedited model responses (discovery, retries, judge) |
| `candidates_validated.json` / `candidates_rejected.json` | What passed and what the validator rejected, with reasons |
| `duplicates_removed.json`, `ablation.json` | Dedupe log; ranking with vs without the standalone weight |
| `eval/review_sheet.csv` + `review_key.json` | Blind human review: AI top 10 mixed with 10 random clips |
| `summary.json`, `parse_report.json`, `run_log.txt` | Counts, API calls, cache hits, timings, parse check |

## Checking quality

1. Fill the 0/1 columns in `eval/review_sheet.csv` without opening `review_key.json`.
2. `python eval_report.py outputs/<run>/eval` prints AI vs random pass rates and lists every AI failure with episode and timestamps.
3. `summary.json` also reports the judge's average score for the top 10 vs random clips. If they are close, the judge is not discriminating.

## Tests

```bash
KUKU_PACK=path/to/pack.zip pytest -q
```
Covers parsing completeness, field extraction, boundary rules (no mid-line cuts), rejection of bad IDs / durations / late hooks / invented quotes, the retry loop and that the judge prompt is blind.

## Deploy (Streamlit Community Cloud, free)
Push this repo, create an app pointing at `app.py`, add `GEMINI_API_KEY` under App settings > Secrets. Users upload the zip in the app.

## Confidentiality
The screenplay pack and every output derived from it are git-ignored. Share outputs only through a restricted link.

## Known limits
- Works from the script, not the video: it cannot see acting, music or pacing.
- LLM scores are an opinion, not a measure of ad performance. Real validation is post-launch CTR / watch-through.
- Cross-episode stitching is supported by the data model but not proposed by the discovery prompt yet.
