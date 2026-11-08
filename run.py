"""CLI: python run.py --input pack.zip [--provider groq --model llama-3.3-70b-versatile] [--out outputs/run1]"""
from dotenv import load_dotenv
load_dotenv()
import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows terminals default to cp1252
import argparse
import time

from promo.llm import LLMClient
from promo.parser import load_episodes
from promo.pipeline import load_config, run

ap = argparse.ArgumentParser(description="Find and rank promo clips in microdrama screenplays")
ap.add_argument("--input", required=True, help="zip file or folder of episode .txt files")
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--provider", help="gemini | groq | anthropic | openai | heuristic")
ap.add_argument("--model")
ap.add_argument("--episodes", help="subset, e.g. 1-5 or 1,4,9")
ap.add_argument("--out", help="output folder (default outputs/<provider>_<timestamp>)")
ap.add_argument("--no-cache", action="store_true")
a = ap.parse_args()

cfg = load_config(a.config)
if a.provider:
    cfg["provider"] = a.provider
if a.model:
    cfg["model"] = a.model
if a.episodes:
    sel = []
    for part in a.episodes.split(","):
        lo, _, hi = part.partition("-")
        sel += list(range(int(lo), int(hi or lo) + 1))
    cfg["episodes"] = sel

episodes = load_episodes(a.input)
client = None
if cfg.get("provider", "heuristic") != "heuristic":
    client = LLMClient(cfg["provider"], cfg["model"], cache_dir=".cache" if not a.no_cache else f".cache/nocache_{time.time()}",
                       min_interval=cfg.get("min_interval", 0))
out = a.out or f"outputs/{cfg.get('provider', 'heuristic')}_{time.strftime('%Y%m%d_%H%M%S')}"
res = run(episodes, cfg, out, client)
print(f"\nResults in {out}/  (top promos: {out}/top_promos.md)")
