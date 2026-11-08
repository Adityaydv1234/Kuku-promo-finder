"""Streamlit app for editors: upload screenplays -> ranked promos with exact cut points.

Run:  streamlit run app.py
"""
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from promo.llm import KEY_ENV, LLMClient
from promo.parser import load_episodes, parse_report
from promo.pipeline import load_config, run

st.set_page_config(page_title="Promo Finder", page_icon="🎬", layout="wide")
cfg = load_config()

DEFAULT_MODELS = {"gemini": "gemini-flash-latest", "groq": "llama-3.3-70b-versatile",
                  "anthropic": "claude-sonnet-4-6", "openai": "gpt-4o-mini", "heuristic": "(no model)"}
CRIT = ["hook_strength", "cliffhanger_strength", "standalone_clarity", "emotional_impact"]

# ---------------- sidebar: model + settings ----------------
with st.sidebar:
    st.header("Model")
    providers = list(DEFAULT_MODELS)
    provider = st.selectbox("Provider", providers, index=providers.index(cfg.get("provider", "gemini")),
                            help="'heuristic' runs offline keyword rules (baseline, no API key).")
    model = st.text_input("Model name", value=cfg.get("model") if provider == cfg.get("provider") else DEFAULT_MODELS[provider],
                          disabled=provider == "heuristic")
    api_key = ""
    if provider != "heuristic":
        env_name = KEY_ENV[provider]
        api_key = st.text_input(f"API key ({env_name})", type="password",
                                help="Leave empty to use the environment variable / Streamlit secret.")
        api_key = api_key or os.environ.get(env_name, "")
        try:
            api_key = api_key or st.secrets.get(env_name, "")
        except Exception:
            pass
        if provider == "openai":
            os.environ["OPENAI_BASE_URL"] = st.text_input("Base URL", os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    st.header("Settings")
    cfg["promos_per_episode"] = st.slider("Promos per episode", 1, 5, cfg.get("promos_per_episode", 3))
    cfg["top_n"] = st.slider("Top N to highlight", 5, 20, cfg.get("top_n", 10))
    cfg["min_interval"] = st.number_input("Seconds between API calls", 0.0, 30.0, float(cfg.get("min_interval", 4.5)),
                                          help="Raise this if you hit free-tier rate limits.")
    subset = st.text_input("Episodes (blank = all)", "", placeholder="e.g. 1-5 or 1,7,12")
    with st.expander("Ranking weights"):
        w = cfg.get("weights", {})
        cfg["weights"] = {c: st.slider(c.replace("_", " "), 0.0, 1.0, float(w.get(c, 0.25)), 0.05) for c in CRIT}

st.title("🎬 Promo Finder")
st.caption("Finds 20 to 60 second promo clips in microdrama screenplays: hook in the first 3 seconds, "
           "cut on a cliffhanger, understandable to a new viewer. An editorial shortlist, not a virality prediction.")

tab_run, tab_view = st.tabs(["Run on screenplays", "Open saved results"])


def parse_subset(s):
    sel = []
    for part in s.replace(" ", "").split(","):
        if part:
            lo, _, hi = part.partition("-")
            sel += list(range(int(lo), int(hi or lo) + 1))
    return sel or None


with tab_run:
    up = st.file_uploader("Upload a .zip of episode screenplays (.txt)", type=["zip"])
    if up:
        episodes = load_episodes(up.getvalue())
        rep = parse_report(episodes)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Episodes", rep["episodes"])
        c2.metric("Beats parsed", rep["total_beats"])
        c3.metric("Runtime", f"{rep['total_runtime_sec'] / 60:.0f} min")
        c4.metric("Parse issues", sum(len(v) for v in rep["issues"].values()))
        if rep["issues"]:
            with st.expander("Parse issues"):
                st.json(rep["issues"])
        if st.button("Find promos", type="primary"):
            cfg["episodes"] = parse_subset(subset)
            client = None
            if provider != "heuristic":
                if not api_key:
                    st.error(f"Add an API key for {provider} in the sidebar.")
                    st.stop()
                client = LLMClient(provider, model, api_key=api_key, cache_dir=".cache",
                                   min_interval=cfg["min_interval"], log=lambda *_: None)
            outdir = Path(tempfile.mkdtemp()) / f"{provider}_{time.strftime('%Y%m%d_%H%M%S')}"
            bar = st.progress(0.0, "Starting")
            logbox = st.empty()
            lines = []

            def log(m):
                lines.append(m)
                logbox.code("\n".join(lines[-8:]))
            try:
                res = run(episodes, cfg, outdir, client, progress=lambda f, t: bar.progress(min(f, 1.0), t), log=log)
            except Exception as e:
                st.error(f"Run failed: {e}")
                st.stop()
            ranked = json.loads((outdir / "promos_ranked.json").read_text("utf-8"))
            shutil.make_archive(str(outdir), "zip", outdir)
            st.session_state["result"] = {
                "ranked": ranked, "summary": res["summary"], "ablation": res["ablation"],
                "rejected": res["rejected"],
                "judge_failed": json.loads((outdir / "judge_failed.json").read_text("utf-8")), "csv": (outdir / "promos_ranked.csv").read_bytes(),
                "zip": Path(str(outdir) + ".zip").read_bytes(), "name": outdir.name}

with tab_view:
    saved = st.file_uploader("Open a promos_ranked.json from an earlier run", type=["json"], key="saved")
    if saved:
        st.session_state["result"] = {"ranked": json.loads(saved.getvalue()), "summary": {}, "ablation": {},
                                      "rejected": [], "csv": None, "zip": None, "name": saved.name}


def seg_text(p):
    return " | ".join(f"Ep{s['episode']} {s['start_ts']} to {s['end_ts']}" for s in p["segments"])


res = st.session_state.get("result")
if res:
    ranked, summ = res["ranked"], res["summary"]
    st.divider()
    if summ:
        c = st.columns(5)
        c[0].metric("Valid promos", summ["promos_validated"])
        c[1].metric("Rejected by validator", summ["promos_rejected"])
        c[2].metric("Judge failed", summ.get("judge_failed", 0), help="No valid score after retries: not ranked, listed below for manual review.")
        c[3].metric("Top-N avg score", summ["top_n_avg_judge_score"])
        c[4].metric("Random clips avg score", summ["random_baseline_avg_judge_score"],
                    help="Same judge on random 20 to 60 s clips. If this is close to the top-N score, the judge is not discriminating.")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download JSON", json.dumps(ranked, indent=2, ensure_ascii=False), "promos_ranked.json", "application/json")
    if res["csv"]:
        d2.download_button("Download CSV", res["csv"], "promos_ranked.csv", "text/csv")
    if res["zip"]:
        d3.download_button("Download all outputs (.zip)", res["zip"], f"{res['name']}.zip", "application/zip")

    top_n = cfg.get("top_n", 10)
    df = pd.DataFrame([{"rank": p["rank"], "promo": p["promo_id"], "title": p["title"], "cut points": seg_text(p),
                        "sec": p["duration"], "score": p["score"], **{c.split("_")[0]: p["scores"].get(c) for c in CRIT},
                        "flags": len(p["warnings"])} for p in ranked])
    st.subheader(f"Ranked promos ({len(ranked)})")
    st.dataframe(df, hide_index=True, width="stretch", height=280)

    st.subheader(f"Top {min(top_n, len(ranked))}: post these first")
    for p in ranked[:top_n] + ([] if not st.checkbox("Show all promos below") else ranked[top_n:]):
        with st.expander(f"#{p['rank']}  {p['title'] or p['promo_id']}  ·  {seg_text(p)}  ·  {p['duration']:.0f}s  ·  score {p['score']}"):
            a, b = st.columns([3, 2])
            with a:
                st.markdown(f"**Hook** (`{p['hook_beat']}`, +{p['hook_offset']:.1f}s): {p['hook']}")
                st.markdown(f"**Cliffhanger** (`{p['cliffhanger_beat']}`): {p['cliffhanger']}")
                st.markdown(f"**Unanswered question (blind judge):** {p.get('unanswered_question') or 'none named'}")
                if p.get("discovery_title") and p["discovery_title"] != p["title"]:
                    st.caption(f"Discovery model's title (may mention events outside the clip): {p['discovery_title']}")
                st.markdown(f"**Why it works:** {p['why']}")
                if p.get("verdict"):
                    st.markdown(f"**Blind judge:** {p['verdict']}")
                if p.get("confusing_for_new_viewer"):
                    st.markdown("**May confuse a new viewer:** " + "; ".join(p["confusing_for_new_viewer"]))
            with b:
                for c in CRIT:
                    if c in p["scores"]:
                        st.markdown(f"**{c.replace('_', ' ')}: {p['scores'][c]}/10**  \n{p['evidence'].get(c, '')}")
            if p["warnings"]:
                st.warning(" · ".join(p["warnings"]))
            if p["adjustments"]:
                st.caption("Boundary adjustments: " + "; ".join(p["adjustments"]))
            st.markdown("**Screenplay inside the clip** (times relative to clip start)")
            st.code(p.get("transcript", ""), language=None)

    if res["ablation"]:
        with st.expander("Ablation: ranking without the standalone-clarity weight"):
            st.json(res["ablation"])
    if res.get("judge_failed"):
        with st.expander(f"Judge failed: not ranked, review manually ({len(res['judge_failed'])})"):
            for p in res["judge_failed"]:
                st.markdown(f"**{p['promo_id']}** {seg_text(p)}: {p['judge_error']}")
                st.code(p.get("judge_raw", "")[:2000], language="json")
    if res["rejected"]:
        with st.expander(f"Rejected by the validator ({len(res['rejected'])})"):
            st.dataframe(pd.DataFrame([{"episode": r["episode"], "attempt": r["attempt"], "errors": "; ".join(r["errors"])}
                                       for r in res["rejected"]]), hide_index=True, width="stretch")
