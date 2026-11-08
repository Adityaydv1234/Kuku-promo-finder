"""Deterministic screenplay parser. No LLM involved.

Turns each episode .txt into structured beats with stable IDs:
    E01-B0007  ->  episode 1, beat 7 (1-based, in file order)

CAMERA, ACTION, SOUND FX and DIALOGUE beats overlap in time, so a clip's
duration is always computed from clip boundaries, never by summing beats.
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

from pydantic import BaseModel

TS_RE = re.compile(r"^(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?$")
TIMING_RE = re.compile(r"^Start:\s*\(([^)]*)\)\s*\|\s*End:\s*\(([^)]*)\)\s*$", re.I)
LABEL_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SPEAKER_RE = re.compile(r"^(.+?)\s*\((.+)\)\s*$")
SCENE_RE = re.compile(r"^SCENE\s+(\d+)\s*$", re.I)
META_RE = re.compile(r"^(EPISODE NUMBER|EPISODE LOGLINE|CONTINUITY NOTES|SCENE HEADING|SCENE START|SCENE END):\s*(.*)$", re.I)
NON_DIALOGUE = {"CAMERA": "CAMERA", "ACTION": "ACTION", "SOUND FX": "SFX", "SFX": "SFX"}


class Beat(BaseModel):
    id: str
    episode: int
    scene: int
    type: str            # CAMERA | ACTION | SFX | DIALOGUE
    speaker: str = ""
    emotion: str = ""
    start: float
    end: float
    text: str
    line: int            # 1-based line number in the source file


class Scene(BaseModel):
    number: int
    heading: str = ""
    start: float | None = None
    end: float | None = None


class Episode(BaseModel):
    number: int
    source_file: str
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = []
    beats: list[Beat] = []
    issues: list[str] = []

    @property
    def duration(self) -> float:
        return max((b.end for b in self.beats), default=0.0)


def parse_ts(s: str) -> float | None:
    m = TS_RE.match(s.strip())
    if not m:
        return None
    mm, ss, frac = m.groups()
    return int(mm) * 60 + int(ss) + (int(frac) / 10 ** len(frac) if frac else 0.0)


def fmt_ts(t: float) -> str:
    return f"{int(t // 60):02d}:{t % 60:05.2f}"


def parse_episode_text(text: str, source_file: str, fallback_number: int | None = None) -> Episode:
    lines = text.splitlines()
    ep = Episode(number=fallback_number or 0, source_file=source_file)
    scene: Scene | None = None
    pending: dict | None = None   # beat being assembled
    raw_beats: list[dict] = []

    def flush():
        nonlocal pending
        if pending is not None:
            raw_beats.append(pending)
            pending = None

    for i, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("--- SCRIPT"):
            flush()
            continue
        m = META_RE.match(line)
        if m:
            flush()
            key, val = m.group(1).upper(), m.group(2).strip()
            if key == "EPISODE NUMBER":
                n = re.search(r"\d+", val)
                if n:
                    ep.number = int(n.group())
            elif key == "EPISODE LOGLINE":
                ep.logline = val
            elif key == "CONTINUITY NOTES":
                ep.continuity = val
            elif scene is not None:
                if key == "SCENE HEADING":
                    scene.heading = val
                elif key == "SCENE START":
                    scene.start = parse_ts(val)
                elif key == "SCENE END":
                    scene.end = parse_ts(val)
            continue
        m = SCENE_RE.match(line)
        if m:
            flush()
            scene = Scene(number=int(m.group(1)))
            ep.scenes.append(scene)
            continue
        if line.lower().rstrip(":") == "actions and dialogue":
            continue
        m = LABEL_RE.match(line)
        if m:
            flush()
            label = m.group(1).strip()
            up = label.upper()
            if up in NON_DIALOGUE:
                btype, speaker, emotion = NON_DIALOGUE[up], "", ""
            else:
                sm = SPEAKER_RE.match(label)
                btype = "DIALOGUE"
                speaker, emotion = (sm.group(1), sm.group(2)) if sm else (label, "")
            pending = {"type": btype, "speaker": speaker.strip(), "emotion": emotion.strip(),
                       "start": None, "end": None, "text_lines": [], "line": i,
                       "scene": scene.number if scene else 0}
            continue
        m = TIMING_RE.match(line)
        if m and pending is not None and pending["start"] is None:
            pending["start"], pending["end"] = parse_ts(m.group(1)), parse_ts(m.group(2))
            if pending["start"] is None or pending["end"] is None:
                ep.issues.append(f"line {i}: malformed timestamp '{line}'")
            continue
        if pending is not None:
            pending["text_lines"].append(line)
        else:
            ep.issues.append(f"line {i}: text outside any beat: '{line[:60]}'")
    flush()

    if not ep.number:
        ep.issues.append("missing EPISODE NUMBER")
    if not ep.scenes:
        ep.issues.append("no SCENE blocks found")
    for k, rb in enumerate(raw_beats, start=1):
        bid = f"E{ep.number:02d}-B{k:04d}"
        text = " ".join(rb["text_lines"]).strip()
        if rb["type"] == "DIALOGUE":
            text = text.strip('"').strip("\u201c\u201d")
        if rb["start"] is None or rb["end"] is None:
            ep.issues.append(f"{bid} (line {rb['line']}): missing/malformed timing, beat dropped")
            continue
        if not text:
            ep.issues.append(f"{bid} (line {rb['line']}): empty text")
        if rb["end"] < rb["start"]:
            ep.issues.append(f"{bid} (line {rb['line']}): end before start")
        ep.beats.append(Beat(id=bid, episode=ep.number, scene=rb["scene"], type=rb["type"],
                             speaker=rb["speaker"], emotion=rb["emotion"], start=rb["start"],
                             end=rb["end"], text=text, line=rb["line"]))
    for s in ep.scenes:
        if s.start is None or s.end is None:
            ep.issues.append(f"scene {s.number}: missing start/end")
    return ep


def load_episodes(source: str | Path | bytes) -> list[Episode]:
    """Load episodes from a folder, a .zip path, or raw zip bytes."""
    files: list[tuple[str, str]] = []
    if isinstance(source, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(source))
        files = [(n, zf.read(n).decode("utf-8", "replace")) for n in zf.namelist()
                 if n.lower().endswith(".txt") and not n.startswith("__MACOSX")]
    else:
        p = Path(source)
        if p.suffix.lower() == ".zip":
            return load_episodes(p.read_bytes())
        files = [(str(f), f.read_text("utf-8", "replace")) for f in p.rglob("*.txt")]
    eps = []
    for name, text in sorted(files):
        n = re.search(r"episode-(\d+)", Path(name).name, re.I)
        eps.append(parse_episode_text(text, Path(name).name, int(n.group(1)) if n else None))
    eps.sort(key=lambda e: e.number)
    return eps


def parse_report(eps: list[Episode]) -> dict:
    from collections import Counter
    types = Counter(b.type for e in eps for b in e.beats)
    nums = [e.number for e in eps]
    dupes = sorted({n for n in nums if nums.count(n) > 1})
    return {
        "episodes": len(eps),
        "episode_numbers": nums,
        "duplicate_episode_numbers": dupes,
        "total_beats": sum(len(e.beats) for e in eps),
        "beat_types": dict(sorted(types.items())),
        "total_runtime_sec": round(sum(e.duration for e in eps), 2),
        "camera_coverage_gaps": camera_gaps(eps),
        "issues": {f"E{e.number:02d}": e.issues for e in eps if e.issues},
    }


def camera_gaps(eps: list[Episode], min_gap: float = 0.3) -> int:
    """Informational: stretches of time with no CAMERA beat (shot) described."""
    n = 0
    for e in eps:
        cams = sorted((b for b in e.beats if b.type == "CAMERA"), key=lambda b: b.start)
        n += sum(1 for a, b in zip(cams, cams[1:]) if b.start - a.end > min_gap)
    return n
