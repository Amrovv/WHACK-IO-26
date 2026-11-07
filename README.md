# Climbing Coach

Real-time AI bouldering coach. A fixed camera watches a climber; the system speaks short cues
mid-climb and gives a persona-driven debrief between attempts. See `CLAUDE.md` for architecture
and rules, `CONTRIBUTING.md` for the git workflow.

## Setup

Needs [uv](https://docs.astral.sh/uv/). It installs Python 3.12 for you.

```bash
uv sync                  # install everything into .venv (first run downloads ~1 GB: torch, mediapipe)
cp .env.example .env     # then fill in GEMINI_API_KEY and ROBOFLOW_API_KEY
```

Get `models/pose_landmarker_full.task` and footage for `data/` from the shared drive.

## Everyday commands

```bash
uv run ruff format . && uv run ruff check .   # format + lint (before every PR)
uv run pytest                                 # tests
uv add <pkg>                                  # add a dependency; commit uv.lock with it
```

## Layout

```
run.py                 # D: single entry point
config/default.yaml    # all thresholds (structure owned by B)
config/personas/       # C
coach/contracts.py     # B: shared dataclasses, change by PR + team chat
coach/sources/         # A: cameras, files, 1-slot queue
coach/perception/      # A: pose, holds, holdmap_editor, contacts
coach/brain/           # B: features, attempts, detectors, arbiter, memory
coach/voice/           # C: speaker, tts, voice, persona, llm
coach/server/          # D: FastAPI + WebSocket
web/index.html         # D: dashboard
scripts/               # record_pose (A), replay_brain (B), build_phrasebank (C), bench_latency (D)
tests/
models/, data/         # gitignored; shared drive
demo/                  # D: script, slides, backup videos
```
