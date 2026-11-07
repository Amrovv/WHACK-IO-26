# Climbing Coach

Real-time AI bouldering coach for incident.io's "Live commentator" hackathon challenge (24 h, 4 people).
A fixed camera watches a climber. The system speaks short coaching cues mid-climb and gives a
persona-driven debrief between attempts, remembering earlier attempts.

## Architecture (two speeds)
- Fast path, every frame, fully local: frame source -> MediaPipe pose -> limb/hold contacts ->
  features -> cue detectors -> speech arbiter -> pre-rendered audio. Target: cue audio within ~0.8 s.
- Slow path, only between attempts: AttemptSummary JSON -> Gemini (streamed) -> Kokoro TTS -> debrief.
- The hold map is built once per problem from a frame of the empty wall: Roboflow hold detector,
  filtered by the colour of one clicked hold, plus manual fixes. Colour-only mode is the fallback.

## Hard rules
- No network calls, LLM calls or slow blocking work in perception, brain or the arbiter.
- Process the newest frame only; queues between threads hold one item and drop the oldest.
- Every cue has a deadline; if audio can't start in time, skip it.
- Brain logic uses state.t (capture time), never the wall clock. Inject clocks in tests.
- Coordinates are normalised 0-1, origin top-left, y down (higher on the wall = smaller y).
- Hold ids are numbered bottom (1) to top (N).
- All thresholds live in config/default.yaml. No magic numbers.

## Ownership: only edit your own folder
- A Perception: coach/sources, coach/perception, scripts/record_pose.py
- B Coach brain: coach/brain, coach/bus.py, coach/contracts.py (changes by PR + team agreement)
- C Voice and persona: coach/voice, config/personas, scripts/build_phrasebank.py
- D Product and demo: run.py, coach/stubs.py, coach/server, web/, scripts/bench_latency.py, demo/
If a task seems to need a change outside your folder or to contracts.py, stop and explain why.

## Stack
Python 3.12, uv, ruff, pytest, asyncio, OpenCV, MediaPipe Tasks (PoseLandmarker), Roboflow
inference-sdk, numpy, PyYAML, FastAPI + uvicorn, sounddevice + soundfile, Kokoro TTS, google-genai.
For fast-moving libraries, check the installed version's API before writing code.

## Commands
- uv sync
- uv run ruff format . && uv run ruff check .
- uv run pytest
- uv run python run.py --source file:<clip> --problem <id> [--voice print] [--no-llm]
- uv run python run.py --replay-pose data/pose_cache/<clip>.jsonl --problem <id> --voice print

## Style
Small, typed, testable functions. Dataclasses from coach/contracts.py for anything crossing a
module boundary. Every module has a runnable __main__ demo. Use logging, not print.

## Git
Branch `<owner>/<task>` off main, small PRs, never push to main. See CONTRIBUTING.md.
