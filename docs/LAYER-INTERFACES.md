# Layer interfaces

**Status: proposal from D. A, B and C please confirm in the team chat.** No change to
`coach/contracts.py`. Everything here matches code that already exists on `main`,
`a/feat-perception-pipeline` and `daniel`, so no one has to rewrite anything to agree to it.

`docs/INTEGRATION-contract.md` (B) still describes how the brain's parts fit together inside
`Brain`. This page is the layer above it: how A, B, C and D plug into each other.

## The pipeline, and who owns each hop

```
camera / file
   │  A: parse_source(spec, fast=) → (t, frame_idx, bgr) frames
   ▼
reader thread ──► A's LatestQueue (frames, thread → thread, 1 slot, drops oldest)
   ▼
pose thread: A's PoseTracker.process(t, frame_idx, bgr) → ClimberState
   │  publishes "state", "frame" (JPEG, ~10 fps); "holdmap" once at start
   ▼
D's AsyncLatestQueue (states, thread → asyncio, 1 slot, drops oldest)
   ▼
asyncio loop: B's Brain.run(states) ──► C's Voice.say(SpeakRequest)
   │  publishes "event", "suppressed", "attempt"      │  publishes "spoken", "debrief_text"
   ▼                                                  ▼
Bus ──► D: dashboard server (/ws, /video), bus log
```

The reader and pose threads are started by D's `PoseThreads` in `run.py`, built only from A's
public functions. A's code never starts threads or touches the bus itself.

## Two queues, two names

The integration crash came from two classes both called `LatestQueue`. They do different jobs:

| Name | Owner | Hop | API |
|---|---|---|---|
| `coach.sources.LatestQueue` | A | thread → thread, inside perception (frames) | `put(x)`, `get(timeout) -> x \| None` (blocking), `close()` |
| `coach.stubs.AsyncLatestQueue` | D | perception thread → brain on the asyncio loop (states) | `put(x)` from any thread, `await get()` |

**Rule:** `run.py` always builds the state hand-off from D's `AsyncLatestQueue` and never imports
`LatestQueue` for it. "LatestQueue" on its own means A's queue.

## What each layer provides

### A, perception (`coach/sources`, `coach/perception`)

| Call | Contract |
|---|---|
| `parse_source(spec, fast=False)` | `"cam:N"` or `"file:PATH"` → an iterable of `(t, frame_idx, bgr)` with `.close()`. `t` is `time.monotonic()` at capture. Files play at 1x and skip frames when late; `fast=True` (tests) plays every frame with no waiting. |
| `PoseTracker(cfg, holdmap=, shapes=)` | `.process(t, frame_idx, bgr) -> ClimberState` and `.close()`. Stateful, so call it from one thread only. Contacts are only filled in when a hold map is given. |
| `load_holdmap_with_shapes(problem_id)` | `data/holdmaps/<id>.json` → `(HoldMap, shapes)`. Raises `FileNotFoundError` when there's no map. |
| `load_perception_config()` | the `perception` section of `config/default.yaml` |
| `EditorState(image, cfg)` (`holdmap_editor`) | The hold map editor's logic with no window: `pick_colour` (runs the Roboflow detector, keeps holds of the clicked colour), `toggle_at`, `delete_at`, `erase_rect`, `add_box`, `mark`, `undo`, `clear_colour`, `load_existing`, `build_with_shapes(problem_id)`. D's dashboard editor drives it over HTTP. |
| `save_holdmap(holdmap, root, shapes)`, `holdmap_path(problem_id, root)`, `load_holds_config()` | Write a map (with polygons) to `data/holdmaps/<id>.json`; where it lives; the `holds` config section. |
| pose-cache JSONL | one `ClimberState` per line (`coach/perception/serialize.py`); `t` is video time in seconds |

### B, brain (`coach/brain`)

| Call | Contract |
|---|---|
| `Brain(config, bus, voice)` | Subscribes to `"holdmap"` (waits up to 1 s at start) and `"spoken"` in the constructor, so build it before perception starts. |
| `await brain.run(states)` | `states` is anything with `async get() -> ClimberState`. Runs on the asyncio loop until cancelled. Publishes `"event"`, `"suppressed"` and `"attempt"`. Uses `state.t` only, never the wall clock. |

### C, voice (`coach/voice`)

| Call | Contract |
|---|---|
| `make_voice(config, bus, no_llm=False)` | returns a `Voice` (the protocol in `contracts.py`) plus `close()`. The phrase bank must already be built. |
| `voice.say(req) -> bool` | Called on the asyncio loop thread; must not block. `False` means it didn't start before the deadline; the line may still play later. |
| `voice.is_speaking()` | `True` for the whole of a debrief, including the gaps between streamed sentences (fixed in #12). |
| **Every voice publishes** | `"spoken"` with a `SpokenLine` when audio actually starts, and `"debrief_text"` per debrief sentence. B's compliance and mood, and the dashboard, depend on it. |

`--voice print` uses D's `stubs.PrintVoice`, because C's `coach/voice/print_voice.py` takes no
bus and so never publishes `"spoken"`. **Ask for C:** give `PrintVoice` an optional `bus` and publish
`SpokenLine`. Then `run.py` can switch to it.

### D, wiring and dashboard (`run.py`, `coach/stubs.py`, `coach/server`, `web/`)

| Piece | Contract |
|---|---|
| `run.py` | Builds the bus, `AsyncLatestQueue`, voice, brain, server and bus log in that order, then starts perception. Ctrl+C stops perception, cancels tasks, closes `/ws` and `/video` connections, closes the voice, flushes the log. |
| `PoseThreads` (`run.py`) | Real perception from A's parts. Falls back to `stubs.VideoPerception` (video, no pose) if A's code isn't on the branch or the pose model fails to load. |
| `coach.server.create_app(bus, hello, control=)`, `serve(bus, host, port)` | The dashboard: `/ws` sends every bus item as `{"topic", "t", "payload"}`; `/video` is MJPEG. With a `control` (run.py's `Session`): `GET /api/sources`, `POST /api/source {"spec"}` to switch, and `WS /camera` for webcam JPEGs from a dashboard. |
| `Session` (`run.py`) | The current source, its perception workers and a brain. `switch(spec)` swaps to any clip under `videos/` or the browser camera while the server, voice and bus log keep running. The brain is rebuilt, so attempts and the hold map start fresh. A clip named `<problem>_<NN>_<name>` gets that problem id. |
| `HoldMapBuilder` (`coach/server/holds.py`) | Builds a problem's hold map from the dashboard: a still of the problem's empty-wall clip (or the current frame), one click on a route hold runs A's Roboflow detector through `EditorState`, then toggle, erase, add, mark start/top, save. Roboflow runs only then, off the event loop, never in the live loop. Picking the colour again starts over in a fresh `EditorState` (A's `pick_colour` keeps hand-placed and loaded holds and adds the detected ones on top). Opening calls `Session.pause()` (perception, brain and voice stop; the last frame is kept); leaving calls `Session.resume()`, and saving calls `Session.reload()`: both restart perception and the brain on the same source, so A's contacts and B's brain both load the new map. Stills are compared with the video's newest frame (ORB features) and the ones filmed from another camera position are listed last and flagged. A video still (an empty-wall clip, or the clip being analysed) can be any moment: `Session.read_still(key, at)`, and the `frame` op swaps the picture while keeping the holds. Uploaded pictures (`POST /api/holds/upload`) are kept as JPEGs in `data/stills/<problem>/` and offered with the others. No change to A's or B's code. |
| `BrowserCamera` (`coach/server/camera.py`) | A frame source fed by `/camera`: iterate it for `(t, frame_idx, bgr)` like A's camera. `t` is arrival time. `PoseThreads` and `VideoPerception` both accept it in place of a `cam:`/`file:` source. |
| bus log | `data/logs/bus-<problem>-<stamp>.jsonl`, every topic except `"frame"`. |

## Shared rules

- **Threads.** A's code runs in D's perception threads. B and C run only on the asyncio loop
  thread. `bus.publish()` is the only thing called from both, and it's thread-safe.
- **Time.** `ClimberState.t` is monotonic capture time from the source. Pose caches store video
  time; `run.py --replay-pose` shifts it to now so latency figures stay real.
- **Frames.** `"frame"` payloads are JPEG `bytes`. They never go in the bus log.
- **Payloads.** Everything else on the bus is a dataclass from `contracts.py` (`"debrief_text"` is
  a `str`).
- **Hold ids** are numbered from the bottom (1) to the top (N) everywhere inside the system. Spoken cues may
  describe holds relative to the climber instead (see open decisions).

## Merge order

`daniel` and `a/feat-perception-pipeline` can now merge into `main` in either order. A combined
trial merge of `main` + A + `daniel` passes the whole test suite and runs `run.py` end to end.

## Open decisions for the team

1. **`high_point` counting feet.** Today a foot reaching a new hold counts as a high point.
   Hands-only is the usual climbing meaning. B is happy either way; the team should decide.
2. **Spoken cues: hold numbers or relative directions** ("left hand, up and left").
   `CoachEvent.data["dir"]` already exists for this, so no contract change is needed: B fills it, C speaks it.
3. **Climb 2 data.** A to push `data/pose_cache/*.jsonl` and the Climb 2 hold map (shared drive or
   LFS), so B can run the real end-to-end test.
4. **Hold map mid-run (B, optional).** The brain reads the hold map once, when it starts, so a map
   saved from the dashboard restarts analysis (attempt history resets). If the team wants to keep
   the history, B could let `Brain` take a new `"holdmap"` from the bus while running; D would then
   publish it instead of restarting. Not needed for the demo.
5. **Footholds in the colour filter (A, optional).** On our Climb 1 footage, clicking a yellow hold
   also keeps some grey screw-on footholds (`hue_tolerance` / `min_saturation` in `holds`). The
   editor lets you leave them out by hand, but a tighter filter would save clicks.
6. **Hold maps belong to a camera position.** A map only fits footage filmed from where its still
   was taken. A's `climb2` map is different footage from our `climb_2_*` clips, so the clips get
   problem `climb_2` and their own map. Even the two `climb_2_00_empty_wall` clips differ: only
   `climb_2_00_empty_wall (1).mp4` matches `climb_2_05_hand_searching` (the other is 9.7% of the
   width off). The dashboard editor checks each still against the video and offers the match first.
