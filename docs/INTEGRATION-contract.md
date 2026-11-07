# B integration contract — how to wire the coach brain

The brain's parts are all built and unit-tested, but nothing assembles them. This is
the contract the orchestrator (D's `run.py`) must follow. `tests/test_pipeline.py` is
the executable version of this document — if you change the wiring, keep that test green.

## Construct once, at startup

```python
ft = FeatureTracker(config["brain"]["move_speed_eps"])
at = AttemptTracker(holdmap, config)  # holdmap may be None
detectors = mvp_detectors(config) + good_tier_detectors(config)
mem = Memory(voice, config)
clock = lambda: current_state.t  # arbiter uses capture time
arbiter = Arbiter(
    voice, config, clock=clock, publish=bus.publish, still_valid=route_still_valid
)  # see "still_valid" below
```

Detectors and trackers hold state across frames — build them **once per run**, not per frame.
`voice` is C's implementation of the `Voice` protocol.

## Per processed frame (one `ClimberState` from A, in order)

1. `features = ft.update(state)`
2. `events = list(at.update(state, features))`  — attempt_start / high_point / top / fall. If
   an `attempt_start` event came back, call `ft.reset_highest_hold()` right after — otherwise
   `highest_hold_id` keeps growing across attempts and `high_point`/`told_you_so` go stale.. If
   this returns an `attempt_start` event, call `ft.reset_highest_hold()` so highest-hold
   tracking starts from zero for the new attempt.
3. **Drain finished attempts into memory** (do this before querying memory):
   ```python
   while seen < len(at.finished_attempts):
       mem.on_attempt_end(at.finished_attempts[seen])
       seen += 1
   ```
4. Build the detector context for this frame:
   ```python
   on_wall = at.phase == "ON_WALL"
   ctx = DetectorContext(config, holdmap, on_wall, last_fall_t, mem.crux_hold())
   ```
5. Run detectors and memory's live view, collecting their events:
   ```python
   for det in detectors:
       events += det.update(features, state, ctx)
   events += mem.observe(state, features)  # stall bookkeeping, told_you_so
   ```
6. Feed and offer every event:
   ```python
   for ev in events:
       mem.on_event(ev)  # habit counting
       arbiter.offer(ev, mem.level_for(ev.kind), mem.mood())
   ```
7. Decide and speak: set the clock to `state.t`, then `spoken = arbiter.tick()`.
8. If an event was spoken and its kind is `"fall"`, set `last_fall_t = spoken.t`
   (this is what mutes cue detectors for `cue_mute_after_fall_s` — the brain cannot
   mute itself without this).

## From C, asynchronously

When C actually plays a line it publishes a `SpokenLine` on the `spoken` topic. Route it to
`mem.on_spoken(spoken_line)` — this is what drives compliance tracking, `mood()`, and
`told_you_so`. Without it, compliance is always empty and mood stays neutral.

## Things that silently break if you skip them

- **`last_fall_t` not set** → no cue is muted after a fall (every cue fires on the way down).
- **`mem.observe()` not called every frame** → `crux_hold()` stays empty and `told_you_so` never fires.
- **`mem.on_spoken()` not wired** → compliance/mood never update; the coach never sounds impressed.
- **`mem.on_attempt_end()` not called** → `level_for()` never rises (no growing specificity) and
  no debrief fires.
- **events offered without `mem.level_for()/mood()`** → lines lose their specificity and tone.

## still_valid (freshness)

The arbiter takes an optional `still_valid(event) -> bool` it calls just before speaking, to drop
a cue whose condition has passed (e.g. the searching hand already found the hold). Each detector
exposes `still_valid(event, state)`. The orchestrator should route an event to the `still_valid`
of the detector that produced it (track event->detector, or dispatch by `event.kind`). If you pass
`still_valid=None`, every in-TTL event is treated as still valid — acceptable for a first cut.

## Clocks

Everything downstream of A keys off `state.t` (capture time, monotonic). The arbiter's injected
`clock` must return the current frame's `state.t` at `tick()` time. Never mix in the wall clock, or
file replays and the live camera diverge.

## Hold-map file schema (what A's perception writes, what B's `Hold` accepts)

`Hold` has exactly four fields: `id`, `center`, `bbox`, `role`. A's hold-map JSON may carry
extra fields per hold (e.g. an optional `"polygon"` for diagonal holds) for perception's own
use — `scripts/replay_brain.py` and any other `Hold` loader must build `Hold` from only the
four contract fields and ignore the rest, never `Hold(**hold)`.

## Known decision: `high_point` / `highest_hold_id` counts feet, not just hands

`FeatureTracker.highest_hold_id` and `AttemptTracker`'s `high_point` event both take the max
hold id across **all four limbs' contacts**, not just the hands. This is intentional (matches
the B guide's "highest hold touched" wording) and legitimate climbing moments (heel hooks,
high steps) do reach a foot higher than any hand has been. If stray `high_point` cues or an
inflated high-point number show up on real footage, the likely cause is a bad foot-contact
reading from perception, not a brain bug — flag it rather than assuming the brain is wrong.
Switching to hands-only is a one-line change in each of `features.py`/`attempts.py` if the
team decides the cue should mean something different — get agreement first, it changes the
cue's meaning, not just a threshold.
