# B (coach brain) — status log

Tracked, not gitignored — unlike `design-stages-exp/` (local notes), this is meant to give
anyone (teammate or future agent) precise context on what's shipped, what's deferred, and
what's blocked, without reading every PR. Update this after every merged PR or team decision
that touches the brain.

## Shipped

| PR | What |
|---|---|
| [#4](https://github.com/Amrovv/WHACK-IO-26/pull/4) | A/Platon's review fixes: `Hold(**hold)` crash on hold maps with extra fields; `docs/INTEGRATION-contract.md` tracked (was gitignored); `replay_brain.py` rewritten to run the full pipeline with `--voice print`; 3 real bugs (`highest_hold_id` never reset between attempts, `hip_speed` vertical-only causing false cues on sideways traverses, `AttemptSummary.stalls/cues_given/cues_followed` always empty); config-reading consolidated to one `cfg()` helper. |
| [#5](https://github.com/Amrovv/WHACK-IO-26/pull/5) | C's requirements: `coach.brain.Brain(config, bus, voice)` orchestrator class with `async run(states)` — wraps the full pipeline, subscribes to `holdmap`/`spoken` bus topics, publishes `event`/`attempt`. `hold_id`/`limb` added to `stalling`/`fall`/`feet_low`/`feet_cut`/`weight_shift`/`told_you_so` event payloads (phrasebank specificity). Debrief habits payload now includes `mood`. Debrief asyncio task kept referenced (was GC-able mid-stream). |
| [#6](https://github.com/Amrovv/WHACK-IO-26/pull/6) | Trivial: removed a duplicate sentence in `docs/INTEGRATION-contract.md` (merge artifact from #5). |
| [#7](https://github.com/Amrovv/WHACK-IO-26/pull/7) | P0 keypoint-visibility gate: `_keypoint()` now requires visibility `>= 0.5`, not `> 0` — kills occlusion-driven false cues. Plus 11 edge-case tests from `ACTION-ITEMS.md`. **Found a real bug while testing**: `FeetLow`'s "95% release band" is described in a code comment but not implemented (`_condition()` returns the same value for trigger and release) — marked `xfail`, fixed in #8. |
| [#8](https://github.com/Amrovv/WHACK-IO-26/pull/8) | Fixed the `FeetLow` bug found in #7: `_condition()` now computes trigger and release separately (release at `0.95 * feet_low_ratio`), mirroring `ArmsBent`/`WeightShift`. The `xfail` test from #7 is a real passing test now. Also added this file. |

## In progress

Nothing right now — #8 merged, no open PRs. Next work is blocked (see below) or deferred
pending team review.

## Deferred (2026-11-07 team decision)

Holding off on anything that changes cue *behaviour/semantics* (not just fixes something
broken) until A's perception, C's voice, and D's orchestrator/dashboard are integrated and
the team can review together against real footage, rather than tuning against nothing:

- Stalling-escalation vs. arbiter per-kind cooldown (stalling emits at 4/7/10s, cooldown is 15s
  — only the first ever speaks)
- `high_point` firing on attempt 1 (no prior best to beat yet — arguably should gate to ≥2)
- `fall` event landing ~1s late (waits `ground_settle_s` to distinguish fall from downclimb)
- All P2 new-cue ideas and P3 new event kinds in `design-stages-exp/ANALYSIS-ideas-catalog.md`
  / `ACTION-ITEMS.md`

Confirmed bugs (like FeetLow above) still get fixed immediately — this deferral is only for
judgment calls.

## Blocked on other owners

- **Real footage validation** — need A to push `data/pose_cache/*.jsonl` + a holdmap. `data/`
  is empty locally; nothing has been run against real footage yet.
- **`run.py` / `LatestQueue` crash** — A's queue takes no constructor args and uses thread
  locks; D's `run.py` expects `LatestQueue(loop)` with thread-safe `put()` and `async get()`.
  A and D need to agree on one interface before both branches merge. Not B's folder.
- **PR review** — none open right now.

## Resolved, just documented (not code changes)

- **`say()` returning `False` but the line still plays** (C) — `spoken` (the actual
  `SpokenLine`) is the true record of what was heard, not `suppressed: late`. Brain logic
  already keys off `spoken` correctly; this is a D-side dashboard display nit.
- **Arbiter vs. a running debrief** (C) — decided `Voice.is_speaking()` should stay `True` for
  the whole debrief (covering gaps between streamed sentences), not toggle per-sentence. No
  arbiter change needed; this is C's implementation to get right.
- **`highest_hold_id`/`high_point` counts feet as well as hands** — intentional, matches the B
  guide's "highest hold touched" wording. If it causes problems on real footage (bad foot
  contacts inflating the high point), the fix is hands-only — documented in
  `docs/INTEGRATION-contract.md`'s "Known decision" section.
