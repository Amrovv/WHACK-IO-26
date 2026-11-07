# Test climbs to film

A shot list for testing the coach on real footage. Each clip targets one thing the system
must detect (the event kinds in `coach/contracts.py`) or one way perception could get confused.
Tick clips off as you go.

## Before you start

- [ ] **Tripod, fixed camera, landscape.** The system assumes the camera never moves. Don't touch it mid-clip; if you have to, start a new clip.
- [ ] **Whole problem in frame**, from the start holds to the top hold, with a little margin above the top and room for the climber's feet below the start.
- [ ] **1080p, 30 fps or more.** Lock exposure and focus (tap and hold on most phones) so they don't drift as the climber moves.
- [ ] **Pick one problem with a single clear hold colour** and use it for most clips, so the hold map is built once and reused. Note its colour and grade.
- [ ] **Film the empty wall first** (clip 00). The hold map is built from it.
- [ ] Say the clip number out loud at the start of each clip, or film a sign. It makes sorting easy later.
- [ ] Ask the gym before filming other people, and get their OK if they're recognisable.

**File names:** `<problem>_<clip>_<short-name>.mp4`, e.g. `pink-v3_07_stall.mp4`. Write the problem
id, hold colour and anything unexpected in `videos/NOTES.md` while you're at the wall.

## 1. Setup clips (do these first)

| # | Clip | How to film it | What we're checking |
|---|---|---|---|
| 00 | **Empty wall** | 10 s of the problem with nobody on or in front of it. | Hold map: the detector plus colour filter finds every hold of the problem's colour. |
| 01 | **Standing at the base** | Climber walks in, stands under the start holds for 10 s, chalks up, walks off. | No `attempt_start` and no cues while they're on the floor. |
| 02 | **Clean send, steady pace** | Climb the problem smoothly to the top, match the top hold for 2 s, climb or jump down. | `attempt_start`, `top`, an `AttemptSummary` with outcome "top". The baseline clip. |

## 2. One clip per coaching cue

Exaggerate each mistake a little so it's clearly there, and keep everything else clean.

| # | Clip | How to film it | Expected cue |
|---|---|---|---|
| 03 | **Feet low** | Climb with feet left low and arms stretched, hips sagging away from the wall. | `feet_low` |
| 04 | **Bent arms** | Hang on bent arms (elbows well under 140°) for several seconds instead of straight arms. | `arms_bent` |
| 05 | **Hand searching** | Reach up and pat around near the next hold for 1–2 s before grabbing it. | `hand_searching` |
| 06 | **Feet cut** | On a steep move, let both feet swing off, then get them back on. | `feet_cut` |
| 07 | **Stall** | Stop on one hold mid-problem and stay there for 12 s or more. | `stalling`, getting firmer at about 4, 7 and 10 s |
| 08 | **Quiet feet** | Place every foot slowly and precisely, no re-adjusting. | `quiet_feet` (praise) |
| 09 | **Noisy feet** | Stamp and re-adjust feet several times on each foothold. | No `quiet_feet`; possibly a footwork cue |
| 10 | **Weight not shifted** | Reach far sideways with the hips still over the other foot. | `weight_shift` |
| 11 | **Big reach coming up** | A move where the next hold is near full arm span. | `route_preview` before the move |
| 12 | **New high point** | Fall higher than any earlier attempt that session (film after clip 13). | `high_point` |

## 3. Attempt endings and memory

The coach should remember earlier attempts. Film these as **one continuous session** (same
problem, camera not moved, ideally one long recording) with short rests between goes.

| # | Clip | How to film it | Expected |
|---|---|---|---|
| 13 | **Fall at the crux** | Fall off the same hard move (say hold 6) on 2–3 attempts in a row. | `fall`, summaries with the same high hold, then `crux_warning` on the next attempt as they near hold 6 |
| 14 | **Down-climb** | Climb halfway, then climb down to the floor under control. | Outcome "downclimb", not "fall" |
| 15 | **Jump off** | Climb partway and drop off on purpose. | Outcome "fall" |
| 16 | **Told you so** | After a cue (e.g. feet low), ignore it and fall on that same move. | `told_you_so` in a later debrief or cue |
| 17 | **Follow the advice** | Same move, but fix the thing the coach said, and send. | Cue counted as followed; the debrief should notice the improvement |
| 18 | **Rest between attempts** | Sit or stand at the base for 30–60 s between two goes. | Debrief plays between attempts; the next attempt is numbered +1 |

## 4. Making perception's life hard

These show how the system copes when the picture isn't clean. Note what happens; the
right answer is usually "carry on calmly" or "say nothing", not a wrong cue.

| # | Clip | How to film it | What should happen |
|---|---|---|---|
| 19 | **Second climber starts below you** | Mid-attempt, a second person starts climbing a different problem below or next to the climber. | Coach keeps tracking the first climber and doesn't cue the second |
| 20 | **Person walks in front of the camera** | Someone walks between the camera and the wall mid-climb. | Brief loss of tracking, no false fall or cue, recovers |
| 21 | **Spotter standing close** | A spotter stands under the climber with arms up the whole time. | Spotter isn't mistaken for the climber |
| 22 | **Two people on the same problem** | Two people climb the same problem side by side (if allowed). | Note which one it tracks |
| 23 | **Climber partly out of frame** | Problem tops out at the edge of the frame, or the climber's feet go out of shot. | Tracking survives; no false fall |
| 24 | **Bright and dark clothing** | Same climb twice: once in clothing that matches the wall colour, once in high contrast. | Pose is found in both |
| 25 | **Clothing the same colour as the holds** | Climber wears a top the colour of the problem's holds. | Holds not confused with the climber |
| 26 | **Overlapping problem** | A problem of a different colour shares holds or crosses ours. | Only our colour's holds count as contacts |
| 27 | **Backlight or lighting change** | Face a window, or have lights change mid-clip if possible. | Pose and contacts hold up |
| 28 | **Fast dynamic move** | A dyno or a quick throw to a hold. | Contact detected after the catch; cues arrive in time (~0.8 s) |
| 29 | **Camera nudged** | Gently bump the tripod once mid-clip. | We see how badly the hold map drifts (expect it to; useful to know) |

## 5. Demo-ready clips

Film a couple of clean, watchable sessions to use as backup videos on stage if live filming
fails.

| # | Clip | How to film it |
|---|---|---|
| 30 | **Demo session A** | 3 attempts on one problem: fall at the crux, fall a bit higher, then send. Good light, nobody else in frame. |
| 31 | **Demo session B** | Same idea on a second problem with a different hold colour, to show the hold map isn't hard-coded. |

## After filming

- Copy clips to the shared drive, under `data/` locally (it's gitignored, so don't commit videos).
- For each clip, write down roughly when the key moments happen (e.g. "07: stall starts 0:12, ends 0:26"). It makes it easy to check whether cues fired at the right time.
- Run a clip through the system: `uv run python run.py --source file:data/<clip>.mp4 --problem <id> --voice print`
