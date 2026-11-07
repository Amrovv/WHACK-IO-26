# Version control workflow

Small PRs, `main` always runs, and anything that affects another owner gets announced before it's merged.

## Owners

| Owner | Area | Branch prefix | Folders |
|---|---|---|---|
| A | Perception | `a/` | `coach/sources/`, `coach/perception/`, `scripts/record_pose.py` |
| B | Coach brain | `b/` | `coach/brain/`, `coach/bus.py`, `coach/contracts.py`, `config/default.yaml` structure |
| C | Voice and persona | `c/` | `coach/voice/`, `config/personas/`, `scripts/build_phrasebank.py` |
| D | Product and demo | `d/` | `run.py`, `coach/stubs.py`, `coach/server/`, `web/`, `scripts/bench_latency.py`, `demo/` |

## Branches

- `main` is protected: no direct pushes, PRs only. It must always run end to end (on stubs if needed).
- One branch per task, named `<owner>/<task>`: `a/holdmap-editor`, `b/arbiter`, `c/debrief`, `d/dashboard`.
- A branch should live **one to two hours**, not a day. If it's growing, split it.
- Always branch from an up-to-date `main`:

  ```bash
  git switch main && git pull
  git switch -c b/arbiter
  ```

- After your PR is merged, delete the branch and start the next task from fresh `main`. Don't keep
  committing to a merged branch: PRs are squash-merged, so the old commits no longer match `main`
  and the next merge can silently undo changes.

## The loop

```bash
git switch main && git pull          # 1. start fresh
git switch -c c/speaker              # 2. branch
# ...work, commit often...
uv run ruff format . && uv run ruff check . && uv run pytest   # 3. check
git fetch && git rebase origin/main  # 4. catch up with main
git push -u origin c/speaker         # 5. push
gh pr create --fill                  # 6. open PR (or use the GitHub UI)
```

If you already pushed and then rebase, push with `git push --force-with-lease` (only ever on your
own branch).

## When to merge

A PR is ready when **all** of these are true:

1. `uv run ruff format . && uv run ruff check .` is clean.
2. `uv run pytest` passes.
3. `run.py` still runs end to end (with `--voice print` and stubs for anything unfinished).
4. It only touches your own folders, or the other owner has approved.
5. It's rebased on current `main` with no conflicts.

Then:

- Anyone can review; the folder's owner approves changes to it. Aim to merge **within an hour or two**
  of opening. If nobody has reviewed after 30 minutes, ping the chat.
- Merge with **Squash and merge** so each PR is one commit on `main`. Delete the branch.
- Post "merged `<branch>`: <one line>" in the team chat.

Don't merge:

- Half-wired features that break `run.py`. Put them behind a config flag or keep them on the branch.
- Big files (footage, models, audio). They go on the shared drive; `models/` and `data/` are gitignored.
- Secrets. `.env` is gitignored; add new variable names (no values) to `.env.example`.

## Shared files (extra care)

`coach/contracts.py`, `coach/bus.py` and the **structure** of `config/default.yaml` change only by:

1. A message in the team chat explaining the change, **before** opening the PR.
2. A PR reviewed by B.

Same for changes to event kinds (frozen at H5), contract fields or folder layout. Tuning a threshold
value in `config/default.yaml` is a normal PR.

## Dependencies and `uv.lock`

- Add packages with `uv add <pkg>` (or `uv add --dev <pkg>`), never by editing `pyproject.toml` by hand.
- Commit `pyproject.toml` and `uv.lock` together, in their own small PR if possible, and say so in chat.
- After pulling, run `uv sync`.
- `uv.lock` conflict during a rebase: never hand-edit it. Resolve `pyproject.toml` by hand (keep
  both sides' packages), take `main`'s lockfile, then regenerate:

  ```bash
  git checkout origin/main -- uv.lock
  uv lock
  git add pyproject.toml uv.lock && git rebase --continue
  ```

## Commit messages

Short, imperative, with a type prefix:

```
feat: add LatestQueue with drop-oldest
fix: wrap-around hue distance for red holds
test: arbiter cooldown and budget
chore: add sounddevice dependency
```

## Checkpoints and tags

D runs each checkpoint; everyone pauses new merges until it passes or the team agrees what to cut.

| Checkpoint | Tag | Meaning |
|---|---|---|
| H2 | — | contracts, bus and config merged; `run.py` runs on stubs on every laptop |
| H6 | `h6-skeleton` | walking skeleton: real pose + hold map, stalling cue, dashboard |
| H11 | `h11-mvp` | all MVP cues, real audio, backup video 1 |
| H18 | `h18-freeze` | feature freeze: only fixes from here on |

```bash
git switch main && git pull
git tag h6-skeleton && git push origin h6-skeleton
```

After `h18-freeze`, PRs are bug fixes only, and each one needs a second pair of eyes.

## When `main` breaks

1. Say so in the chat immediately.
2. **Revert first, fix second**: `git revert <sha>` on a branch, PR, merge. Then fix properly on a new branch.
3. If it's late and messy, go back to the last tag: `git switch -c d/rescue h11-mvp`.

## Comms

- One team chat channel for experiment numbers, "blocked on X" and "merged Y".
- Before sleeping, post a three-line handover: **done**, **in progress**, **next**.
