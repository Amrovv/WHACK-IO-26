"""Run pose over a clip as fast as possible and write one ClimberState per line.

    uv run python scripts/record_pose.py --source file:data/footage/clip.mp4 \
        --out data/pose_cache/clip.jsonl

`t` in the cache is video time in seconds (frame_idx / fps), so replays are deterministic.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without installing

from coach.perception.holds import load_holdmap
from coach.perception.pose import PoseTracker, load_perception_config
from coach.perception.serialize import state_to_line
from coach.sources import FileSource, parse_source

log = logging.getLogger("record_pose")


def record(source: FileSource, tracker: PoseTracker, out: Path) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    n = found = 0
    with open(out, "w", encoding="utf-8") as f:
        for _, idx, frame in source:
            state = tracker.process(idx / source.fps, idx, frame)
            f.write(state_to_line(state) + "\n")
            n += 1
            found += state.person_found
    return n, found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="file:<path>")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--problem", help="hold map id: record real contacts instead of None")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = parse_source(args.source, fast=True)
    if not isinstance(source, FileSource):
        parser.error("record_pose needs a file: source")
    holdmap = load_holdmap(args.problem) if args.problem else None
    tracker = PoseTracker(load_perception_config(), holdmap=holdmap)
    try:
        n, found = record(source, tracker, args.out)
    finally:
        source.close()
        tracker.close()
    log.info("wrote %d states to %s (person found in %.0f%%)", n, args.out, 100 * found / max(n, 1))


if __name__ == "__main__":
    main()
