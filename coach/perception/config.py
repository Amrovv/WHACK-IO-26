"""Loads config/default.yaml for the perception modules."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else REPO_ROOT / "config" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
