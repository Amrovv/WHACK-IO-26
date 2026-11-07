"""Single config-reading helper for the coach brain.

Config is always the plain dict loaded from ``config/default.yaml`` (or an
equivalent plain dict built in tests) — no other shape is supported. One
helper, used everywhere a threshold is read, so there is exactly one place
that knows how config is shaped.
"""

from __future__ import annotations

from typing import Any


def cfg(config: dict[str, Any], section: str, name: str) -> Any:
    return config[section][name]
