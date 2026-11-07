"""C: speaker, tts, voice, persona, llm."""

import importlib


def __getattr__(name: str):
    """Lazy, so that `python -m coach.voice.tts` doesn't import its own module twice."""
    if name in ("Voice", "make_voice"):
        return getattr(importlib.import_module("coach.voice.voice"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
