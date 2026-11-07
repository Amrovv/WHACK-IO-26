"""Pub/sub bus: everyone publishes here; D turns it into the dashboard and the bus log.

Owned by B. Change only by PR + a message in the team chat.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from enum import Enum
from pathlib import Path
from typing import Any


def to_jsonable(obj: Any) -> Any:
    """Recursively turn a contracts dataclass (or any nested value) into plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {
            (k.value if isinstance(k, Enum) else k): to_jsonable(v) for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


class Bus:
    """Tiny pub/sub with timestamps. publish() is safe to call from any thread.

    Subscribers get the original payload object (not JSON); to_jsonable() is only
    used for the optional bus log and by D's server when forwarding to the WebSocket.
    """

    def __init__(self, log_path: str | Path | None = None) -> None:
        self._topics: dict[str, list[asyncio.Queue]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._log_file = Path(log_path).open("a", encoding="utf-8") if log_path else None

    def bind_loop(self, loop: asyncio.AbstractEventLoop | None = None) -> None:
        """Call once from inside the running asyncio loop, before any other thread publishes."""
        self._loop = loop or asyncio.get_running_loop()

    def subscribe(self, topic: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._topics.setdefault(topic, []).append(queue)
        return queue

    def publish(self, topic: str, payload: Any) -> None:
        if self._log_file is not None:
            record = {"t": time.monotonic(), "topic": topic, "payload": to_jsonable(payload)}
            self._log_file.write(json.dumps(record) + "\n")
            self._log_file.flush()

        queues = self._topics.get(topic)
        if not queues:
            return

        def _deliver() -> None:
            for q in queues:
                q.put_nowait(payload)

        if self._loop is not None:
            self._loop.call_soon_threadsafe(_deliver)
        else:
            _deliver()

    def close(self) -> None:
        if self._log_file is not None:
            self._log_file.close()


if __name__ == "__main__":
    import logging

    from coach.contracts import CoachEvent

    logging.basicConfig(level=logging.INFO)

    async def _demo() -> None:
        bus = Bus()
        bus.bind_loop()
        queue = bus.subscribe("event")
        bus.publish(
            "event", CoachEvent(t=time.monotonic(), kind="top", priority=0, ttl_s=2.0)
        )
        got = await queue.get()
        print(f"received: {got}")

    asyncio.run(_demo())
