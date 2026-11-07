"""Structural guards tying the frozen EVENT_KINDS to the config priority map.

This is the test class that would have caught the B3 "downclimb" bug: any event
kind that a detector or the attempt machine can emit must resolve to a priority
(and therefore a ttl) via make_event, or it KeyErrors mid-climb.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from coach.brain.attempts import make_event
from coach.contracts import EVENT_KINDS


def real_config() -> dict:
    return yaml.safe_load(Path("config/default.yaml").read_text(encoding="utf-8"))


def test_every_event_kind_has_a_priority():
    config = real_config()
    priorities = config["brain"]["priorities"]
    missing = [kind for kind in EVENT_KINDS if kind not in priorities]
    assert missing == [], f"EVENT_KINDS with no config priority: {missing}"


def test_make_event_resolves_for_every_event_kind():
    config = real_config()
    for kind in EVENT_KINDS:
        event = make_event(kind, 1.0, {}, config)
        assert event.kind == kind
        assert event.priority in (0, 1, 2)
        assert event.ttl_s > 0  # ttl pulled from arbiter.ttl_s[p{priority}]


def test_priority_map_has_no_unknown_kinds():
    # Priorities should only name real event kinds; "downclimb" is an outcome,
    # not an event, and must NOT appear here.
    config = real_config()
    extra = [kind for kind in config["brain"]["priorities"] if kind not in EVENT_KINDS]
    assert extra == [], f"config priorities names non-EVENT_KINDS: {extra}"


def test_attempt_start_is_priority_zero_so_it_cuts_a_running_debrief():
    # Debriefs keep Voice.is_speaking() True for their whole run (team decision). Only
    # priority 0 interrupts that (arbiter.py); attempt_start must stay priority 0 or every
    # cue in a fast-restarted attempt gets busy-suppressed behind the previous debrief.
    config = real_config()
    assert config["brain"]["priorities"]["attempt_start"] == 0
