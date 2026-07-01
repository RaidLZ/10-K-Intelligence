"""Lightweight, thread-safe pipeline tracing.

Every stage of the answer pipeline records a timed :class:`~tenk.models.TraceStep`
via the module-level :func:`step`. A tracer is bound to the current context with
:func:`start`, so the modules doing the work never need a tracer passed in — and when
no tracer is active (the smoke tests, direct library use) every call is a no-op.

    from tenk import trace
    tracer = trace.start(on_step=print)   # optional live callback
    trace.step("route", "graph · heuristic")
    ...
    ans.steps = tracer.steps

The same steps stream to the ``tenk`` logger, so `make ask -v` shows them live.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextvars import ContextVar

from tenk.models import TraceStep

log = logging.getLogger("tenk")

# Short glyph per step name, used by the CLI/UI renderers.
ICONS = {
    "llm": "🧠",
    "route": "🧭",
    "decompose": "🪓",
    "retrieve": "🔎",
    "grade": "⚖️",
    "rewrite": "✏️",
    "corrective": "🛟",
    "generate": "✍️",
}


class Tracer:
    """Accumulates timed steps; optionally invokes a callback as each arrives."""

    def __init__(self, on_step: Callable[[TraceStep], None] | None = None) -> None:
        self.steps: list[TraceStep] = []
        self._last = time.perf_counter()
        self._on_step = on_step

    def step(self, name: str, detail: str = "", **meta) -> TraceStep:
        now = time.perf_counter()
        s = TraceStep(name=name, detail=detail, ms=round((now - self._last) * 1000, 1), meta=meta)
        self._last = now
        self.steps.append(s)
        log.info("%-10s %s", name, detail)
        if self._on_step is not None:
            try:
                self._on_step(s)
            except Exception:  # a UI callback must never break the pipeline
                pass
        return s


_current: ContextVar[Tracer | None] = ContextVar("tenk_tracer", default=None)


def start(on_step: Callable[[TraceStep], None] | None = None) -> Tracer:
    """Begin a fresh trace bound to the current context and return it."""
    tracer = Tracer(on_step=on_step)
    _current.set(tracer)
    return tracer


def step(name: str, detail: str = "", **meta) -> None:
    """Record a step on the active tracer, if any (otherwise a no-op)."""
    tracer = _current.get()
    if tracer is not None:
        tracer.step(name, detail, **meta)


def configure_logging(verbose: bool) -> None:
    """Attach a console handler to the ``tenk`` logger so steps stream live.

    Idempotent: safe to call once per CLI invocation. ``verbose`` toggles INFO vs. WARNING.
    """
    log.setLevel(logging.INFO if verbose else logging.WARNING)
    if not any(getattr(h, "_tenk", False) for h in log.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("  · %(message)s"))
        handler._tenk = True  # type: ignore[attr-defined]
        log.addHandler(handler)
    log.propagate = False
