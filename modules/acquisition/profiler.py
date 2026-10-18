"""Lightweight per-stage timing for the acquisition loop.

Usage inside the loop::

    with profiler.time("bin"):
        ...
    profiler.tick()          # logs the averages every ``log_every`` frames

When ``enabled`` is False every call is a cheap no-op (the context manager still
exists, so call sites do not branch).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


class StageProfiler:
    """Average of how long each named pipeline stage takes (milliseconds).

    Each log line averages the samples recorded since the previous line, so a
    stage that stopped running (or one timed only on sampled frames) never
    shows a stale value.

    Args:
        enabled: Record and log timings; ``False`` makes the profiler inert.
        log_every: How many ticks pass between log lines.
        detail_every: :meth:`want_detail` is true on one frame in this many
            (``0`` disables detail timing).
        label: Optional callable returning a short ``str`` (e.g. ``"backend=cpu"``)
            shown in each log line.  When its value changes, the samples are
            cleared so timings of the previous configuration do not linger.
        shown_stage: Optional stage name recorded once per displayed frame; its
            sample count gives the ``shown=`` rate in the log line.

    Each log line also carries ``fps=``: ticks per second since the previous
    line (or since the label last changed).

    Threading:
        Not thread-safe; use one profiler per acquisition thread.
    """

    class _Span:
        __slots__ = ("_prof", "_name", "_t0")

        def __init__(self, prof: "StageProfiler", name: str):
            self._prof = prof
            self._name = name

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            self._prof._record(self._name, (time.perf_counter() - self._t0) * 1000.0)

    def __init__(self, enabled: bool = True, log_every: int = 90, label=None, detail_every: int = 10,
                 shown_stage: str | None = None):
        self.enabled = enabled
        self.log_every = log_every
        self.detail_every = detail_every
        self.label = label
        self.shown_stage = shown_stage
        self._label_value = None
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._frame_count = 0
        self._period_start = time.perf_counter()
        self._period_ticks = 0
        self._counts: dict[str, int] = {}

    def time(self, name: str) -> "StageProfiler._Span":
        """Context manager that records the duration of the ``with`` block under ``name``."""
        return self._Span(self, name)

    def add_count(self, name: str, n: int):
        """Add ``n`` to the counter ``name``, shown as ``name=total`` in the next line's header."""
        if self.enabled:
            self._counts[name] = self._counts.get(name, 0) + int(n)

    def want_detail(self) -> bool:
        """True on one frame in ``detail_every`` while enabled.

        Stages that cost extra to time (the GPU per-kernel breakdown) are
        timed only on these frames.
        """
        return self.enabled and self.detail_every > 0 and self._frame_count % self.detail_every == 0

    def _record(self, name: str, dt_ms: float):
        if not self.enabled:
            return
        self._samples[name].append(dt_ms)

    def tick(self):
        """Count one processed frame and log the averages every ``log_every`` frames."""
        if not self.enabled:
            return
        self._frame_count += 1
        self._period_ticks += 1
        if self.label is not None:
            value = self.label()
            if value != self._label_value:
                # New configuration: drop samples of stages that may no longer run;
                # the rate period starts at this tick.
                self._samples.clear()
                self._counts.clear()
                self._label_value = value
                self._period_start = time.perf_counter()
                self._period_ticks = 0
        if self._frame_count % self.log_every == 0:
            self._log_summary()

    def _log_summary(self):
        if not self._samples:
            return
        parts = [f"{name}={sum(buf) / len(buf):.2f}ms" for name, buf in self._samples.items() if buf]
        now = time.perf_counter()
        elapsed = now - self._period_start
        rates = []
        if elapsed > 0:
            rates.append(f"fps={self._period_ticks / elapsed:.1f}")
            if self.shown_stage is not None:
                rates.append(f"shown={len(self._samples.get(self.shown_stage, ())) / elapsed:.1f}")
        rates.extend(f"{name}={total}" for name, total in self._counts.items())
        self._counts.clear()
        self._samples.clear()
        self._period_start = now
        self._period_ticks = 0
        header = " ".join(([self._label_value] if self._label_value else []) + rates)
        header = f"[profile {header}]" if header else "[profile]"
        logger.info("%s %s", header, "  ".join(parts))
