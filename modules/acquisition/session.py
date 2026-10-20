"""Qt-free scan session: state machine, recording hand-over and visual-selection preview.

A *session* tracks one scan from "start" to "cube ready" without knowing about windows,
cameras or stages:

    PREVIEW --begin()--> SCANNING --finish(...)--> PREVIEW   (cube / reference preview)
                            |--fail(...)--------> PREVIEW   (partial scan discarded)
    any state --set_error()--> ERROR

The GUI controller drives it (arming the worker, moving the stage, updating widgets) and
reads results from the returned :class:`ScanOutcome`.

Data conventions:
    ``images`` are the recorded binned frames ``(slit, wavelength_bin)``; the cube built
    from them by :mod:`modules.cube_io` is ``(slit, scan, wavelength)``. ``raw`` frames
    (Keep Spectrogram) are ``(slit, spectral_pixel)`` sensor frames, index-aligned with
    ``images``.

Threading:
    GUI thread only (the worker uses :class:`ScanRecorder` for its side).

Example:
    >>> import numpy as np
    >>> s = ScanSession()
    >>> s.begin(); s.state
    'SCANNING'
    >>> outcome = s.finish([np.zeros((4, 3), np.uint16)] * 2, raw=[])
    >>> outcome.kind, s.state, len(outcome.images)
    ('cube', 'PREVIEW', 2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ScanOutcome:
    """What a finished scan produced.

    Attributes:
        kind: ``"cube"`` (frames to build a cube from), ``"empty"`` (nothing recorded),
            ``"reference"`` (visual-selection preview ready) or ``"reference_failed"``.
        images: Binned frames (``kind == "cube"``).
        raw: Raw sensor frames kept by Keep Spectrogram (may be empty).
        request_id: Visual-selection request this outcome answers, if any.
        reference_columns: RGB columns ``(slit, 3)`` collected for a visual-selection preview.
        error: Reason for ``"reference_failed"``.
    """

    kind: str
    images: list = field(default_factory=list)
    raw: list = field(default_factory=list)
    request_id: Optional[int] = None
    reference_columns: list = field(default_factory=list)
    error: str = ""


class ScanSession:
    """State and buffers of the scan currently in progress (or idle)."""

    PREVIEW = "PREVIEW"
    SCANNING = "SCANNING"
    ERROR = "ERROR"

    def __init__(self):
        self.state = self.PREVIEW
        self.recording = False
        self.images: list = []
        self.spectrograms: list = []
        self.sat_per_frame: list = []
        self.reference_preview_id: Optional[int] = None
        self.reference_columns: list = []

    # ── starting ──────────────────────────────────────────

    def begin(self) -> None:
        """Clear the previous scan's buffers and enter ``SCANNING`` (recording stays off)."""
        self.images = []
        self.spectrograms = []
        self.sat_per_frame = []
        self.recording = False
        self.state = self.SCANNING

    def start_reference_preview(self, request_id: int) -> None:
        """Mark the next scan as a visual-selection preview (no cube is published)."""
        self.reference_preview_id = request_id
        self.reference_columns = []

    def cancel_reference_preview(self) -> None:
        """Forget a visual-selection request that could not start."""
        self.reference_preview_id = None
        self.reference_columns = []

    # ── during the scan ───────────────────────────────────

    def add_reference_column(self, columns: np.ndarray) -> None:
        """Append the RGB column ``(slit, 3)`` of one recorded frame of a visual-selection scan."""
        self.reference_columns.append(columns)

    # ── ending ────────────────────────────────────────────

    def finish(self, images: list, raw: list) -> ScanOutcome:
        """End the scan normally and return what to do with the recording.

        Args:
            images: Binned frames taken from the worker's recorder (after trimming).
            raw: Raw frames taken from the worker's recorder.

        Returns:
            A :class:`ScanOutcome`. For a visual-selection scan ``kind`` is
            ``"reference"`` (columns attached) or ``"reference_failed"`` (no columns),
            and the recording is discarded; otherwise ``"cube"`` or ``"empty"``.
        """
        self.recording = False
        self.state = self.PREVIEW
        if self.reference_preview_id is not None:
            request_id, columns = self.reference_preview_id, self.reference_columns
            self.cancel_reference_preview()
            self.images, self.spectrograms = [], []
            if not columns:
                return ScanOutcome("reference_failed", request_id=request_id,
                                   error="No preview frames captured")
            return ScanOutcome("reference", request_id=request_id, reference_columns=columns)
        self.images, self.spectrograms = images, raw
        return ScanOutcome("cube" if images else "empty", images=images, raw=raw)

    def fail(self) -> Optional[int]:
        """Discard a scan whose motion failed; return the visual-selection request it voids.

        Returns:
            The pending visual-selection ``request_id`` (now cancelled), or ``None``.
        """
        self.recording = False
        self.state = self.PREVIEW
        request_id = self.reference_preview_id
        self.cancel_reference_preview()
        return request_id

    def set_error(self) -> None:
        """Enter the ``ERROR`` state (e.g. the cube could not be built)."""
        self.state = self.ERROR
