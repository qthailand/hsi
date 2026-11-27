"""What a recording keeps of each frame's saturation, and when a level counts as evidence.

* :class:`LevelEvidence` - the level frames are checked against, or why there is none. Hosts
  build it from a **verified saturation profile** (``modules.saturation_profile``: the value a
  clipped pixel was measured to read); no profile, an unverified one, another configuration or
  a gain outside the verified range gives ``level=None``: the frame is *not checked*, never
  "no saturation". The nominal full scale of the bit depth is never evidence.
* :class:`SaturationRecord` - one frame's :class:`~.core.FrameSaturation` in a compact form:
  values only at the near-saturated cells. A frame with nothing near saturation holds nothing.
* :func:`scan_status` - ``checked`` / ``partial`` / ``not_checked`` of a recording's records
  (``None`` = that frame was not checked).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .core import NEAR_RATIO, POLICY, FrameSaturation, pack_mask, unpack_mask

STATUS_CHECKED = "checked"
STATUS_PARTIAL = "partial"
STATUS_NOT_CHECKED = "not_checked"


# ── the level ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LevelEvidence:
    """The level frames are checked against, or why they cannot be.

    Attributes:
        level: Counts a clipped raw pixel reads, or ``None`` (no evidence: do not check).
        source: ``'measured'`` (a verified saturation profile), ``'error'`` (the lookup failed:
            may work next time) or another word saying why there is no level (``'missing'``,
            ``'inconclusive'``, ``'incompatible'``, ``'gain_outside'``, ...).
        reason: One line for logs and tooltips (for ``'measured'``: the provenance itself).
        provenance: Where the level comes from (the profile and its measurement), or ``None``.
        profile_id: The saturation profile's id (``<file stem>@<measured_at>``), when one was found.
        profile_status: That profile's state (``verified``, ``inconclusive``, ...), when found.
    """

    level: Optional[int]
    source: str
    reason: str
    provenance: Optional[str] = None
    profile_id: Optional[str] = None
    profile_status: Optional[str] = None

    @property
    def trusted(self) -> bool:
        """Whether there is a level to check frames against."""
        return self.level is not None

    @property
    def transient(self) -> bool:
        """Whether asking again may give another answer (an SDK query failed)."""
        return self.source == "error"


# ── one recorded frame ────────────────────────────────────────────────────

@dataclass(frozen=True)
class SaturationRecord:
    """One recorded frame's saturation, compact (see :meth:`from_frame`, :meth:`to_frame`).

    Everything is stored at the *near-saturated* positions of the binned line (clipped ones are
    among them), in flat C order; the positions themselves are sparse (``near_index``) or a
    bit-packed mask (``near_packed``), whichever is smaller for this frame.

    Attributes:
        shape: ``(rows, bands)`` of the frame's binned line.
        near_index: ``uint32`` flat indices of the near-saturated cells (sparse form), or ``None``.
        near_packed: :func:`~.core.pack_mask` of ``near_saturation_mask`` (dense form), or ``None``.
            Both ``None``: nothing near saturation (then nothing clipped).
        clipped_at: Packed bits (``bitorder='little'``), one per near position: clipped there;
            ``None`` when nothing clipped.
        near_count, clipped_count: Affected corrected pixels of the band (``fraction * w``) at each
            near position (``uint8``; ``uint16`` for bins wider than 255 pixels); ``clipped_count``
            ``None`` when nothing clipped.
        raw_clipped_fraction, raw_near_fraction, level, near_ratio, policy: As in
            :class:`~.core.FrameSaturation`.
    """

    shape: Tuple[int, int]
    near_index: Optional[np.ndarray]
    near_packed: Optional[np.ndarray]
    clipped_at: Optional[np.ndarray]
    near_count: Optional[np.ndarray]
    clipped_count: Optional[np.ndarray]
    raw_clipped_fraction: float
    raw_near_fraction: float
    level: int
    near_ratio: float = NEAR_RATIO
    policy: str = POLICY

    def any_near(self) -> bool:
        """Whether any band of the frame is near saturation (clipped included)."""
        return self.near_index is not None or self.near_packed is not None

    def any_clipped(self) -> bool:
        """Whether any band of the frame is affected by a clipped raw pixel."""
        return self.clipped_at is not None

    @property
    def nbytes(self) -> int:
        """Bytes held by the arrays (what a recording keeps per frame)."""
        return sum(a.nbytes for a in (self.near_index, self.near_packed, self.clipped_at, self.near_count,
                                      self.clipped_count) if a is not None)

    @classmethod
    def from_frame(cls, frame: FrameSaturation, width) -> "SaturationRecord":
        """Compact ``frame``; ``width`` is the binner's ``w`` (pixels per band, the bands of
        ``frame``), which turns the affected fractions back into exact counts."""
        near = np.ascontiguousarray(frame.near_saturation_mask)
        shape = tuple(int(v) for v in near.shape)
        at = np.flatnonzero(near)
        scalars = (float(frame.raw_clipped_fraction), float(frame.raw_near_fraction), int(frame.level),
                   float(frame.near_ratio), frame.policy)
        if at.size == 0:
            return cls(shape, None, None, None, None, None, *scalars)
        width = np.asarray(width)
        if width.shape != (shape[1],):
            raise ValueError(f"width has {width.shape} entries, the frame {shape[1]} bands")
        count_type = np.uint8 if int(width.max(initial=0)) <= np.iinfo(np.uint8).max else np.uint16
        w = width[at % shape[1]].astype(np.float64)

        def counts(fraction):
            return np.rint(np.ascontiguousarray(fraction).ravel()[at] * w).astype(count_type)

        cells = shape[0] * shape[1]
        if at.size * 4 < (cells + 7) // 8:                     # indices smaller than the packed mask
            near_index, near_packed = at.astype(np.uint32), None
        else:
            near_index, near_packed = None, pack_mask(near)[0]
        clipped_here = np.ascontiguousarray(frame.clipped_mask).ravel()[at]
        if clipped_here.any():
            clipped_at = np.packbits(clipped_here, bitorder="little")
            clipped_count = counts(frame.affected_fraction)
        else:
            clipped_at = clipped_count = None
        return cls(shape, near_index, near_packed, clipped_at, counts(frame.near_affected_fraction),
                   clipped_count, *scalars)

    def to_frame(self, width) -> FrameSaturation:
        """The :class:`~.core.FrameSaturation` this record was made from (``width`` as given to
        :meth:`from_frame`); fractions are ``count / w`` in float32, as :func:`~.core.bin_masks`
        computes them."""
        rows, bands = self.shape
        cells = rows * bands
        near = np.zeros(cells, dtype=bool)
        clipped = np.zeros(cells, dtype=bool)
        near_fraction = np.zeros(cells, dtype=np.float32)
        clip_fraction = np.zeros(cells, dtype=np.float32)
        if self.any_near():
            if self.near_index is not None:
                at = self.near_index.astype(np.intp)
            else:
                at = np.flatnonzero(unpack_mask(self.near_packed, self.shape))
            near[at] = True
            w = np.asarray(width)[at % bands]
            near_fraction[at] = self.near_count / w
            if self.clipped_at is not None:
                here = np.unpackbits(self.clipped_at, count=at.size, bitorder="little").astype(bool)
                clipped[at[here]] = True
                clip_fraction[at] = self.clipped_count / w
        return FrameSaturation(clipped.reshape(self.shape), near.reshape(self.shape),
                               clip_fraction.reshape(self.shape), near_fraction.reshape(self.shape),
                               self.raw_clipped_fraction, self.raw_near_fraction, self.level,
                               self.near_ratio, self.policy)


def scan_status(records: Sequence[Optional[SaturationRecord]]) -> str:
    """``checked`` when every frame has a record, ``not_checked`` when none has (or there are
    no frames), else ``partial`` (``None`` entries are the frames not checked)."""
    checked = sum(1 for r in records if r is not None)
    if checked == 0:
        return STATUS_NOT_CHECKED
    return STATUS_CHECKED if checked == len(records) else STATUS_PARTIAL


def scan_summary(records: Sequence[Optional[SaturationRecord]]) -> dict:
    """Counts of a recording: ``status``, ``frames``, ``checked``, ``near_frames`` and
    ``clipped_frames`` (frames with any band affected), ``bytes`` held by the records."""
    checked = [r for r in records if r is not None]
    return {"status": scan_status(records), "frames": len(records), "checked": len(checked),
            "near_frames": sum(1 for r in checked if r.any_near()),
            "clipped_frames": sum(1 for r in checked if r.any_clipped()),
            "bytes": sum(r.nbytes for r in checked)}
