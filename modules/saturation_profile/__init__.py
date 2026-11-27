"""Saturation level profile: measure what a clipped raw pixel reads, per camera, format, bits and bin.

The level a camera's frames clip at is measured on real saturated frames, never computed from the
bit depth: an exposure sweep on a bright scene, plateau analysis of the top of the raw histogram
(:func:`analyse`), and a verdict - ``verified`` / ``inconclusive`` / ``failed`` - stored per
configuration in the camera profile's ``saturation/`` folder (:class:`SaturationProfileStore`).
Only a verified profile, inside the gain range it was verified over, gives the threshold for
*confirmed clipping*; everything else means *not checked*. Quality data only: the intensities,
the nominal full scale and every other calibration are never touched.

* :func:`measure` / :class:`SweepPlan` / :class:`FrameSource` - the sweep (``core.py``);
* :func:`analyse`, :func:`step_candidates`, :class:`PlateauParams` - the plateau analysis;
* :class:`SaturationProfileStore` / :class:`ProfileLookup` - files, history, compatibility (``store.py``);
* :class:`ProfileLevelSource` - the level of the active configuration for the acquisition engine
  (``level.py``); :class:`~.sources.BackendFrameSource` - a sweep straight on a backend (``sources.py``);
* :class:`SaturationMeasureWidget` - the calibration window (``widget.py``, loaded lazily);
  ``python -m modules.saturation_profile`` runs it on a simulated camera (``demo.py``).

Data conventions:
    Raw integer frames as the camera delivers them; levels in raw counts; exposure in ms; gain in
    the backend's units. File format: :mod:`modules.contracts.saturation_profile`.

Threading:
    The analysis is pure; :func:`measure` blocks (run it on a worker thread); the store does plain
    file I/O (one writer at a time).

Example:
    >>> import numpy as np
    >>> from modules.saturation_profile import step_from_frames, step_candidates
    >>> frame = np.full((40, 40), 1000, np.uint16); frame[:10] = 65504
    >>> step_candidates(step_from_frames([frame], 1, 10.0, 65520), 65520)
    {65504: 400.0}
"""
from .core import (DEFAULT_PARAMS, FrameSource, MeasurementCancelled, PlateauParams, StepFrames, SweepPlan,
                   analyse, failed_record, frame_timeout_s, fully_saturated, measure, step_candidates,
                   step_from_frames, top_histogram)
from .store import (STATE_FAILED, STATE_GAIN_OUTSIDE, STATE_INCOMPATIBLE, STATE_INCONCLUSIVE, STATE_INVALID,
                    STATE_MISSING, STATE_VERIFIED, ProfileLookup, SaturationProfileStore, SaveResult,
                    software_revision)
from .level import ProfileLevelSource

__all__ = ["PlateauParams", "DEFAULT_PARAMS", "StepFrames", "top_histogram", "step_from_frames", "step_candidates",
           "fully_saturated", "analyse", "failed_record", "MeasurementCancelled", "FrameSource", "SweepPlan",
           "frame_timeout_s", "measure", "SaturationProfileStore", "ProfileLookup", "SaveResult",
           "software_revision", "STATE_VERIFIED", "STATE_MISSING", "STATE_INCONCLUSIVE", "STATE_FAILED",
           "STATE_INCOMPATIBLE", "STATE_INVALID", "STATE_GAIN_OUTSIDE", "ProfileLevelSource",
           "SaturationMeasureWidget"]
__requires__ = ("contracts", "saturation_mask")
__widgets__ = ("SaturationMeasureWidget",)


def __getattr__(name):
    """Import the Qt widget only when it is requested, so ``import modules.saturation_profile`` stays Qt-free."""
    if name == "SaturationMeasureWidget":
        from .widget import SaturationMeasureWidget
        return SaturationMeasureWidget
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
