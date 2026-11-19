"""Flat-field correction along the slit of a pushbroom scan (Qt-free).

``compute_flatfield`` fits the relative response of every slit row and band on a scan of a
target that is uniform along the slit and spans it; ``apply_flatfield`` divides a cube by it;
``FlatFieldStore`` keeps profiles per camera and finds the one that fits a scan's setting.
The reflectance scale is not part of this module: pass the profile to
``modules.radiometric.compute_flatfield_factors_array`` with a scan of a reflectance target.

Data conventions:
    Cubes are ``(lines, samples, bands)`` dark-subtracted counts; a profile is float32
    ``(lines, bands)``, about 1, finite and positive.

Threading:
    Synchronous and thread-agnostic; the store does plain file I/O (one thread at a time).

Example:
    >>> from modules.flatfield import compute_flatfield, apply_flatfield
    >>> flat = compute_flatfield(strip_cube, strip_mask)          # doctest: +SKIP
    >>> evened = apply_flatfield(scan_cube, flat['profile'])      # doctest: +SKIP
"""
from .core import (DEFAULT_MAX_SATURATED, DEFAULT_MIN_COVERAGE, DEFAULT_STEP_THRESHOLD, apply_flatfield,
                   check_profile, compute_flatfield)
from .store import (FLATFIELD_FORMAT_VERSION, FlatFieldMatch, FlatFieldProfile, FlatFieldStore,
                    flatfield_mismatch)

__all__ = ["compute_flatfield", "apply_flatfield", "check_profile", "DEFAULT_MIN_COVERAGE", "DEFAULT_MAX_SATURATED",
           "DEFAULT_STEP_THRESHOLD",
           "FlatFieldStore", "FlatFieldProfile", "FlatFieldMatch", "flatfield_mismatch",
           "FLATFIELD_FORMAT_VERSION"]
__requires__ = ("radiometric",)
__widgets__ = ()
