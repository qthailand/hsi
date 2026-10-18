"""Lamp line lists from text files, and fully automatic (anchor-free) wavelength calibration.

:func:`load_line_list` reads a reference list such as ``OceanOptics_AR-2_wavelengths_nm.txt``
(one wavelength in nm per line). :func:`search_calibration` then identifies the detected peaks
of a lamp spectrum with lines of that list **without any operator-supplied wavelengths**: every
pair of strong peaks combined with every pair of catalog lines proposes a dispersion
``wavelength = m * pixel + c``; the proposal that explains the most peaks (ties: smallest
residual) wins and is refined by re-matching and re-fitting until the assignment is stable.

Data conventions:
    Pixels are spectral-column indices at the image's hardware bin (sub-pixel), wavelengths are
    nm, ``m`` is nm/pixel (> 0: wavelength increases with the column). The match tolerance is
    given in pixels so it scales with the (unknown) dispersion.

Threading:
    Pure functions; call from any thread. A search over 25 peaks x 60 lines takes well under a
    second.

Example:
    >>> import numpy as np
    >>> lines = [870.0, 912.3, 965.8, 1047.0, 1107.9, 1211.2, 1295.7, 1350.4]
    >>> x = np.arange(640.0)
    >>> spectrum = 5 + sum(500 * np.exp(-0.5 * ((x - (l - 850) / 1.3) / 1.5) ** 2) for l in lines)
    >>> result = search_calibration(spectrum, lines)
    >>> round(result.calibration.m, 3), round(result.calibration.c, 1), len(result.pixels)
    (1.3, 850.0, 8)
"""
from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from .core import AutoCalibrationResult, detect_peaks, fit_wavelength, refine_peaks

_NUMBER = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def load_line_list(path) -> Tuple[float, ...]:
    """Read reference wavelengths (nm) from a text file.

    Format: one line per wavelength; the first number on a line is used (so ``value`` alone,
    ``value<TAB>intensity`` or ``value, name`` all work). Blank lines and lines starting with
    ``#`` or ``;`` are ignored, as are lines without a number (e.g. a header).

    Args:
        path: File path.

    Returns:
        Sorted, de-duplicated wavelengths in nm.

    Raises:
        ValueError: fewer than two wavelengths were found, or a value is not positive.
    """
    values = []
    for raw in Path(path).read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line[0] in "#;":
            continue
        match = _NUMBER.search(line)
        if match:
            values.append(float(match.group()))
    values = sorted(set(values))
    if len(values) < 2:
        raise ValueError(f"{Path(path).name}: needs at least two wavelengths (nm), one per line")
    if values[0] <= 0:
        raise ValueError(f"{Path(path).name}: wavelengths must be positive nm values")
    return tuple(values)


def _match(peaks, lines, m, c, tolerance_px):
    """Assign each line at most one peak whose predicted wavelength is within tolerance."""
    predicted = m * peaks + c
    index = np.clip(np.searchsorted(lines, predicted), 1, len(lines) - 1)
    left, right = lines[index - 1], lines[index]
    nearest = np.where(np.abs(predicted - left) <= np.abs(right - predicted), index - 1, index)
    distance = np.abs(lines[nearest] - predicted)
    ok = distance <= tolerance_px * m
    chosen = {}
    for peak, line, d in zip(peaks[ok], nearest[ok], distance[ok]):
        if line not in chosen or d < chosen[line][1]:
            chosen[line] = (peak, d)
    return {float(peak): float(lines[line]) for line, (peak, _) in chosen.items()}


def search_calibration(spectrum, catalog: Sequence[float], *, height_fraction: float = .05, distance: int = 5,
                       tolerance_px: float = 2.0, m_range: Tuple[float, float] = (0.02, 50.0),
                       max_peaks: int = 25, camera_bin: int = 1, frame_shape=(), selected_row: int = 0,
                       max_iterations: int = 10) -> AutoCalibrationResult:
    """Find the linear calibration that matches the most peaks to ``catalog`` lines.

    Args:
        spectrum: 1-D lamp spectrum of one slit row.
        catalog: Reference wavelengths (nm), e.g. from :func:`load_line_list`.
        height_fraction: Peak detection threshold (fraction of the maximum).
        distance: Minimum peak spacing (px).
        tolerance_px: A peak matches a line if its predicted wavelength is within this many
            pixels (``tolerance_px * m`` nm) of it.
        m_range: Plausible dispersion range (nm/px); proposals outside are skipped.
        max_peaks: The strongest peaks used to form proposals (all peaks are scored).
        camera_bin, frame_shape, selected_row: Stored in the resulting calibration.
        max_iterations: Refit/rematch rounds after the best proposal.

    Returns:
        :class:`AutoCalibrationResult` with every matched point marked ``'auto'``.

    Raises:
        ValueError: fewer than 3 peaks or catalog lines, or no proposal explains 3 or more peaks
            (wrong line list for this lamp/range, or wrong peak settings).
    """
    lines = np.asarray(sorted(set(float(v) for v in catalog)), dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    peaks = refine_peaks(spectrum, detect_peaks(spectrum, height_fraction, distance))
    if len(peaks) < 3 or len(lines) < 3:
        raise ValueError("Automatic calibration needs at least 3 peaks and 3 catalog lines")
    heights = spectrum[np.clip(np.round(peaks).astype(int), 0, len(spectrum) - 1)]
    strong = np.sort(peaks[np.argsort(heights)[::-1][:max_peaks]])
    line_pairs = np.array(list(combinations(range(len(lines)), 2)))
    dl = lines[line_pairs[:, 1]] - lines[line_pairs[:, 0]]

    best = (0, np.inf, None)          # (matches, mean squared distance in px, (m, c))
    for i, j in combinations(range(len(strong)), 2):
        dx = strong[j] - strong[i]
        m = dl / dx
        keep = (m >= m_range[0]) & (m <= m_range[1])
        if not keep.any():
            continue
        m = m[keep]
        c = lines[line_pairs[keep, 0]] - m * strong[i]
        predicted = m[:, None] * peaks[None, :] + c[:, None]                      # proposals x peaks
        index = np.clip(np.searchsorted(lines, predicted), 1, len(lines) - 1)
        gap = np.minimum(np.abs(predicted - lines[index - 1]), np.abs(lines[index] - predicted)) / m[:, None]
        within = gap <= tolerance_px
        count = within.sum(axis=1)
        score = np.where(within, gap ** 2, 0).sum(axis=1) / np.maximum(count, 1)
        k = int(np.lexsort((score, -count))[0])
        if count[k] > best[0] or (count[k] == best[0] and score[k] < best[1]):
            best = (int(count[k]), float(score[k]), (float(m[k]), float(c[k])))

    if best[2] is None or best[0] < 3:
        raise ValueError("No consistent match between the peaks and the line list: check that the list "
                         "belongs to this lamp and spectral range, or adjust the peak settings")
    m, c = best[2]
    matched = _match(peaks, lines, m, c, tolerance_px)
    for _ in range(max_iterations):
        if len(matched) < 2:
            break
        x = np.array(sorted(matched))
        m, c = np.polyfit(x, [matched[v] for v in x], 1)
        refreshed = _match(peaks, lines, m, c, tolerance_px)
        if refreshed == matched:
            break
        matched = refreshed
    # Reject chance coincidences (a stray peak that happens to fall near a listed line):
    # drop the worst point while it is far off the fit compared with the others.
    while len(matched) > 3:
        x = np.array(sorted(matched))
        y = np.array([matched[v] for v in x])
        m, c = np.polyfit(x, y, 1)
        residual_px = np.abs(y - (m * x + c)) / m
        worst = int(np.argmax(residual_px))
        typical = float(np.median(residual_px))
        if residual_px[worst] <= max(0.5, 4.0 * typical):
            break
        del matched[float(x[worst])]
    if len(matched) < 3:
        raise ValueError("Fewer than 3 peaks could be identified with the line list")
    x = sorted(matched)
    y = [matched[v] for v in x]
    calibration = fit_wavelength(x, y, camera_bin, frame_shape, selected_row)
    unmatched = tuple(float(p) for p in peaks if float(p) not in matched)
    return AutoCalibrationResult(tuple(x), tuple(y), ("auto",) * len(x), unmatched, calibration)
