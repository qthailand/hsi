"""Circle / square target check on the preview image, for stage-speed calibration.

Scan a round or square target: when the stage speed matches the frame rate the target
keeps its shape; a wrong speed stretches it along the scan axis only.  ``detect_shape``
finds the target in the RGB preview and reports how round / square it is and the
scan/slit stretch ``aspect`` (``> 1`` = too many frames over the target = stage too slow,
so multiply the speed by ``aspect``).

Data conventions:
    Input is the displayed ``(slit rows, scan columns, 3)`` uint8 RGB image; coordinates
    are image ``(x, y)`` = ``(scan column, slit row)`` with pixel centres at integers.

Algorithm:
    grey -> 5x5 Gaussian blur -> Otsu threshold, both polarities (bright target on dark
    and dark target on bright) -> 3x3 opening -> outer contours.  Contours touching the
    image border or smaller than ``MIN_AREA_PX`` are dropped; among the contours at least
    ``MIN_RELATIVE_AREA`` of the largest one, the best scoring for the requested shape wins.

    * circularity = 4*pi*A / P**2 (1 = circle, pi/4 = square), capped at 1; the perimeter
      is taken on a lightly smoothed contour so pixel staircase does not lower the score.
      It is insensitive to a small stretch (a 1.1:1 ellipse still gives 0.994), so use
      ``aspect`` to tune the speed.
    * squareness = (short / long side of the min-area rectangle) * (A / rectangle area)
      (1 = square; a circle gives pi/4).
    * aspect = sqrt(mu20 / mu02) of the region (scan width / slit height).  For any shape
      with 90-degree symmetry (circle, square at any angle) it is exactly the stretch.

Threading:
    Pure function, no shared state; call from any thread.

Example:
    >>> import numpy as np, cv2
    >>> img = np.zeros((120, 160, 3), np.uint8)
    >>> _ = cv2.circle(img, (80, 60), 40, (255, 255, 255), -1)
    >>> r = detect_shape(img, "circle")
    >>> round(r.score, 2) >= 0.97 and abs(r.aspect - 1) < 0.02
    True
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SHAPES = ("circle", "square")
MIN_AREA_PX = 50
MIN_RELATIVE_AREA = 0.2
_SMOOTH_SIGMA_PX = 1.5


@dataclass(frozen=True)
class ShapeResult:
    """One detected target.

    Attributes:
        kind: ``"circle"`` or ``"square"`` (what was asked for).
        score: circularity (circle) or squareness (square), 1.0 = perfect.
        circularity: 4*pi*A / P**2 of the target.
        squareness: (short / long side) * fill of the min-area rectangle.
        aspect: scan width / slit height from second moments (1.0 = no stretch).
        width, height: axis-aligned bounding box in pixels (scan, slit).
        center: ``(x, y)`` centroid in image pixels.
        area: region area in pixels.
        outline: ``(N, 2)`` float polygon of the fitted ellipse (circle) or
            min-area rectangle (square), image coordinates.
    """

    kind: str
    score: float
    circularity: float
    squareness: float
    aspect: float
    width: int
    height: int
    center: tuple[float, float]
    area: float
    outline: np.ndarray


def _smoothed_perimeter(points: np.ndarray) -> float:
    n = len(points)
    if n < 8:
        return float(cv2.arcLength(points.reshape(-1, 1, 2).astype(np.float32), True))
    radius = int(np.ceil(3 * _SMOOTH_SIGMA_PX))
    taps = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (taps / _SMOOTH_SIGMA_PX) ** 2)
    kernel /= kernel.sum()
    pts = points.astype(np.float64)
    padded = np.concatenate([pts[-radius:], pts, pts[:radius]])
    smooth = np.column_stack([np.convolve(padded[:, i], kernel, mode="valid") for i in range(2)])
    return float(np.sum(np.hypot(*(np.roll(smooth, -1, axis=0) - smooth).T)))


def _measure(contour: np.ndarray, kind: str) -> ShapeResult | None:
    moments = cv2.moments(contour)
    area = moments["m00"]
    if area <= 0 or moments["mu02"] <= 0:
        return None
    points = contour.reshape(-1, 2)
    perimeter = _smoothed_perimeter(points)
    circularity = min(1.0, 4.0 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
    rect = cv2.minAreaRect(contour)
    rw, rh = rect[1]
    squareness = (min(rw, rh) / max(rw, rh)) * min(1.0, area / (rw * rh)) if rw > 0 and rh > 0 else 0.0
    if kind == "circle" and len(points) >= 5:
        (ex, ey), (ea, eb), angle = cv2.fitEllipse(contour)
        t = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        px, py = ea / 2 * np.cos(t), eb / 2 * np.sin(t)
        outline = np.column_stack([ex + px * c - py * s, ey + px * s + py * c])
    else:
        outline = cv2.boxPoints(rect)
    _, _, bw, bh = cv2.boundingRect(contour)
    return ShapeResult(
        kind=kind,
        score=float(circularity if kind == "circle" else squareness),
        circularity=float(circularity),
        squareness=float(squareness),
        aspect=float(np.sqrt(moments["mu20"] / moments["mu02"])),
        width=int(bw),
        height=int(bh),
        center=(moments["m10"] / area, moments["m01"] / area),
        area=float(area),
        outline=np.asarray(outline, dtype=np.float64).reshape(-1, 2),
    )


def _candidate_contours(gray: np.ndarray) -> list[np.ndarray]:
    h, w = gray.shape
    _, bright = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    found = []
    for mask in (bright, cv2.bitwise_not(bright)):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        for contour, (_, _, _, parent) in zip(contours, hierarchy[0]):
            if parent != -1:                 # hole boundary, not an object
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
                continue
            if cv2.contourArea(contour) < MIN_AREA_PX:
                continue
            found.append(contour)
    return found


def detect_shape(rgb8: np.ndarray, kind: str) -> ShapeResult | None:
    """Find the most circle-like or square-like target in an RGB preview image.

    Args:
        rgb8: ``(H, W, 3)`` uint8 RGB image (or ``(H, W)`` grey).
        kind: ``"circle"`` or ``"square"``.

    Returns:
        The best :class:`ShapeResult`, or ``None`` when no target is found.

    Raises:
        ValueError: ``kind`` is not one of ``SHAPES``.
    """
    if kind not in SHAPES:
        raise ValueError(f"kind must be one of {SHAPES}, got {kind!r}")
    img = np.ascontiguousarray(rgb8, dtype=np.uint8)
    if img.ndim != 2 and (img.ndim != 3 or img.shape[2] < 3):
        return None
    if img.shape[0] < 3 or img.shape[1] < 3:
        return None
    gray = img if img.ndim == 2 else cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    contours = _candidate_contours(gray)
    if not contours:
        return None
    largest = max(cv2.contourArea(c) for c in contours)
    results = [_measure(c, kind) for c in contours if cv2.contourArea(c) >= MIN_RELATIVE_AREA * largest]
    return max((r for r in results if r is not None), key=lambda r: r.score, default=None)
