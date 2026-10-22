"""Spectrum records of the cube viewer: names, colours, geometry, state and statistics.

One :class:`SpectrumRecord` per picked point or drawn area; a :class:`SpectrumSet` owns
them in creation order and enforces the naming and colour rules.  numpy + stdlib only
(no Qt), so it is testable without a display.

Rules:
    * ``rid`` is monotonic and never reused, not even after :meth:`SpectrumSet.clear`,
      so a late worker result for a deleted record can always be recognised and ignored.
    * Colour = ``palette[creation_count % len(palette)]``, assigned once at creation.
      Deleting a record never recolours the others.  ``creation_count`` restarts at 0
      after :meth:`SpectrumSet.clear` - with ``TAB20_HEX`` as the palette this is exactly
      today's point look (first dot after 'Clear All Points' is ``TAB20_HEX[0]``).
      :meth:`SpectrumSet.discard` drops a record that should never have existed (an area
      found empty after the fact); if it is still the newest one, its colour and default
      name are handed to the next record.
    * Names are unique (case-sensitive), stripped, non-empty and contain no control or
      line-separator characters (tab/newline would break the clipboard TSV; any other
      text - Thai, commas, quotes - is fine).  Defaults: a point is
      ``'(ix,iy)'`` (``'(ix,iy)-2'``, ``'-3'``... when that name exists); an area is
      ``'<source_name>-mean-<k>'`` with ``k`` counting areas from 0 (Spectronon style;
      ``k`` is bumped past a name the user already took).
    * ``class_name`` defaults to the name and follows renames while it still equals the
      name; it is stored from day 1 so later phases (classes) need no format change.
    * ``state``: ``'pending'`` until :meth:`SpectrumSet.set_stats`; then ``'ready'``, or
      ``'failed'`` when there are no stats or no band has a valid pixel.

Data conventions:
    Geometry is in image-corner coordinates: pixel ``(c, r)`` spans ``[c, c+1) x [r, r+1)``;
    x = column = scan, y = row = slit; ``cube[r, c, :]`` (cube ``(slit, scan,
    wavelength)``) is its spectrum.  ``stats`` is a read-only
    :class:`~modules.cube_viewer.roi_stats.RoiStats` (float64 per band, the cube's units,
    std ddof=0) owned by the record; colours are ``'#rrggbb'`` strings.

    * point ``{'x': ix, 'y': iy}`` - ints.
    * rect ``{'c0', 'r0', 'c1', 'r1'}`` - ints, half-open, non-empty (already snapped).
    * lasso ``{'vertices': [[x, y], ...]}`` - >= 3 finite float vertices in corner
      coordinates (JSON-ready lists, not an ndarray).

Threading:
    Not thread-safe; owned by the GUI thread.  Workers only compute ``RoiStats`` and
    hand them back by ``rid``.

Example:
    >>> s = SpectrumSet(["#1f77b4", "#aec7e8"], source_name="leaf")
    >>> s.add("point", {"x": 3, "y": 4}).name
    '(3,4)'
    >>> s.add("rect", {"c0": 0, "r0": 0, "c1": 5, "r1": 5}).name
    'leaf-mean-0'
    >>> [r.color for r in s]
    ['#1f77b4', '#aec7e8']
"""
from __future__ import annotations

import datetime as _dt
import operator
import re
from dataclasses import dataclass, field

import numpy as np

from .roi_stats import RoiStats

__all__ = ["KINDS", "AREA_KINDS", "POLYGON_KINDS", "STATES", "SpectrumRecord", "SpectrumSet"]

KINDS = ("point", "rect", "lasso", "polygon")
AREA_KINDS = ("rect", "lasso", "polygon")
#: Kinds whose geometry is ``{"vertices": [[x, y], ...]}`` (freehand lasso or click-vertex polygon).
POLYGON_KINDS = ("lasso", "polygon")
STATES = ("pending", "ready", "failed")

_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
# Unicode categories Cc, Zl and Zp: every character str.splitlines() or a TSV/CSV reader
# could treat as a line or cell break (\t, \n, \r, \x1c-\x1e, \x85,  ,  , ...).
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f-\x9f  ]")


def _now_iso() -> str:
    """Local time as ISO 8601 with UTC offset, whole seconds."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(eq=False)
class SpectrumRecord:
    """One spectrum in the viewer: a picked point or a drawn area and its statistics.

    Mutable on purpose (the GUI toggles ``visible``/``show_std`` and the set updates
    ``name``/``state``/``stats``).  Rename only through :meth:`SpectrumSet.rename` so
    uniqueness is kept.  Compared by identity (``eq=False``) because ``stats`` holds
    arrays.

    Attributes:
        rid: Unique id, monotonic, never reused.
        name: Unique display/export name.
        class_name: Class label; defaults to ``name``.
        kind: ``'point'``, ``'rect'`` or ``'lasso'``.
        geometry: See the module docstring.
        color: ``'#rrggbb'`` (lower case), assigned once.
        visible: Shown in plot/image and included in exports.
        show_std: Draw the +-std band in the plot (exports always carry std).
        state: ``'pending'``, ``'ready'`` or ``'failed'``.
        stats: :class:`~modules.cube_viewer.roi_stats.RoiStats` once computed.
        created_at: ISO 8601 local time with offset.
    """

    rid: int
    name: str
    class_name: str
    kind: str
    geometry: dict
    color: str
    visible: bool = True
    show_std: bool = True
    show_median: bool = False
    state: str = "pending"
    stats: RoiStats | None = None
    created_at: str = field(default_factory=_now_iso)

    @property
    def is_area(self) -> bool:
        """True for rect and lasso records (they have a meaningful std)."""
        return self.kind in AREA_KINDS

    @property
    def is_ready(self) -> bool:
        """True when the statistics are available (``state == 'ready'``)."""
        return self.state == "ready" and self.stats is not None


def _check_color(color) -> str:
    """Validate ``'#rrggbb'`` and return it in lower case."""
    text = str(color)
    if not _HEX_RE.match(text):
        raise ValueError(f"colour must be '#rrggbb', got {color!r}")
    return text.lower()


def _clean_name(name) -> str:
    """Strip ``name`` and reject empty names and control characters."""
    if not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    text = name.strip()
    if not text:
        raise ValueError("name must not be empty")
    if _CONTROL_RE.search(text):
        raise ValueError("name must not contain tabs, line breaks or other control characters")
    return text


def _pixel_index(value, key: str) -> int:
    """Non-negative integer pixel index (accepts numpy ints and integral floats)."""
    try:
        index = operator.index(value)
    except TypeError:
        number = float(value)
        if not number.is_integer():
            raise ValueError(f"geometry {key!r} must be an integer, got {value!r}") from None
        index = int(number)
    if index < 0:
        raise ValueError(f"geometry {key!r} must be >= 0, got {index}")
    return index


def _normalize_geometry(kind: str, geometry) -> dict:
    """Copy ``geometry`` and normalise its required keys for ``kind`` (extra keys kept)."""
    if not isinstance(geometry, dict):
        raise TypeError(f"geometry must be a dict, got {type(geometry).__name__}")
    geom = dict(geometry)
    try:
        if kind == "point":
            geom["x"] = _pixel_index(geometry["x"], "x")
            geom["y"] = _pixel_index(geometry["y"], "y")
        elif kind == "rect":
            for key in ("c0", "r0", "c1", "r1"):
                geom[key] = _pixel_index(geometry[key], key)
            if geom["c1"] <= geom["c0"] or geom["r1"] <= geom["r0"]:
                raise ValueError(f"rect is empty: {geom}")
        else:
            vertices = np.asarray(geometry["vertices"], dtype=np.float64)
            if vertices.ndim != 2 or vertices.shape[1] != 2 or vertices.shape[0] < 3:
                raise ValueError(f"lasso needs >= 3 (x, y) vertices, got shape {vertices.shape}")
            if not np.isfinite(vertices).all():
                raise ValueError("lasso vertices must be finite")
            geom["vertices"] = vertices.tolist()
    except KeyError as exc:
        raise ValueError(f"{kind} geometry is missing {exc.args[0]!r}") from None
    return geom


class SpectrumSet:
    """Ordered, uniquely named collection of :class:`SpectrumRecord` (creation order).

    Args:
        palette: Non-empty sequence of ``'#rrggbb'`` colours, used in order by
            creation count (the viewer passes ``TAB20_HEX``).
        source_name: Prefix of default area names, normally the cube file stem;
            blank -> ``'cube'``.

    Raises:
        ValueError: Empty palette or a colour that is not ``'#rrggbb'``.
    """

    def __init__(self, palette, source_name: str = "cube"):
        colours = tuple(_check_color(c) for c in palette)
        if not colours:
            raise ValueError("palette must contain at least one colour")
        self._palette = colours
        self._source_name = "cube"
        self.source_name = source_name
        self._records: dict[int, SpectrumRecord] = {}
        self._next_rid = 1
        self._n_created = 0      # colour index; restarts at 0 on clear()
        self._n_areas = 0        # k of the next default area name; restarts on clear()
        self._undo = None        # (rid, n_created, n_areas) before the newest add(), for discard()

    # ── properties ──────────────────────────────────────────

    @property
    def palette(self) -> tuple:
        """The colour palette (tuple of lower-case ``'#rrggbb'``)."""
        return self._palette

    @property
    def source_name(self) -> str:
        """Prefix of default area names (``'<source_name>-mean-<k>'``)."""
        return self._source_name

    @source_name.setter
    def source_name(self, value) -> None:
        """Set the prefix; control characters are dropped, blank or ``None`` -> ``'cube'``.

        Takes effect for areas added afterwards; existing names are not changed.
        """
        text = _CONTROL_RE.sub("", "" if value is None else str(value)).strip()
        self._source_name = text or "cube"

    # ── naming ──────────────────────────────────────────────

    def _names(self) -> set:
        """Names currently in use."""
        return {r.name for r in self._records.values()}

    def _area_default(self, names: set) -> tuple:
        """``('<source>-mean-<k>', k)``: ``k`` = areas created so far, bumped past taken names."""
        k = self._n_areas
        while f"{self._source_name}-mean-{k}" in names:
            k += 1
        return f"{self._source_name}-mean-{k}", k

    @staticmethod
    def _point_default(geom: dict, names: set) -> str:
        """``'(ix,iy)'``, or ``'(ix,iy)-2'``, ``'-3'``... when that name is in use."""
        base = f"({geom['x']},{geom['y']})"
        if base not in names:
            return base
        n = 2
        while f"{base}-{n}" in names:
            n += 1
        return f"{base}-{n}"

    def default_name(self, kind: str, geometry=None) -> str:
        """Name :meth:`add` would give a new ``kind`` record now (nothing is changed).

        Args:
            kind: ``'point'``, ``'rect'`` or ``'lasso'``.
            geometry: Needed for a point (its name is its pixel); ignored for areas.

        Raises:
            ValueError: Unknown ``kind``, or a point without valid geometry.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        names = self._names()
        if kind == "point":
            if geometry is None:
                raise ValueError("a point's default name needs its geometry {'x', 'y'}")
            return self._point_default(_normalize_geometry(kind, geometry), names)
        return self._area_default(names)[0]

    # ── mutation ────────────────────────────────────────────

    def add(self, kind: str, geometry, *, name=None, color=None) -> SpectrumRecord:
        """Create a ``'pending'`` record and append it.

        Args:
            kind: ``'point'``, ``'rect'`` or ``'lasso'``.
            geometry: See the module docstring; copied and normalised.
            name: Explicit name (stripped, must be unique); default per the naming rule.
            color: Explicit ``'#rrggbb'``; default ``palette[creation_count % len]``.
                The creation count advances either way.

        Returns:
            SpectrumRecord: The new record (``state='pending'``, ``stats=None``).

        Raises:
            ValueError: Unknown kind, invalid geometry, bad colour, or an empty /
                duplicate / control-character name.
        """
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        geom = _normalize_geometry(kind, geometry)
        names = self._names()
        area_k = None
        if name is None:
            if kind == "point":
                name = self._point_default(geom, names)
            else:
                name, area_k = self._area_default(names)
        else:
            name = _clean_name(name)
            if name in names:
                raise ValueError(f"name {name!r} is already used")
        colour = (self._palette[self._n_created % len(self._palette)]
                  if color is None else _check_color(color))
        record = SpectrumRecord(rid=self._next_rid, name=name, class_name=name,
                                kind=kind, geometry=geom, color=colour)
        self._records[record.rid] = record
        self._undo = (record.rid, self._n_created, self._n_areas)
        self._next_rid += 1
        self._n_created += 1
        if kind in AREA_KINDS:
            self._n_areas = (area_k if area_k is not None else self._n_areas) + 1
        return record

    def remove(self, rid: int):
        """Remove a record; returns it, or ``None`` when ``rid`` is unknown (idempotent)."""
        return self._records.pop(rid, None)

    def discard(self, rid: int):
        """Remove a record as if it had never been added (an area found empty afterwards).

        When it is still the newest record - nothing was added after it - the colour and
        default-area-name counters step back, so the next record gets the colour and name
        it would have had without the discarded one (as for an area discarded before it
        was added).  Otherwise this is :meth:`remove`: other records never change colour.
        The ``rid`` is never reused.

        Returns:
            SpectrumRecord | None: The removed record, or ``None`` for an unknown ``rid``.
        """
        record = self._records.pop(rid, None)
        if record is not None and self._undo is not None and self._undo[0] == rid:
            self._n_created, self._n_areas = self._undo[1], self._undo[2]
            self._undo = None
        return record

    def rename(self, rid: int, new_name: str) -> SpectrumRecord:
        """Rename a record (``class_name`` follows while it still equals the old name).

        Renaming to the record's own current name is a no-op.

        Returns:
            SpectrumRecord: The renamed record.

        Raises:
            KeyError: Unknown ``rid``.
            ValueError: Empty name, control characters, or a name another record
                already uses (case-sensitive).
        """
        record = self._records[rid]
        name = _clean_name(new_name)
        if name == record.name:
            return record
        if any(r.name == name for r in self._records.values() if r.rid != rid):
            raise ValueError(f"name {name!r} is already used")
        if record.class_name == record.name:
            record.class_name = name
        record.name = name
        return record

    def set_class(self, rid: int, class_name: str) -> SpectrumRecord:
        """Set a record's ``class_name`` (stripped; the same rules as a name, but classes
        may be shared by many records).  Once it differs from the name it no longer
        follows renames; setting it back to the name makes it follow again.

        Returns:
            SpectrumRecord: The record.

        Raises:
            KeyError: Unknown ``rid``.
            TypeError, ValueError: Not a str, empty, or control characters.
        """
        record = self._records[rid]
        record.class_name = _clean_name(class_name)
        return record

    def set_color(self, rid: int, color: str) -> SpectrumRecord:
        """Give one record a new colour (the 'Set colour...' action); no other record changes.

        Returns:
            SpectrumRecord: The record, colour stored as lower-case ``'#rrggbb'``.

        Raises:
            KeyError: Unknown ``rid``.
            ValueError: ``color`` is not ``'#rrggbb'``.
        """
        record = self._records[rid]
        record.color = _check_color(color)
        return record

    def set_stats(self, rid: int, stats):
        """Attach computed statistics and settle the state.

        ``'ready'`` when ``stats`` has at least one band with a valid pixel, otherwise
        ``'failed'`` (``stats=None`` marks a failed or cancelled job).

        Returns:
            SpectrumRecord | None: The record, or ``None`` for an unknown ``rid`` (a late
            result of a deleted record - ignore it).
        """
        record = self._records.get(rid)
        if record is None:
            return None
        record.stats = stats
        ok = stats is not None and bool(np.any(np.asarray(stats.n_valid) > 0))
        record.state = "ready" if ok else "failed"
        return record

    def clear(self) -> None:
        """Remove every record; colour and area-name counters restart, rids do not."""
        self._records.clear()
        self._n_created = 0
        self._n_areas = 0
        self._undo = None

    # ── queries ─────────────────────────────────────────────

    def get(self, rid: int, default=None):
        """Record with id ``rid``, or ``default`` when there is none."""
        return self._records.get(rid, default)

    def find(self, name: str):
        """Record whose name equals ``name`` exactly, or ``None``."""
        for record in self._records.values():
            if record.name == name:
                return record
        return None

    def records(self, visible_only: bool = False) -> list:
        """Records in creation order (only visible ones when ``visible_only``)."""
        return [r for r in self._records.values() if r.visible or not visible_only]

    def by_class(self) -> dict:
        """``{class_name: [records]}`` in creation order of first appearance."""
        groups: dict = {}
        for record in self._records.values():
            groups.setdefault(record.class_name, []).append(record)
        return groups

    def __iter__(self):
        """Iterate over a snapshot of the records in creation order."""
        return iter(tuple(self._records.values()))

    def __len__(self) -> int:
        """Number of records."""
        return len(self._records)

    def __contains__(self, rid) -> bool:
        """True when a record with id ``rid`` exists."""
        return rid in self._records
