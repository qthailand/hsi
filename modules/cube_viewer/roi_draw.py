"""ROI drawing tools and record overlays for the cube viewer's two image views.

``RoiDrawMixin`` adds three tools to a ``ZoomPanMixin`` image widget: ``CubeImageCanvas``
(single band) and ``_RoiPixmapLabel`` (the RGB view, a subclass of cube_preview's
``CenteredPixmapLabel``, which itself is left unchanged so the Scan window is unaffected).

Data conventions:
    Float image-corner coordinates as in ``roi_stats``: pixel ``(c, r)`` covers
    ``[c, c + 1) x [r, r + 1)``; x = column = scan, y = row = slit of the
    ``(slit, scan)`` image shown.  Geometries are emitted as JSON-ready ints / floats;
    the overlay records are held by reference and only read.

Tools (``set_tool(name)``):
    * ``'point'`` (default) - today's behaviour: a left click emits the widget's own
      ``pixel_clicked`` / ``clicked(ix, iy)``; the mixin does nothing.
    * ``'rect'`` - press, drag, release.  While dragging, the raw rubber band is drawn
      dashed and the rectangle snapped by :func:`roi_stats.snap_rect` (closed pixel-centre
      rule) solid, on the pixel edges, with its size: what you see is what is averaged.
      Release emits ``roi_drawn('rect', {'c0', 'r0', 'c1', 'r1'})`` (ints, half-open).
    * ``'lasso'`` - freehand.  A vertex is added once the pointer is at least 2 widget px
      and 1 image px away from the previous vertex, up to 5000 vertices; release closes
      the shape and emits ``roi_drawn('lasso', {'vertices': [[x, y], ...]})`` (floats;
      the closing edge is implicit, the first vertex is not repeated).

    A drawing that cannot hold a pixel is discarded with ``roi_discarded(reason)``: a
    rectangle that encloses no pixel centre, a lasso with fewer than 3 vertices or whose
    bounding box encloses no pixel centre.  (A thin lasso can still contain no centre;
    :func:`roi_selection` then reports 0 px - the caller decides.)  Esc cancels a drawing
    in progress (also ``roi_discarded('Drawing cancelled')``).  Points are not reported
    through ``roi_drawn``; they keep the widgets' click signals.

    Wheel zoom and middle-button pan keep working while drawing, because the drawing is
    stored in float image-corner coordinates (``_zp_widget_to_image_f``, built on
    ``ZoomPanMixin._zp_transform``; the mixin's own int mapping truncates and refuses
    points outside the image).  A double click still zooms to fit: a lasso closes on
    release, never on a double click.

Overlays:
    ``set_roi_overlays(records, selected_rid)`` draws spectrum records - any objects with
    ``rid``, ``name``, ``kind``, ``geometry``, ``color`` (``'#rrggbb'``), ``visible`` and
    ``state``, normally :class:`~modules.cube_viewer.spectra.SpectrumRecord`.  Areas are
    outlined with a 1 px line in the record colour and a 1 px dark drop shadow (dashed
    while ``'pending'``, dotted when ``'failed'``; the selected record gets a 2.5 px line on
    a dark halo) and a name tag; points are today's dot (record colour, white ring, at the
    pixel's corner as before) with the name as tag, placed by each view's own legacy rule
    (``_roi_point_tag_inside``): the RGB view flips the tag to the left of the dot at the
    right edge and clamps it to the top edge, as ``CenteredPixmapLabel`` always did; the
    single-band canvas does neither, as its old click dots did.  Hidden records are not drawn.
    Outline paths are cached per record in image coordinates (geometry dicts are treated
    as immutable: replace the dict to change it) and drawn through the zoom transform with
    cosmetic pens; the rendered layer is cached as a pixmap and reused until the size,
    zoom / pan, selection or a shown record attribute changes.
    ``set_class_overlay(pixmap_or_None, opacity)`` is the phase-2 hook for a class map: an
    image-sized ``QPixmap`` / ``QImage`` painted over the base image (nearest neighbour,
    exactly where the base pixmap is drawn) and under the outlines.  Nothing produces one
    yet.  In the RGB view the base class paints the legacy ``add_click_point`` dots before
    the overlay, so the record overlays should be used for points there.

Threading:
    GUI thread only.  :func:`roi_selection` is pure numpy and safe in a worker thread.

Measured cost:
    See ``README.md`` (overlay painting with 50 records and a 5000-vertex lasso).

Example:
    >>> from modules.cube_viewer.canvas import CubeImageCanvas        # doctest: +SKIP
    >>> canvas = CubeImageCanvas()                                     # doctest: +SKIP
    >>> canvas.set_tool("rect")                                        # doctest: +SKIP
    >>> canvas.roi_drawn.connect(lambda kind, geom: print(kind, geom))  # doctest: +SKIP
"""

from __future__ import annotations

import math

import numpy as np
from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF, QTransform

from modules.cube_preview.widget import SHAPE_OVERLAY_COLOR, CenteredPixmapLabel
from modules.cube_viewer.roi_stats import polygon_mask, snap_rect

__all__ = ("TOOLS", "LASSO_MAX_VERTICES", "LASSO_MIN_WIDGET_PX", "LASSO_MIN_IMAGE_PX",
           "RoiDrawMixin", "roi_selection")

TOOLS = ("point", "rect", "lasso", "polygon")
#: A polygon closes when a click lands within this many widget px of its first vertex.
POLYGON_CLOSE_PX = 8.0
LASSO_MAX_VERTICES = 5000
LASSO_MIN_WIDGET_PX = 2.0
LASSO_MIN_IMAGE_PX = 1.0

_DRAW_COLOR = QColor(SHAPE_OVERLAY_COLOR)      # rubber band: the Scan window's shape-outline yellow
_RUBBER_COLOR = QColor(255, 255, 255, 200)
_HALO_COLOR = QColor(0, 0, 0, 170)
_TAG_BG = QColor(0, 0, 0, 160)
_TAG_TEXT = QColor(255, 255, 255, 230)
_DOT_RING = QColor(255, 255, 255, 220)
# Outlines are 1 px lines with a 1 px drop shadow: Qt draws 1 px lines on a fast path
# (a self-overlapping 5000-vertex lasso: 0.4 ms), while wide strokes go through the
# stroker (60 ms for the same lasso).  Only the selected record gets the wide outline.
_PEN_WIDTH = 1.0
_PEN_WIDTH_SELECTED = 2.5
_SHADOW_SHIFT = QTransform.fromTranslate(1.0, 1.0)
_RUBBER_WIDTH = 1.5        # the snapped box of a rectangle being dragged
_TAG_MAX_PX = 160
_STATE_STYLE = {"pending": Qt.DashLine, "failed": Qt.DotLine}


def roi_selection(kind, geometry, height, width, *, cancel=None):
    """Pixels of a record geometry, ready for ``roi_stats(cube, bbox, local_mask)``.

    Args:
        kind: ``'point'``, ``'rect'`` or ``'lasso'``.
        geometry: The record geometry (see :mod:`modules.cube_viewer.spectra`).
        height, width: Image size in pixels.
        cancel: Optional callable handed to :func:`roi_stats.polygon_mask` for a lasso
            (polled between batches of about 10 ms); True stops it with ``RoiCancelled``.

    Returns:
        ``(bbox, local_mask, n_px)``: ``bbox = (c0, r0, c1, r1)`` half-open and clipped to
        the image; ``local_mask`` is None for points and rectangles, else the lasso's bool
        mask of shape ``(r1 - r0, c1 - c0)`` (:func:`roi_stats.polygon_mask`); ``n_px`` is
        the pixel count (0 when nothing of the shape is inside the image).

    Raises:
        ValueError: Unknown ``kind`` or bad lasso vertices.
        RoiCancelled: ``cancel()`` returned True while a lasso mask was rasterised.
    """
    h, w = int(height), int(width)
    if kind == "point":
        x, y = int(geometry["x"]), int(geometry["y"])
        if 0 <= x < w and 0 <= y < h:
            return (x, y, x + 1, y + 1), None, 1
        return (0, 0, 0, 0), None, 0
    if kind == "rect":
        c0, c1 = (min(max(int(geometry[k]), 0), w) for k in ("c0", "c1"))
        r0, r1 = (min(max(int(geometry[k]), 0), h) for k in ("r0", "r1"))
        return (c0, r0, c1, r1), None, max(0, c1 - c0) * max(0, r1 - r0)
    if kind in ("lasso", "polygon"):
        vertices = np.asarray(geometry["vertices"], dtype=np.float64).reshape(-1, 2)
        c0, r0, mask = polygon_mask(vertices, h, w, cancel=cancel)
        bbox = (c0, r0, c0 + mask.shape[1], r0 + mask.shape[0])
        return bbox, mask, int(np.count_nonzero(mask))
    raise ValueError(f"kind must be one of {TOOLS}, got {kind!r}")


def _cosmetic(color, width, style=Qt.SolidLine) -> QPen:
    """A pen whose width is in device pixels whatever the painter transform."""
    pen = QPen(QColor(color), width, style)
    pen.setCosmetic(True)
    return pen


def _area_path(kind, geometry) -> QPainterPath:
    """Outline of a rect / lasso geometry in image-corner coordinates (empty otherwise)."""
    path = QPainterPath()
    if kind == "rect":
        c0, r0, c1, r1 = (float(geometry[k]) for k in ("c0", "r0", "c1", "r1"))
        path.addRect(QRectF(c0, r0, c1 - c0, r1 - r0))
    elif kind in ("lasso", "polygon"):
        vertices = geometry.get("vertices") or ()
        if len(vertices) >= 2:
            path.addPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y in vertices]))
            path.closeSubpath()
    return path


class _Drag:
    """A drawing in progress, in image-corner coordinates."""

    __slots__ = ("kind", "start", "cur", "vertices", "poly", "image_size", "capped")

    def __init__(self, kind, x, y, image_size):
        self.kind = kind
        self.start = (x, y)
        self.cur = (x, y)
        self.vertices = [(x, y)]
        self.poly = QPolygonF([QPointF(x, y)])
        self.image_size = image_size
        self.capped = False


class RoiDrawMixin:
    """Rect / lasso drawing and record overlays for a ``ZoomPanMixin`` image widget.

    The host is a ``QWidget`` that also inherits ``ZoomPanMixin``, calls
    ``_init_roi_draw()`` in ``__init__`` and routes its events:

    * ``mousePressEvent``: after ``_zp_mouse_press``, return if ``_roi_mouse_press(event)``.
    * ``mouseMoveEvent``: unless panning, return if ``_roi_mouse_move(event)``.
    * ``mouseReleaseEvent``: return if ``_roi_mouse_release(event)``; after a pan ends
      call ``_roi_restore_cursor()``.
    * ``keyPressEvent``: return if ``_roi_key_press(event)``.
    * ``paintEvent``: after the base image ``_roi_paint_class_overlay(painter)``, then
      ``_roi_paint_records(painter)`` and ``_roi_paint_drawing(painter)``.

    See the module docstring for the tools, the overlays and the coordinates.

    Signals:
        roi_drawn(str, object): ``(kind, geometry)`` of a finished rectangle or lasso.
        roi_discarded(str): A drawing was dropped (empty, too few points, Esc); the text
            is meant for a status bar.
    """

    roi_drawn = pyqtSignal(str, object)
    roi_discarded = pyqtSignal(str)

    # Point tags: True keeps the tag inside the widget (flip left at the right edge, clamp
    # at the top) - the RGB view's legacy rule; the single-band canvas sets False (its
    # legacy tags were never moved).
    _roi_point_tag_inside = True

    def _init_roi_draw(self):
        """Set up the tool state; call in ``__init__`` after ``_init_zoom_pan()``."""
        self._roi_tool = "point"
        self._roi_drag: _Drag | None = None
        self._roi_records: tuple = ()
        self._roi_selected = None
        self._roi_paths: dict = {}
        self._roi_layer: QPixmap | None = None    # rendered records, reused while the key holds
        self._roi_layer_key = None
        self._roi_class_pixmap: QPixmap | None = None
        self._roi_class_opacity = 0.5

    # ── tools ─────────────────────────────────────────────

    @property
    def tool(self) -> str:
        """str: The active tool, ``'point'``, ``'rect'`` or ``'lasso'``."""
        return self._roi_tool

    def set_tool(self, name: str) -> None:
        """Select the drawing tool; a drawing in progress is cancelled.

        Args:
            name: ``'point'`` (click = pixel spectrum, today's behaviour), ``'rect'`` or
                ``'lasso'``.

        Raises:
            ValueError: Unknown tool name.
        """
        if name not in TOOLS:
            raise ValueError(f"tool must be one of {TOOLS}, got {name!r}")
        if name == self._roi_tool:
            return
        self.cancel_drawing()
        self._roi_tool = name
        if name == "point":
            self.unsetCursor()
        else:
            self._roi_restore_cursor()

    @property
    def is_drawing(self) -> bool:
        """bool: True while a rectangle or lasso is being dragged."""
        return self._roi_drag is not None

    def cancel_drawing(self) -> bool:
        """Drop the drawing in progress without emitting anything.

        Returns:
            bool: True if a drawing was cancelled.
        """
        if self._roi_drag is None:
            return False
        self._roi_drag = None
        self.update()
        return True

    def drawing_preview(self):
        """The drawing in progress in geometry form, or None.

        Returns:
            dict | None: ``{'kind': 'rect', 'c0', 'r0', 'c1', 'r1'}`` - the snapped box
            being shown (may be empty) - or ``{'kind': 'lasso', 'vertices': [...]}``.
        """
        d = self._roi_drag
        if d is None:
            return None
        if d.kind == "rect":
            c0, r0, c1, r1 = self._roi_snapped(d)
            return {"kind": "rect", "c0": c0, "r0": r0, "c1": c1, "r1": r1}
        return {"kind": d.kind, "vertices": [[x, y] for x, y in d.vertices]}

    def _zp_widget_to_image_f(self, wx: float, wy: float) -> tuple[float, float]:
        """Map widget coordinates to float image-corner coordinates (no clipping, no truncation)."""
        ox, oy, sx, sy = self._zp_transform()
        return (wx - ox) / sx, (wy - oy) / sy

    def _roi_restore_cursor(self):
        """Show the cross cursor of the drawing tools (after a pan reset it to an arrow)."""
        if self._roi_tool != "point":
            self.setCursor(Qt.CrossCursor)

    # ── overlays ──────────────────────────────────────────

    def set_roi_overlays(self, records, selected_rid=None) -> None:
        """Set the spectrum records drawn over the image (replaces the previous list).

        The records are kept by reference: after changing one in place (``name``,
        ``color``, ``visible``, ``state``) a repaint (``update()`` or this call again)
        shows it.  A geometry dict must be replaced, not mutated.

        Args:
            records: Iterable of record objects (see the module docstring), drawn in
                this order; records with ``visible`` False are skipped.
            selected_rid: ``rid`` drawn with the thick outline (or a bigger dot), or None.
        """
        records = tuple(records)
        live = {r.rid for r in records}
        for rid in [rid for rid in self._roi_paths if rid not in live]:
            del self._roi_paths[rid]
        self._roi_records = records
        self._roi_selected = selected_rid
        self.update()

    def set_selected_roi(self, rid) -> None:
        """Change only which record is drawn as selected (``None`` for none)."""
        if rid != self._roi_selected:
            self._roi_selected = rid
            self.update()

    def roi_overlay_ids(self) -> list:
        """Ids of the records that are drawn (the visible ones), in paint order."""
        return [r.rid for r in self._roi_records if getattr(r, "visible", True)]

    def set_class_overlay(self, pixmap, opacity: float = 0.5) -> None:
        """Show (or with None remove) a class map over the image - the phase-2 hook.

        Args:
            pixmap: Image-sized ``QPixmap`` or ``QImage`` (e.g. Indexed8 with the class
                colours and alpha 0 for unclassified pixels), or None.  It is stretched
                over the image with nearest-neighbour sampling.
            opacity: 0..1 (clipped).
        """
        if isinstance(pixmap, QImage):
            pixmap = QPixmap.fromImage(pixmap)
        if pixmap is not None and pixmap.isNull():
            pixmap = None
        self._roi_class_pixmap = pixmap
        self._roi_class_opacity = min(max(float(opacity), 0.0), 1.0)
        self.update()

    def class_overlay(self):
        """The class overlay as ``(QPixmap | None, opacity)``."""
        return self._roi_class_pixmap, self._roi_class_opacity

    def _roi_path(self, record) -> QPainterPath:
        """Cached outline of an area record (rebuilt when its geometry dict is replaced)."""
        geometry = record.geometry
        hit = self._roi_paths.get(record.rid)
        if hit is not None and hit[0] is geometry and hit[1] == record.kind:
            return hit[2]
        path = _area_path(record.kind, geometry)
        self._roi_paths[record.rid] = (geometry, record.kind, path)
        return path

    # ── mouse / keyboard (called by the host's event handlers) ──

    def _roi_event_image_pos(self, event) -> tuple[float, float]:
        """Float image-corner position of a mouse event."""
        pos = event.localPos()
        return self._zp_widget_to_image_f(pos.x(), pos.y())

    def _roi_mouse_press(self, event) -> bool:
        """Start a rectangle / lasso on a left press (not in the point tool); True if consumed.

        Polygon tool: every left press adds a vertex (a press near the first vertex closes
        the shape) and a right press closes it.
        """
        d = self._roi_drag
        if d is not None and d.kind == "polygon":
            if event.button() == Qt.RightButton:
                self._roi_finish_polygon()
                return True
            if event.button() != Qt.LeftButton:
                return False
            x, y = self._roi_event_image_pos(event)
            if len(d.vertices) >= 3 and self._roi_near_first_vertex(d, event):
                self._roi_finish_polygon()
            elif (x, y) != d.vertices[-1]:
                d.vertices.append((x, y))
                d.poly.append(QPointF(x, y))
                d.cur = (x, y)
                self.update()
            return True
        if self._roi_tool == "point" or event.button() != Qt.LeftButton:
            return False
        size = self._zp_image_size()
        if size[0] <= 0 or size[1] <= 0:
            return True
        x, y = self._roi_event_image_pos(event)
        self._roi_drag = _Drag(self._roi_tool, x, y, tuple(size))
        self.update()
        return True

    def _roi_mouse_move(self, event) -> bool:
        """Follow the pointer while drawing; True if consumed."""
        d = self._roi_drag
        if d is None:
            return False
        if tuple(self._zp_image_size()) != d.image_size:
            self.cancel_drawing()
            return True
        x, y = self._roi_event_image_pos(event)
        d.cur = (x, y)
        if d.kind == "lasso":
            self._roi_lasso_add(d, x, y)
        self.update()
        return True

    def _roi_mouse_release(self, event) -> bool:
        """Finish the drawing on the left release and emit ``roi_drawn`` / ``roi_discarded``."""
        d = self._roi_drag
        if d is not None and d.kind == "polygon":
            return event.button() in (Qt.LeftButton, Qt.RightButton)   # closed by Enter / right / first vertex
        if d is None or event.button() != Qt.LeftButton:
            return False
        self._roi_drag = None
        self.update()
        if tuple(self._zp_image_size()) != d.image_size:
            return True                       # the image changed under the drawing
        x, y = self._roi_event_image_pos(event)
        d.cur = (x, y)
        if d.kind == "rect":
            c0, r0, c1, r1 = self._roi_snapped(d)
            if c1 <= c0 or r1 <= r0:
                self.roi_discarded.emit("Rectangle discarded: it encloses no pixel centre")
            else:
                self.roi_drawn.emit("rect", {"c0": c0, "r0": r0, "c1": c1, "r1": r1})
            return True
        self._roi_lasso_add(d, x, y)
        if len(d.vertices) < 3:
            self.roi_discarded.emit("Lasso discarded: hold the button down and draw around the area")
            return True
        xs = [v[0] for v in d.vertices]
        ys = [v[1] for v in d.vertices]
        w, h = d.image_size
        c0, r0, c1, r1 = snap_rect(min(xs), min(ys), max(xs), max(ys), h, w)
        if c1 <= c0 or r1 <= r0:
            self.roi_discarded.emit("Lasso discarded: it encloses no pixel centre")
            return True
        self.roi_drawn.emit("lasso", {"vertices": [[float(vx), float(vy)] for vx, vy in d.vertices]})
        return True

    def _roi_key_press(self, event) -> bool:
        """Esc cancels a drawing; for a polygon Enter closes it and Backspace removes the last
        vertex.  True if consumed."""
        d = self._roi_drag
        if d is None:
            return False
        if event.key() == Qt.Key_Escape:
            self.cancel_drawing()
            self.roi_discarded.emit("Drawing cancelled")
            return True
        if d.kind != "polygon":
            return False
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._roi_finish_polygon()
            return True
        if event.key() == Qt.Key_Backspace:
            if len(d.vertices) > 1:
                d.vertices.pop()
                d.poly.remove(d.poly.size() - 1)
                self.update()
            else:
                self.cancel_drawing()
                self.roi_discarded.emit("Drawing cancelled")
            return True
        return False

    def _roi_mouse_double_click(self, event) -> bool:
        """A double click closes a polygon being drawn (instead of zooming to fit); True if consumed."""
        d = self._roi_drag
        if d is None or d.kind != "polygon" or event.button() != Qt.LeftButton:
            return False
        self._roi_finish_polygon()
        return True

    def _roi_near_first_vertex(self, d, event) -> bool:
        """True when the event is within ``POLYGON_CLOSE_PX`` widget px of the first vertex."""
        ox, oy, sx, sy = self._zp_transform()
        fx, fy = d.vertices[0]
        pos = event.localPos()
        return math.hypot(pos.x() - (ox + fx * sx), pos.y() - (oy + fy * sy)) <= POLYGON_CLOSE_PX

    def _roi_finish_polygon(self) -> None:
        """Close the polygon being drawn: emit ``roi_drawn('polygon', ...)`` or ``roi_discarded``."""
        d = self._roi_drag
        self._roi_drag = None
        self.update()
        if d is None:
            return
        if tuple(self._zp_image_size()) != d.image_size:
            return
        if len(d.vertices) < 3:
            self.roi_discarded.emit("Polygon discarded: click at least 3 corners, then Enter")
            return
        xs = [v[0] for v in d.vertices]
        ys = [v[1] for v in d.vertices]
        w, h = d.image_size
        c0, r0, c1, r1 = snap_rect(min(xs), min(ys), max(xs), max(ys), h, w)
        if c1 <= c0 or r1 <= r0:
            self.roi_discarded.emit("Polygon discarded: it encloses no pixel centre")
            return
        self.roi_drawn.emit("polygon", {"vertices": [[float(vx), float(vy)] for vx, vy in d.vertices]})

    def _roi_lasso_add(self, d: _Drag, x: float, y: float) -> None:
        """Append a lasso vertex when it is far enough from the previous one (and below the cap)."""
        if len(d.vertices) >= LASSO_MAX_VERTICES:
            d.capped = True
            return
        lx, ly = d.vertices[-1]
        scale = self._zp_transform()[2]
        step = max(LASSO_MIN_IMAGE_PX, LASSO_MIN_WIDGET_PX / scale) if scale > 0 else LASSO_MIN_IMAGE_PX
        if math.hypot(x - lx, y - ly) < step:
            return
        d.vertices.append((x, y))
        d.poly.append(QPointF(x, y))

    def _roi_snapped(self, d: _Drag) -> tuple:
        """The rectangle drag snapped by the closed pixel-centre rule (may be empty)."""
        w, h = d.image_size
        return snap_rect(d.start[0], d.start[1], d.cur[0], d.cur[1], h, w)

    # ── painting (called by the host's paintEvent) ────────

    def _roi_transform(self) -> QTransform:
        """Image-corner -> widget transform of the current zoom / pan."""
        ox, oy, sx, sy = self._zp_transform()
        return QTransform(sx, 0.0, 0.0, sy, ox, oy)

    def _roi_paint_class_overlay(self, painter: QPainter) -> None:
        """Paint the class overlay exactly over the drawn base image (nearest neighbour)."""
        pm = self._roi_class_pixmap
        if pm is None or self._roi_class_opacity <= 0.0:
            return
        iw, ih = self._zp_image_size()
        if iw <= 0 or ih <= 0:
            return
        ox, oy, sx, sy = self._zp_transform()
        # the base pixmap is drawn at int(ox), int(oy) with size int(iw*sx) x int(ih*sy)
        target = QRectF(int(ox), int(oy), int(iw * sx), int(ih * sy))
        painter.save()
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)
        painter.setOpacity(self._roi_class_opacity)
        painter.drawPixmap(target, pm, QRectF(0.0, 0.0, pm.width(), pm.height()))
        painter.restore()

    def _roi_paint_records(self, painter: QPainter) -> None:
        """Paint the visible records from a cached layer, re-rendered when anything shown changes.

        The layer key holds the widget size, the zoom / pan, the selection and each visible
        record's rid, name, colour, kind, state and geometry identity, so band changes and
        a drawing in progress reuse the layer (one blit) while edits are picked up.
        """
        records = [r for r in self._roi_records if getattr(r, "visible", True)]
        if not records:
            return
        iw, ih = self._zp_image_size()
        if iw <= 0 or ih <= 0:
            return
        dpr = self.devicePixelRatioF()
        key = (self.width(), self.height(), dpr, self._zp_transform(), self._roi_selected,
               tuple((r.rid, r.name, r.color, r.kind, getattr(r, "state", ""), id(r.geometry))
                     for r in records))
        if key != self._roi_layer_key or self._roi_layer is None:
            layer = QPixmap(max(1, round(self.width() * dpr)), max(1, round(self.height() * dpr)))
            layer.setDevicePixelRatio(dpr)
            layer.fill(Qt.transparent)
            layer_painter = QPainter(layer)
            layer_painter.setRenderHint(QPainter.Antialiasing)
            layer_painter.setFont(self.font())
            self._roi_render_records(layer_painter, records)
            layer_painter.end()
            self._roi_layer, self._roi_layer_key = layer, key
        painter.drawPixmap(0, 0, self._roi_layer)

    def _roi_render_records(self, painter: QPainter, records) -> None:
        """Draw area outlines with name tags, then point dots, in widget coordinates."""
        selected = self._roi_selected
        areas = [r for r in records if r.kind != "point"]
        areas.sort(key=lambda r: r.rid == selected)          # the selected one on top
        points = [r for r in records if r.kind == "point"]
        transform = self._roi_transform()

        painter.save()
        font = painter.font()
        font.setPixelSize(10)
        painter.setFont(font)
        if areas:
            painter.setBrush(Qt.NoBrush)
            for r in areas:
                path = self._roi_path(r)
                if path.isEmpty():
                    continue
                style = _STATE_STYLE.get(getattr(r, "state", ""), Qt.SolidLine)
                if r.rid == selected:
                    painter.setTransform(transform)
                    painter.setPen(_cosmetic(_HALO_COLOR, _PEN_WIDTH_SELECTED + 2.0))
                else:
                    painter.setTransform(transform * _SHADOW_SHIFT)
                    painter.setPen(_cosmetic(_HALO_COLOR, _PEN_WIDTH, style))
                painter.drawPath(path)
                painter.setTransform(transform)
                width = _PEN_WIDTH_SELECTED if r.rid == selected else _PEN_WIDTH
                painter.setPen(_cosmetic(r.color, width, style))
                painter.drawPath(path)
            painter.resetTransform()
            view = QRectF(0.0, 0.0, self.width(), self.height())
            for r in areas:
                path = self._roi_path(r)
                if path.isEmpty():
                    continue
                box = transform.mapRect(path.boundingRect())
                if not box.intersects(view):
                    continue
                self._roi_draw_area_tag(painter, r.name, box, QColor(r.color))
        for r in points:
            self._roi_draw_point(painter, r, r.rid == selected)
        painter.restore()

    def _roi_draw_area_tag(self, painter: QPainter, text: str, box: QRectF, color: QColor) -> None:
        """Name tag above an outline's top-left corner, kept inside the widget."""
        metrics = painter.fontMetrics()
        text = metrics.elidedText(text, Qt.ElideRight, _TAG_MAX_PX)
        tw = metrics.horizontalAdvance(text) + 8
        th = metrics.height() + 4
        x = min(max(0.0, box.left()), max(0.0, self.width() - tw))
        y = box.top() - th - 2
        if y < 0:
            y = box.top() + 2
        y = min(max(0.0, y), max(0.0, self.height() - th))
        rect = QRectF(x, y, tw, th)
        painter.setPen(QPen(color, 1))
        painter.setBrush(QBrush(_TAG_BG))
        painter.drawRoundedRect(rect, 3, 3)
        painter.setPen(QPen(_TAG_TEXT))
        painter.drawText(rect, Qt.AlignCenter, text)

    def _roi_draw_point(self, painter: QPainter, record, selected: bool) -> None:
        """Today's click dot (record colour, white ring, at the pixel corner) with the name tag."""
        geometry = record.geometry
        cx, cy = self._zp_map_image_to_widget(geometry["x"], geometry["y"])
        if cx < -20 or cy < -20 or cx > self.width() + 20 or cy > self.height() + 20:
            return
        radius = 5 if selected else 3
        painter.setPen(QPen(_DOT_RING, 2))
        painter.setBrush(QBrush(QColor(record.color)))
        painter.drawEllipse(QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius))

        metrics = painter.fontMetrics()
        label = metrics.elidedText(record.name, Qt.ElideRight, _TAG_MAX_PX)
        tw = metrics.horizontalAdvance(label) + 8
        th = metrics.height() + 4
        lx = cx + radius + 4
        ly = cy - th // 2
        if self._roi_point_tag_inside:
            if lx + tw > self.width():
                lx = cx - radius - 4 - tw
            if ly < 0:
                ly = 0
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(_TAG_BG))
        painter.drawRoundedRect(QRectF(lx, ly, tw, th), 3, 3)
        painter.setPen(QPen(_TAG_TEXT))
        painter.drawText(QRectF(lx, ly, tw, th), Qt.AlignCenter, label)

    def _roi_paint_drawing(self, painter: QPainter) -> None:
        """Paint the drawing in progress (rubber band, snapped box, lasso path)."""
        d = self._roi_drag
        if d is None:
            return
        transform = self._roi_transform()
        painter.save()
        painter.setTransform(transform)
        painter.setBrush(Qt.NoBrush)
        tag = ""
        if d.kind == "rect":
            (x0, y0), (x1, y1) = d.start, d.cur
            painter.setPen(_cosmetic(_RUBBER_COLOR, 1.0, Qt.DashLine))
            painter.drawRect(QRectF(QPointF(min(x0, x1), min(y0, y1)), QPointF(max(x0, x1), max(y0, y1))))
            c0, r0, c1, r1 = self._roi_snapped(d)
            if c1 > c0 and r1 > r0:
                box = QRectF(c0, r0, c1 - c0, r1 - r0)
                painter.setPen(_cosmetic(_HALO_COLOR, _RUBBER_WIDTH + 2.0))
                painter.drawRect(box)
                painter.setPen(_cosmetic(_DRAW_COLOR, _RUBBER_WIDTH))
                painter.drawRect(box)
                tag = f"{c1 - c0}×{r1 - r0} = {(c1 - c0) * (r1 - r0)} px"
            else:
                tag = "0 px"
        else:
            poly = QPolygonF(d.poly)
            poly.append(QPointF(*d.cur))
            painter.setTransform(transform * _SHADOW_SHIFT)     # 1 px lines: fast for 5000 vertices
            painter.setPen(_cosmetic(_HALO_COLOR, 1.0))
            painter.drawPolyline(poly)
            painter.setTransform(transform)
            painter.setPen(_cosmetic(_DRAW_COLOR, 1.0))
            painter.drawPolyline(poly)
            painter.setPen(_cosmetic(_DRAW_COLOR, 1.0, Qt.DashLine))
            painter.drawLine(QPointF(*d.cur), QPointF(*d.start))
            if d.capped:
                tag = f"max {LASSO_MAX_VERTICES} points"
        painter.restore()
        if tag:
            cur = transform.map(QPointF(*d.cur))
            painter.save()
            font = painter.font()
            font.setPixelSize(10)
            painter.setFont(font)
            self._roi_draw_area_tag(painter, tag, QRectF(cur.x() + 12, cur.y() + 24, 0, 0), _DRAW_COLOR)
            painter.restore()


class _RoiPixmapLabel(CenteredPixmapLabel, RoiDrawMixin):
    """The viewer's RGB view: ``CenteredPixmapLabel`` plus the ROI tools and overlays.

    In the point tool it behaves exactly like its base class (left click emits
    ``clicked(ix, iy)``).  The base class paints the pixmap, the legacy click dots and the
    shape outline; this class then paints the class overlay, the record overlays and the
    drawing in progress on top.  Construct with no arguments; GUI thread only.

    Signals:
        clicked(int, int), zoom_changed(float): from ``CenteredPixmapLabel``.
        roi_drawn(str, object), roi_discarded(str): from :class:`RoiDrawMixin`.
    """

    def __init__(self, parent=None, background: str = "#0d0d0d"):
        super().__init__(parent, background)
        self._init_roi_draw()

    def paintEvent(self, event):
        """Qt handler: the base view, then the class overlay, records and drawing."""
        super().paintEvent(event)
        iw, ih = self._zp_image_size()
        if iw <= 0 or ih <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        self._roi_paint_class_overlay(painter)
        self._roi_paint_records(painter)
        self._roi_paint_drawing(painter)
        painter.end()

    def mousePressEvent(self, event):
        """Qt handler: left press starts a drawing (rect / lasso tool), else the base behaviour."""
        if self._roi_mouse_press(event):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Qt handler: pan has priority, then the drawing, else the base behaviour."""
        if not self._zp_dragging and self._roi_mouse_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Qt handler: finish a drawing, else the base behaviour (end of a pan)."""
        if self._roi_mouse_release(event):
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MiddleButton:
            self._roi_restore_cursor()

    def keyPressEvent(self, event):
        """Qt handler: Esc cancels a drawing in progress (Enter / Backspace for a polygon)."""
        if self._roi_key_press(event):
            return
        super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Qt handler: a double click closes a polygon being drawn, else the base behaviour."""
        if self._roi_mouse_double_click(event):
            return
        super().mouseDoubleClickEvent(event)
