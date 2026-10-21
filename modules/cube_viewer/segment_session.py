"""Segmentation state of one cube viewer: classes, method, result, thresholds and the overlay.

``SegmentSession`` (a ``QObject`` owned by :class:`~modules.cube_viewer.window.CubeViewerWindow`)
is the single owner of everything Phase 2 adds on top of the spectra:

* the method key and its parameters (:data:`modules.cube_viewer.segment.METHODS`; importing
  this module imports :mod:`modules.cube_viewer.compare`, which registers Random Forest and
  SVM when scikit-learn is installed);
* the last :class:`~modules.cube_viewer.segment.SegmentResult` and the classes it was
  trained on;
* the reject thresholds: per class ``default x global multiplier`` (angle / distance /
  divergence methods, multiplier 0.5-3) or one minimum confidence (probability methods),
  with optional per-class overrides; re-thresholding uses the stored best-label and score
  maps (``segment.apply_thresholds``), never a recompute;
* the overlay: an Indexed8 ``QImage`` (index 0 = unclassified, alpha 0; index k+1 =
  class k's colour) converted once to a ``QPixmap`` and pushed to every view with
  ``set_class_overlay``.  Opacity changes re-push the same pixmap; colour changes edit the
  colour table only;
* the 'stale' flag: any ROI / class / colour change after a run (:meth:`mark_stale`)
  keeps the result but flags it until the user re-runs or clears.

Runs (build classes + fit + classify), separability and method comparison happen on the
session's OWN :class:`~modules.cube_viewer.jobs.JobRunner` (not the ROI-statistics one), so
ROI drawing stays responsive during a long run.  Each job polls ``cancel`` per chunk.

Data conventions:
    The cube is ``(H, W, B)`` (slit, scan, wavelength), held by reference and never
    copied whole.  Records are :class:`~modules.cube_viewer.spectra.SpectrumRecord`
    objects from ``records_provider()``; a shallow copy of each is taken in the GUI thread
    before a job is queued, so edits during the run cannot change its classes.  Labels are
    int16 with -1 = unclassified; colours are ``'#rrggbb'``.

Threading:
    Create and call from the GUI thread; all signals are emitted in the GUI thread.

Example:
    >>> from modules.cube_viewer.segment_session import SegmentSession     # doctest: +SKIP
    >>> session = SegmentSession(cube, spectrum_set.records, views=(canvas,))  # doctest: +SKIP
    >>> session.result_ready.connect(lambda: print(session.class_counts()))  # doctest: +SKIP
    >>> session.run()                                                     # doctest: +SKIP
"""

from __future__ import annotations

import dataclasses
import time

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QColor, QImage, QPixmap, qRgba

from modules.cube_viewer import compare as _compare
from modules.cube_viewer import segment as _segment
from modules.cube_viewer.jobs import JobRunner

__all__ = ("SegmentSession", "JOB_RUN", "JOB_SEPARABILITY", "JOB_COMPARE",
           "SCALE_RANGE", "DEFAULT_OPACITY", "MAX_CLASSES")

#: JobRunner keys of the three kinds of job (one live job per kind).
JOB_RUN, JOB_SEPARABILITY, JOB_COMPARE = 1, 2, 3
#: Range of the global threshold multiplier.
SCALE_RANGE = (0.5, 3.0)
#: Overlay opacity of a new session.
DEFAULT_OPACITY = 0.5
#: Most classes an Indexed8 overlay can show (index 0 is 'unclassified').
MAX_CLASSES = 255
_PROGRESS_STEP = 0.01          # emit progress at most once per 1 %


def _snapshot(records) -> list:
    """Shallow copies of the records (geometry dict copied too) for a worker job."""
    out = []
    for record in records:
        geometry = dict(record.geometry) if isinstance(record.geometry, dict) else record.geometry
        out.append(dataclasses.replace(record, geometry=geometry))
    return out


def _ready_class_names(records) -> list:
    """Distinct ``class_name`` of the ready records, in creation order."""
    names: dict = {}
    for record in records:
        if getattr(record, "is_ready", False):
            names.setdefault(record.class_name, None)
    return list(names)


class SegmentSession(QObject):
    """Segmentation state and background runs of one viewer (see the module docstring).

    Args:
        cube: ``(H, W, B)`` array (ndarray / memmap / view), held by reference.
        records_provider: ``records_provider() -> list`` of the current spectrum records.
        views: Widgets with ``set_class_overlay(pixmap_or_None, opacity)``.
        parent: Optional ``QObject`` parent.

    Signals:
        state_changed(): anything shown changed (result, counts, stale, visibility, running).
        progress(float, str): ``(fraction 0..1, stage text)`` of the running job.
        result_ready(): a run finished; :attr:`result` is set.
        separability_ready(object): the ``segment.separability`` dict (plus ``'classes'``
            names and ``'build_warnings'``).
        compare_ready(object): the ``compare.compare_methods`` rows (list of dicts).
        failed(str): a job failed (``'<Type>: text'``).
    """

    state_changed = pyqtSignal()
    progress = pyqtSignal(float, str)
    result_ready = pyqtSignal()
    separability_ready = pyqtSignal(object)
    compare_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    _job_progress = pyqtSignal(int, int, float, str)      # (job key, token, fraction, text)

    def __init__(self, cube, records_provider, views=(), parent=None):
        super().__init__(parent)
        self._cube = cube
        self._records_provider = records_provider
        self._views = list(views)
        self._runner = JobRunner(self, name="cube-viewer-segment")
        self._runner.done.connect(self._on_done)
        self._runner.failed.connect(self._on_failed)
        self._job_progress.connect(self._on_job_progress)
        self._tokens = {JOB_RUN: 0, JOB_SEPARABILITY: 0, JOB_COMPARE: 0}
        self._closed = False

        self.method = "sam"
        self._params: dict = {}                      # method key -> overrides
        self.max_per_class = 5000
        self.seed = 0
        self.chunk_rows = 8

        self.result = None                           # segment.SegmentResult
        self.classes: list = []                      # ClassSpec list of the result
        self.warnings: list = []
        self._generation = 0                         # bumped by mark_stale
        self._result_generation = 0
        self.stale = False

        self.threshold_scale = 1.5 / _segment.THRESHOLD_FACTOR   # 1.0 = the default rule
        self.min_confidence = 0.0
        self._overrides: dict = {}                   # class index -> absolute threshold
        self.overlay_visible = True
        self.opacity = DEFAULT_OPACITY

        self._labels = None                          # thresholded (H, W) int16
        self._counts = None
        self._n_unclassified = 0
        self._index = None                           # (H, stride) uint8 behind the QImage
        self._image = None                           # Indexed8 QImage
        self._pixmap = None
        self._run_started = 0.0

    # ── configuration ─────────────────────────────────────

    def set_views(self, views) -> None:
        """Replace the views the overlay is pushed to (the old ones are cleared)."""
        for view in self._views:
            view.set_class_overlay(None, self.opacity)
        self._views = list(views)
        self._push_overlay()

    def set_method(self, key: str, params=None) -> None:
        """Choose the method (a key of ``segment.METHODS``) and optionally its parameter overrides.

        Raises:
            ValueError: Unknown method key.
        """
        if key not in _segment.METHODS:
            raise ValueError(f"unknown method {key!r}")
        self.method = key
        if params is not None:
            self._params[key] = dict(params)
        self.state_changed.emit()

    def params(self, key: str | None = None) -> dict:
        """Parameter overrides of ``key`` (default: the current method)."""
        return dict(self._params.get(key or self.method, {}))

    def set_params(self, key: str, params) -> None:
        """Store parameter overrides of method ``key`` (used by the next run / compare)."""
        self._params[key] = dict(params or {})

    def method_info(self, key: str | None = None):
        """``segment.MethodInfo`` of ``key`` (default: the current method)."""
        return _segment.METHODS[key or self.method]

    def ready_class_names(self) -> list:
        """Class names that currently have at least one ready record."""
        return _ready_class_names(self._records_provider())

    def can_run(self) -> bool:
        """True when >= 2 classes have ready spectra (and the session is open)."""
        return not self._closed and len(self.ready_class_names()) >= 2

    def estimate_seconds(self, key: str | None = None) -> float:
        """Estimated fit + full-image time of ``key`` on this cube (``compare.estimate_seconds``)."""
        key = key or self.method
        h, w, b = (int(s) for s in self._cube.shape)
        k = max(2, len(self.ready_class_names()))
        return float(_compare.estimate_seconds(key, h * w, b, k, self._params.get(key)))

    # ── jobs ──────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        """True while a classification run is queued or running."""
        return self._runner.is_pending(JOB_RUN)

    def job_pending(self, which: int) -> bool:
        """True while a job of kind ``which`` (JOB_RUN / JOB_SEPARABILITY / JOB_COMPARE) is queued or running."""
        return self._runner.is_pending(which)

    def busy(self) -> bool:
        """True while any job (run, separability, compare) is queued or running."""
        return bool(self._runner.pending())

    def run(self) -> None:
        """Build the classes, fit the current method and classify the cube in the worker.

        Emits ``progress`` while running, then ``result_ready`` (or ``failed``).  A run
        already in progress is cancelled first.

        Raises:
            RuntimeError: The session was shut down.
        """
        self._check_open()
        records = _snapshot(self._records_provider())
        cube, method = self._cube, self.method
        params = self._params.get(method)
        max_px, seed, rows = self.max_per_class, self.seed, self.chunk_rows
        generation = self._generation
        report = self._reporter(JOB_RUN)

        def job(cancel):
            report(0.0, "building classes")
            classes, warns = _segment.build_classes(cube, records, max_per_class=max_px,
                                                    seed=seed, cancel=cancel)
            if len(classes) < 2:
                raise ValueError(f"need at least 2 classes with pixels, got {len(classes)}")
            if len(classes) > MAX_CLASSES:
                raise ValueError(f"at most {MAX_CLASSES} classes can be shown, got {len(classes)}")
            report(0.03, "fitting")
            model = _segment.fit(method, classes, params, cancel=cancel)
            report(0.1, "classifying")
            result = _segment.classify_cube(
                cube, model, chunk_rows=rows, cancel=cancel,
                progress=lambda f: report(0.1 + 0.9 * f, "classifying"))
            return classes, list(warns) + list(getattr(model, "warnings", [])), result, generation

        self._run_started = time.perf_counter()
        self._submit(JOB_RUN, job)

    def compute_separability(self, method: str = "sam", params=None) -> None:
        """Build the classes and run ``segment.separability`` in the worker (``separability_ready``)."""
        self._check_open()
        records = _snapshot(self._records_provider())
        cube, max_px, seed = self._cube, self.max_per_class, self.seed
        report = self._reporter(JOB_SEPARABILITY)

        def job(cancel):
            report(0.0, "building classes")
            classes, warns = _segment.build_classes(cube, records, max_per_class=max_px,
                                                    seed=seed, cancel=cancel)
            if len(classes) < 2:
                raise ValueError(f"need at least 2 classes with pixels, got {len(classes)}")
            report(0.3, "separability")
            out = _segment.separability(classes, method=method, params=params, cancel=cancel)
            out["build_warnings"] = list(warns)
            out.setdefault("class_names", [c.name for c in classes])
            out["colors"] = [c.color for c in classes]
            return out

        self._submit(JOB_SEPARABILITY, job)

    def run_compare(self, method_keys, params=None) -> None:
        """Run ``compare.compare_methods`` for ``method_keys`` in the worker (``compare_ready``).

        Args:
            method_keys: Method keys in table order.
            params: Optional ``{key: params}``; default: the session's stored overrides.
        """
        self._check_open()
        records = _snapshot(self._records_provider())
        cube, max_px, seed = self._cube, self.max_per_class, self.seed
        keys = list(method_keys)
        all_params = dict(self._params) if params is None else dict(params)
        shape = tuple(int(s) for s in cube.shape[:2])
        report = self._reporter(JOB_COMPARE)

        def job(cancel):
            report(0.0, "building classes")
            classes, _warns = _segment.build_classes(cube, records, max_per_class=max_px,
                                                     seed=seed, cancel=cancel)
            if len(classes) < 2:
                raise ValueError(f"need at least 2 classes with pixels, got {len(classes)}")
            return _compare.compare_methods(
                classes, keys, params=all_params, cancel=cancel, image_shape=shape,
                progress=lambda f: report(0.05 + 0.95 * f, "comparing methods"))

        self._submit(JOB_COMPARE, job)

    def cancel(self, which=None) -> None:
        """Cancel the running / queued jobs (all, or the given job key)."""
        if which is None:
            self._runner.cancel_all()
        else:
            self._runner.cancel(which)
        self.state_changed.emit()

    def shutdown(self, timeout_ms: int = 2000) -> bool:
        """Cancel every job, stop the worker thread, remove the overlay and drop the cube.

        Idempotent; returns ``JobRunner.shutdown``'s result.
        """
        first = not self._closed
        self._closed = True
        ok = self._runner.shutdown(timeout_ms)
        if first:
            self._drop_result()
            for view in self._views:
                view.set_class_overlay(None, self.opacity)
            self._views = []
            self._cube = None
        return ok

    # ── result, thresholds ────────────────────────────────

    @property
    def has_result(self) -> bool:
        """True when a classification result is stored."""
        return self.result is not None

    @property
    def is_probability(self) -> bool:
        """True when the stored result's (else the current method's) score is a probability."""
        kind = self.result.score_kind if self.result is not None else self.method_info().score_kind
        return kind == "probability"

    def thresholds(self) -> np.ndarray:
        """Effective ``(K,)`` thresholds of the stored result (empty without one)."""
        if self.result is None:
            return np.zeros(0, dtype=np.float64)
        default = np.asarray(self.result.default_thresholds, dtype=np.float64)
        if self.is_probability:
            thr = np.maximum(default, float(self.min_confidence))
        else:
            thr = default * float(self.threshold_scale)
        for k, value in self._overrides.items():
            if 0 <= k < thr.size:
                thr[k] = float(value)
        return thr

    def set_threshold_scale(self, scale: float) -> None:
        """Global multiplier of the default per-class thresholds (clipped to :data:`SCALE_RANGE`)."""
        self.threshold_scale = min(max(float(scale), SCALE_RANGE[0]), SCALE_RANGE[1])
        self._rethreshold()

    def set_min_confidence(self, value: float) -> None:
        """Minimum probability of the probability methods (clipped to 0..1)."""
        self.min_confidence = min(max(float(value), 0.0), 1.0)
        self._rethreshold()

    def set_class_threshold(self, k: int, value=None) -> None:
        """Absolute threshold of class ``k`` (None returns it to the global rule)."""
        if value is None:
            self._overrides.pop(int(k), None)
        else:
            self._overrides[int(k)] = float(value)
        self._rethreshold()

    def labels(self):
        """The thresholded ``(H, W)`` int16 label map (-1 = unclassified), or None."""
        return self._labels

    def class_counts(self):
        """``(counts (K,) int64, n_unclassified)`` of the thresholded map, or None."""
        if self._counts is None:
            return None
        return self._counts.copy(), int(self._n_unclassified)

    def summary(self):
        """What the tree's Segmentation node shows, or None without a result.

        Returns:
            dict with ``method``, ``label``, ``names``, ``colors``, ``counts``,
            ``n_unclassified``, ``total``, ``stale``, ``overlay_visible``, ``running``.
        """
        if self.result is None:
            return None
        info = _segment.METHODS.get(self.result.method)
        return {
            "method": self.result.method,
            "label": info.label if info is not None else self.result.method,
            "names": list(self.result.class_names),
            "colors": list(self.result.colors),
            "counts": [int(c) for c in self._counts],
            "n_unclassified": int(self._n_unclassified),
            "total": int(self._labels.size),
            "stale": bool(self.stale),
            "overlay_visible": bool(self.overlay_visible),
            "running": self.is_running,
        }

    # ── overlay ───────────────────────────────────────────

    def set_overlay_visible(self, visible: bool) -> None:
        """Show / hide the overlay; hiding keeps the result (re-showing is instant)."""
        self.overlay_visible = bool(visible)
        self._push_overlay()
        self.state_changed.emit()

    def set_opacity(self, opacity: float) -> None:
        """Overlay opacity 0..1 (the same pixmap is re-pushed; nothing is rebuilt)."""
        self.opacity = min(max(float(opacity), 0.0), 1.0)
        self._push_overlay()

    def overlay_image(self):
        """The Indexed8 ``QImage`` of the overlay, or None."""
        return self._image

    def refresh_colors(self) -> None:
        """Re-read the class colours from the records (colour of each class's first record)
        and update the colour table only (labels are not touched)."""
        if self.result is None:
            return
        first: dict = {}
        for record in sorted(self._records_provider(), key=lambda r: r.rid):
            first.setdefault(record.class_name, record.color)
        colors = tuple(first.get(name, color)
                       for name, color in zip(self.result.class_names, self.result.colors))
        if colors != tuple(self.result.colors):
            self.result.colors = colors
            self._apply_color_table()
            self.state_changed.emit()

    # ── clear / stale ─────────────────────────────────────

    def clear(self) -> None:
        """Remove the overlay everywhere and forget the result (and cancel a running run).

        ROIs and spectra are not touched.
        """
        self._runner.cancel(JOB_RUN)
        self._drop_result()
        self._push_overlay()
        self.state_changed.emit()

    def mark_stale(self) -> None:
        """Note an ROI / class / colour change: a stored result is flagged stale but kept."""
        self._generation += 1
        if self.result is not None and not self.stale:
            self.stale = True
            self.state_changed.emit()

    # ── internal ──────────────────────────────────────────

    def _check_open(self) -> None:
        """RuntimeError after :meth:`shutdown`."""
        if self._closed or self._cube is None:
            raise RuntimeError("SegmentSession is shut down")

    def _submit(self, key: int, job) -> None:
        """Queue ``job`` under ``key`` (an earlier job of the same kind is cancelled)."""
        self._runner.submit(key, job)
        self.state_changed.emit()

    def _reporter(self, key: int):
        """Thread-safe ``report(fraction, text)`` for one job (throttled to 1 % steps)."""
        self._tokens[key] += 1
        token = self._tokens[key]
        last = [-1.0, ""]
        signal = self._job_progress

        def report(fraction, text):
            fraction = float(fraction)
            if fraction - last[0] >= _PROGRESS_STEP or text != last[1] or fraction >= 1.0:
                last[0], last[1] = fraction, text
                signal.emit(key, token, fraction, text)
        return report

    @pyqtSlot(int, int, float, str)
    def _on_job_progress(self, key: int, token: int, fraction: float, text: str) -> None:
        """GUI thread: forward the progress of a live job."""
        if not self._closed and token == self._tokens.get(key) and self._runner.is_pending(key):
            self.progress.emit(fraction, text)

    def _on_done(self, key: int, payload) -> None:
        """GUI thread: a job finished."""
        if self._closed:
            return
        if key == JOB_RUN:
            classes, warnings, result, generation = payload
            self._drop_result()
            self.classes = list(classes)
            self.warnings = list(warnings)
            self.result = result
            self._result_generation = generation
            self.stale = generation != self._generation
            self._overrides.clear()
            self._rethreshold(emit=False)
            self.result_ready.emit()
            self.state_changed.emit()
        elif key == JOB_SEPARABILITY:
            self.separability_ready.emit(payload)
            self.state_changed.emit()
        elif key == JOB_COMPARE:
            self.compare_ready.emit(payload)
            self.state_changed.emit()

    def _on_failed(self, key: int, message: str) -> None:
        """GUI thread: a job raised."""
        if self._closed:
            return
        self.failed.emit(message)
        self.state_changed.emit()

    def _drop_result(self) -> None:
        """Forget the result, maps and overlay image."""
        self.result = None
        self.classes = []
        self.warnings = []
        self.stale = False
        self._overrides.clear()
        self._labels = self._counts = None
        self._n_unclassified = 0
        self._index = self._image = self._pixmap = None

    def _rethreshold(self, emit: bool = True) -> None:
        """Apply the thresholds to the stored maps, rebuild the overlay and push it."""
        if self.result is None:
            return
        labels = _segment.apply_thresholds(self.result, self.thresholds())
        counts, n_unc = _segment.class_counts(labels, self.result.n_classes)
        self._labels, self._counts, self._n_unclassified = labels, counts, n_unc
        h, w = labels.shape
        stride = (w + 3) & ~3
        if self._index is None or self._index.shape != (h, stride):
            self._index = np.zeros((h, stride), dtype=np.uint8)
        np.add(labels, 1, out=self._index[:, :w], casting="unsafe")
        image = QImage(self._index.data, w, h, stride, QImage.Format_Indexed8)
        self._image = image
        self._apply_color_table()
        if emit:
            self.state_changed.emit()

    def _apply_color_table(self) -> None:
        """Set the colour table (index 0 transparent) and rebuild the pixmap."""
        if self._image is None or self.result is None:
            return
        table = [qRgba(0, 0, 0, 0)]
        for color in self.result.colors:
            c = QColor(color)
            table.append(qRgba(c.red(), c.green(), c.blue(), 255))
        table.extend([qRgba(0, 0, 0, 0)] * (256 - len(table)))
        self._image.setColorTable(table)
        self._pixmap = QPixmap.fromImage(self._image)
        self._push_overlay()

    def _push_overlay(self) -> None:
        """Hand the overlay (or None when hidden / absent) to every view."""
        pixmap = self._pixmap if (self.overlay_visible and self._pixmap is not None) else None
        for view in self._views:
            view.set_class_overlay(pixmap, self.opacity)
