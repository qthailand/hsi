# smile/corrector.py
"""Sklearn-style fit / transform API for smile correction."""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .core import _INTERP
from .surface import Surface2D
from .analysis import SmileAnalysis, analyze_smile
from .calibration import (
    SmileCalibration,
    build_maps,
    calibration_from_analysis,
    calibration_from_keystone,
    crop_geometry,
    crop_margin,
    load_calibration,
    save_calibration,
)

__all__ = ["SmileCorrector"]


class SmileCorrector:
    """Sklearn-style smile correction: ``fit`` once, ``transform`` many.

    Usage::

        corrector = SmileCorrector(deg_y=2, deg_x=1, interp="cubic")
        corrector.fit(hgar_frame)
        corrector.save("smile_model.npz")

        corrected = corrector.transform(raw_frame, crop=True)

        corrector = SmileCorrector.load("smile_model.npz")
        corrected = corrector.transform(raw_frame)
        restored = corrector.inverse_transform(corrected)

    Smile + keystone (one combined map, see :mod:`modules.smile.keystone`)::

        corrector = SmileCorrector().fit(lamp_frame)
        corrector.fit_keystone(line_target_frame, white=white_frame)
        corrector.save("smile_model_bin2.npz", camera_setting)   # format 4 (map_x + map_y)

    Args:
        deg_y: Polynomial degree in the spatial (row) direction.
        deg_x: Polynomial degree in the spectral (column) direction.
        interp: Interpolation method ("linear", "cubic", "lanczos").
        sat_level: Sensor full-scale (None = auto-detect).
        snr_min: Minimum peak-to-noise ratio for tracing.
        n_ref_iter: Number of reference-offset iterations.
    """

    def __init__(
        self,
        deg_y: int = 2,
        deg_x: int = 1,
        interp: str = "cubic",
        sat_level: Optional[float] = None,
        snr_min: float = 5.0,
        n_ref_iter: int = 2,
        y_ref: Optional[float] = None,
    ):
        """Store the fit parameters; nothing is fitted until :meth:`fit` or :meth:`load`.

        Args:
            deg_y: Polynomial degree in the spatial (row) direction.
            deg_x: Polynomial degree in the spectral (column) direction.
            interp: ``"linear"``, ``"cubic"`` or ``"lanczos"``.
            sat_level: Sensor full-scale in counts; ``None`` infers it from the frame.
            snr_min: Minimum peak-to-noise ratio for line tracing.
            n_ref_iter: Number of reference-offset refinement iterations.
            y_ref: Zero-displacement reference row (0-based); ``None`` uses the shared
                default (:func:`modules.contracts.default_reference_row`).
        """
        self.deg_y = deg_y
        self.deg_x = deg_x
        self.interp = interp
        self.sat_level = sat_level
        self.snr_min = snr_min
        self.n_ref_iter = n_ref_iter
        self.y_ref = y_ref

        # Fitted state (set by fit() or load()).
        self._surface: Optional[Surface2D] = None
        self._map_x: Optional[np.ndarray] = None
        self._margin: Tuple[int, int] = (0, 0)
        self._image_shape: Optional[Tuple[int, int]] = None
        self._analysis: Optional[SmileAnalysis] = None
        # Keystone state (set by fit_keystone() or load() of a format-4 model).
        self._map_y: Optional[np.ndarray] = None
        self._keystone = None            # KeystoneAnalysis (fitted here)
        self._keystone_json: Optional[dict] = None   # its description (fitted or loaded)
        self._geometry = None            # keystone.GeometryModel

    # ─── Core API ───

    @classmethod
    def from_analysis(cls, analysis: SmileAnalysis) -> SmileCorrector:
        """Create a fitted corrector from an existing analysis result.

        This public factory lets GUI and batch clients reuse a completed
        :class:`SmileAnalysis` without mutating the corrector's private state.
        """
        obj = cls(
            deg_y=analysis.surface.deg_y,
            deg_x=analysis.surface.deg_x,
            interp=analysis.interp,
            sat_level=analysis.sat_level,
            y_ref=analysis.y_ref,
        )
        obj._analysis = analysis
        obj._surface = analysis.surface
        obj._map_x, _, _ = build_maps(analysis.frame_orig.shape, obj._surface)
        obj._margin = tuple(analysis.margin)
        obj._image_shape = tuple(analysis.frame_orig.shape)
        return obj

    def fit_keystone(self, target: np.ndarray, white: Optional[np.ndarray] = None,
                     dark: Optional[np.ndarray] = None, polarity: str = "dark", **kwargs):
        """Add the keystone from a line-target frame: the maps become one smile + keystone map.

        The smile part is re-fitted jointly from the lamp traces of :meth:`fit` (it must have
        been called in this session; a loaded model has no traces).

        Args:
            target: Line-target frame (same camera setting as the lamp frame).
            white: Optional white frame (target removed) to divide out the illumination.
            dark: Optional dark frame for *target* and *white*.
            polarity: ``"dark"`` or ``"bright"`` bars.
            **kwargs: Passed to :func:`modules.smile.keystone.analyze_keystone`.

        Returns:
            The :class:`~modules.smile.keystone.KeystoneAnalysis`.

        Raises:
            RuntimeError: No lamp analysis (not fitted, or loaded from a file).
            ValueError: Unusable target frame (see ``analyze_keystone``).
        """
        from .keystone import analyze_keystone
        if self._analysis is None:
            raise RuntimeError("Keystone needs the lamp analysis: call fit(lamp_frame) first "
                               "(a model loaded from a file has no lamp traces).")
        k = analyze_keystone(self._analysis, target, white=white, dark=dark, polarity=polarity,
                             interp=self.interp, **kwargs)
        return self.attach_keystone(k)

    def attach_keystone(self, k):
        """Use a :class:`~modules.smile.keystone.KeystoneAnalysis` computed elsewhere (e.g. in a
        worker thread from this corrector's :attr:`analysis`): the maps become its joint maps.

        Args:
            k: The keystone analysis; its frame size must equal the fitted one.

        Returns:
            *k*.

        Raises:
            RuntimeError: Not fitted.
            ValueError: *k* belongs to another frame size.
        """
        self._check_fitted()
        if tuple(k.map_x.shape) != tuple(self._image_shape):
            raise ValueError(f"Keystone maps {k.map_x.shape} do not match the fitted frame {self._image_shape}.")
        self._keystone = k
        self._keystone_json = k.to_json()
        self._geometry = k.model
        self._map_x = np.ascontiguousarray(k.map_x, dtype=np.float32)
        self._map_y = np.ascontiguousarray(k.map_y, dtype=np.float32)
        self._margin = tuple(k.margin)
        return k

    def clear_keystone(self) -> SmileCorrector:
        """Drop the keystone and return to the smile-only map of the lamp fit."""
        self._check_fitted()
        self._keystone = self._keystone_json = self._geometry = None
        self._map_y = None
        self._map_x, _, _ = build_maps(self._image_shape, self._surface)
        if self._analysis is not None:
            self._margin = tuple(self._analysis.margin)
        else:
            from .calibration import margin_from_map
            self._margin = margin_from_map(self._map_x, self.interp)
        return self

    @property
    def has_keystone(self) -> bool:
        """True when the maps include the keystone (rows are remapped too)."""
        return self._map_y is not None

    @property
    def keystone(self):
        """The :class:`~modules.smile.keystone.KeystoneAnalysis` of :meth:`fit_keystone`, or ``None``."""
        return self._keystone

    @property
    def row_margin(self) -> Tuple[int, int]:
        """``(top, bottom)`` border rows without a full source (``(0, 0)`` without keystone)."""
        if self._keystone_json and self._keystone_json.get("row_margin"):
            top, bottom = self._keystone_json["row_margin"]
            return int(top), int(bottom)
        return 0, 0

    def fit(self, frame: np.ndarray, verbose: bool = False) -> SmileCorrector:
        """Learn the smile model from a calibration frame.

        Args:
            frame: Hg-Ar calibration image ``(H, W)``.
            verbose: Print analysis report.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If no usable emission lines are found.
        """
        self._analysis = analyze_smile(
            np.asarray(frame, dtype=np.float32),
            sat_level=self.sat_level,
            deg_y=self.deg_y,
            deg_x=self.deg_x,
            interp=self.interp,
            snr_min=self.snr_min,
            n_ref_iter=self.n_ref_iter,
            y_ref=self.y_ref,
            verbose=verbose,
        )
        self.y_ref = self._analysis.y_ref
        self._surface = self._analysis.surface
        self._map_x, _, _ = build_maps(
            self._analysis.frame_orig.shape, self._surface
        )
        self._map_y = None                      # a new lamp fit invalidates a previous keystone
        self._keystone = self._keystone_json = self._geometry = None
        self._margin = self._analysis.margin
        self._image_shape = tuple(self._analysis.frame_orig.shape)
        return self

    def transform(
        self, frame: np.ndarray, crop: bool = True
    ) -> np.ndarray:
        """Apply the fitted smile correction to a new frame.

        Args:
            frame: Image ``(H, W)`` matching the fitted sensor shape.
            crop: Remove invalid border columns, and with keystone border rows
                (data is truly lost).

        Returns:
            Corrected image.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If *frame* shape does not match.
        """
        self._check_fitted()
        frame = np.asarray(frame, dtype=np.float32)
        if frame.shape != self._image_shape:
            raise ValueError(
                f"Frame shape {frame.shape} != fitted {self._image_shape}"
            )
        corrected = cv2.remap(
            np.ascontiguousarray(frame, dtype=np.float32),
            np.ascontiguousarray(self._map_x, dtype=np.float32),
            self._get_map_y(),
            interpolation=_INTERP[self.interp][0],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return crop_geometry(corrected, self._margin, self.row_margin) if crop else corrected

    def fit_transform(
        self, frame: np.ndarray, crop: bool = True, verbose: bool = False
    ) -> np.ndarray:
        """Fit on *frame*, then transform the same frame.

        Args:
            frame: Calibration image ``(H, W)``.
            crop: Remove invalid border columns.
            verbose: Print analysis report.

        Returns:
            Corrected image.
        """
        self.fit(frame, verbose=verbose)
        return self.transform(frame, crop=crop)

    def inverse_transform(self, frame: np.ndarray) -> np.ndarray:
        """Undo the smile correction (reverse displacement).

        Approximate: double interpolation introduces small round-trip
        error.  Useful for coordinate mapping and debugging.  With keystone
        the inverse maps are found by fixed-point iteration of the joint model.

        Args:
            frame: Corrected image ``(H, W)`` (uncropped).

        Returns:
            Image with smile restored.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_fitted()
        n_rows, n_cols = self._image_shape
        if self._geometry is not None:
            ys, xs = np.mgrid[0:n_rows, 0:n_cols].astype(np.float64)
            dX, dY = self._geometry.destination(xs, ys)
            return cv2.remap(
                np.ascontiguousarray(frame, dtype=np.float32),
                np.ascontiguousarray(dX, dtype=np.float32),
                np.ascontiguousarray(dY, dtype=np.float32),
                interpolation=_INTERP[self.interp][0],
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        X, Y = np.meshgrid(
            np.arange(n_cols, dtype=np.float32),
            np.arange(n_rows, dtype=np.float32),
        )
        S = self._surface(Y, X)
        inv_map_x = np.ascontiguousarray(X - S, dtype=np.float32)
        inv_map_y = np.ascontiguousarray(
            np.repeat(
                np.arange(n_rows, dtype=np.float32)[:, None], n_cols, axis=1
            )
        )
        return cv2.remap(
            np.ascontiguousarray(frame, dtype=np.float32),
            inv_map_x,
            inv_map_y,
            interpolation=_INTERP[self.interp][0],
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    # ─── Persistence ───

    def save(self, path: str, camera_setting: Optional[dict] = None) -> None:
        """Save the fitted model to a compressed ``.npz`` file.

        Args:
            path: Destination file path.
            camera_setting: Camera configuration the model belongs to
                (``CameraSetting.to_dict()``), stored in the file so a model of another
                camera setting can be refused; ``None`` stores none.
        """
        self._check_fitted()
        if self._analysis is not None and self._keystone is not None:
            cal = calibration_from_keystone(self._analysis, self._keystone)
            if cal.interp != self.interp:          # set_params(interp=...) after the keystone fit
                cal.set_interpolation(self.interp)
        elif self._analysis is not None:
            cal = calibration_from_analysis(self._analysis)
        else:
            # Reconstruct from fitted state (loaded from file).
            cal = SmileCalibration(
                map_x=self._map_x,
                margin=self._margin,
                image_shape=self._image_shape,
                deg_y=self._surface.deg_y,
                deg_x=self._surface.deg_x,
                surface_coef=self._surface.coef.copy(),
                y_rng=self._surface.y_rng,
                x_rng=self._surface.x_rng,
                interp=self.interp,
                fit_rms=float("nan"),
                cv_rms=float("nan"),
                rms_before=float("nan"),
                rms_after=float("nan"),
                n_lines_used=0,
                n_lines_total=0,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"),
                notes="re-saved from loaded model",
                y_ref=self.y_ref,
                map_y=self._map_y,
                keystone=self._keystone_json if self._map_y is not None else None,
            )
        if camera_setting is not None:
            cal.camera_setting = dict(camera_setting)
        save_calibration(path, cal)

    @classmethod
    def load(cls, path: str) -> SmileCorrector:
        """Load a fitted model from a ``.npz`` file.

        Args:
            path: Path to the saved file.

        Returns:
            A ready-to-use :class:`SmileCorrector`.
        """
        cal = load_calibration(path)
        obj = cls(
            deg_y=cal.deg_y,
            deg_x=cal.deg_x,
            interp=cal.interp,
            y_ref=cal.y_ref,
        )
        obj._map_x = cal.map_x
        obj._margin = cal.margin
        obj._image_shape = tuple(cal.image_shape)
        obj._surface = cal.surface()
        if cal.map_y is not None:
            obj._map_y = cal.map_y
            obj._keystone_json = cal.keystone
            if cal.keystone and cal.keystone.get("model"):
                from .keystone import GeometryModel
                obj._geometry = GeometryModel.from_json(cal.keystone["model"])
        return obj

    # ─── Parameters ───

    def get_params(self) -> dict:
        """Return hyperparameters as a dictionary.

        Returns:
            Dict of parameter names and values.
        """
        return {
            "deg_y": self.deg_y,
            "deg_x": self.deg_x,
            "interp": self.interp,
            "sat_level": self.sat_level,
            "snr_min": self.snr_min,
            "n_ref_iter": self.n_ref_iter,
            "y_ref": self.y_ref,
        }

    def set_params(self, **params: object) -> SmileCorrector:
        """Update hyperparameters.

        Args:
            **params: Parameter name-value pairs.

        Returns:
            self.

        Raises:
            ValueError: If a parameter name is unknown.
        """
        for key, val in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Unknown parameter: {key!r}")
            setattr(self, key, val)
        if "interp" in params and self.is_fitted:
            # The fitted map is interpolation-independent; only the remap
            # kernel changes. Recompute the valid crop for the new kernel.
            from .calibration import margin_from_map
            self._margin = margin_from_map(self._map_x, self.interp)
            if self._map_y is not None and self._keystone_json is not None:
                from .keystone import row_margin_from_map
                w = self._map_x.shape[1]
                self._keystone_json["row_margin"] = list(row_margin_from_map(
                    self._map_y, self.interp, (self._margin[0], w - 1 - self._margin[1])))
        return self

    # ─── Introspection ───

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` or :meth:`load` has been called."""
        return self._surface is not None

    @property
    def margin(self) -> Tuple[int, int]:
        """Border pixels lost by correction ``(left, right)``."""
        return self._margin

    @property
    def analysis(self) -> Optional[SmileAnalysis]:
        """Full analysis result (after :meth:`fit`, not :meth:`load`)."""
        return self._analysis

    def summary(self) -> str:
        """Return a human-readable summary of the fitted model.

        Returns:
            Multi-line summary string.
        """
        if not self.is_fitted:
            return "SmileCorrector (not fitted)"
        left, right = self._margin
        a = self._analysis
        if a is not None:
            cv_str = (
                f"{a.cv_rms:.5f}" if np.isfinite(a.cv_rms) else "n/a"
            )
            return (
                f"SmileCorrector (fitted)\n"
                f"  Sensor       : "
                f"{self._image_shape[1]}x{self._image_shape[0]} px\n"
                f"  Lines used   : {len(a.used_lines)}/{len(a.all_lines)}\n"
                f"  Fit          : deg={self.deg_y}x{self.deg_x}  "
                f"interp={self.interp}\n"
                f"  Straightness : "
                f"{a.rms_before:.4f} -> {a.rms_after:.4f} px\n"
                f"  CV (LOO)     : {cv_str} px\n"
                f"  Border loss  : L={left} R={right} ({left + right} px)"
                + self._keystone_line()
            )
        return (
            f"SmileCorrector (loaded)\n"
            f"  Sensor       : "
            f"{self._image_shape[1]}x{self._image_shape[0]} px\n"
            f"  Fit          : deg={self.deg_y}x{self.deg_x}  "
            f"interp={self.interp}\n"
            f"  Border loss  : L={left} R={right} ({left + right} px)"
            + self._keystone_line()
        )

    def _keystone_line(self) -> str:
        """Keystone part of :meth:`summary` (empty without keystone)."""
        if self._map_y is None:
            return ""
        q = (self._keystone_json or {}).get("qc") or {}
        s = (self._keystone_json or {}).get("summary") or {}
        top, bottom = self.row_margin

        def px(v):
            return "n/a" if v is None else f"{float(v):.3f}"
        return (
            f"\n  Keystone     : max {px(s.get('keystone_max_px'))} px, tilt {px(s.get('tilt_deg'))} deg"
            f"\n  Re-traced    : lamp {px(q.get('retrace_lamp_rms_px'))} px, "
            f"target {px(q.get('retrace_target_rms_px'))} px RMS"
            f"\n  Border rows  : T={top} B={bottom}"
        )

    def __repr__(self) -> str:
        """Parameters and fitted state in one line."""
        status = "fitted" if self.is_fitted else "not fitted"
        return (
            f"SmileCorrector(deg_y={self.deg_y}, deg_x={self.deg_x}, "
            f"interp={self.interp!r}, {status})"
        )

    # ─── Internal ───

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if the corrector has not been fitted or loaded."""
        if self._surface is None:
            raise RuntimeError("Not fitted. Call fit() or load() first.")

    def _get_map_y(self) -> np.ndarray:
        """Row map ``float32 (H, W)``: the keystone map when fitted / loaded, else the identity rows."""
        if self._map_y is not None:
            return np.ascontiguousarray(self._map_y, dtype=np.float32)
        h, w = self._image_shape
        return np.ascontiguousarray(
            np.repeat(
                np.arange(h, dtype=np.float32)[:, None], w, axis=1
            )
        )
