"""``FlatFieldDialog``: save the scan just finished as the camera's flat-field profile.

The user scanned a target that is uniform along the slit and spans it (a white strip), presses
*Save as Flat-field Profile* and gets this dialog. It is the radiometric reference dialog
(:class:`~modules.radiometric.reference_dialog.RadiometricReferenceDialog`: the same image, ROI
tools, clean-up and keyboard) without the reflectance table: *Create flat-field profile* fits
:func:`~modules.flatfield.core.compute_flatfield` on the effective ROI and stores the profile with
the scan's camera setting in a :class:`~modules.flatfield.store.FlatFieldStore`.

The small plot shows the ROI's brightness along the slit: the falloff the profile will remove, and
joints between pieces of the target as narrow dips.

Data conventions:
    As the reference dialog: ``cube`` is ``(lines, samples, bands)`` dark-subtracted counts.

Threading:
    GUI thread only; the fit runs when *Create flat-field profile* is pressed.

Example:
    >>> from modules.flatfield.dialog import FlatFieldDialog                    # doctest: +SKIP
    >>> dlg = FlatFieldDialog(cube, wavelengths, metadata, store)               # doctest: +SKIP
    >>> if dlg.exec_():                                                         # doctest: +SKIP
    ...     print(dlg.reference.id)
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QApplication, QDialog, QDoubleSpinBox, QFormLayout, QLabel, QVBoxLayout

from modules.radiometric import roi_tools as rt
from modules.radiometric.reference_dialog import RadiometricReferenceDialog, _escape

from .core import DEFAULT_MIN_COVERAGE, compute_flatfield
from .store import _key_problems

logger = logging.getLogger(__name__)

__all__ = ["FlatFieldDialog", "FlatFieldViewDialog", "DEFAULT_SEAM_PERCENT"]

#: Default joint threshold (%): a slit row this far from what its neighbours predict is interpolated.
DEFAULT_SEAM_PERCENT = 3.0
_NO_STORE = "No camera profile is open: the flat-field profile cannot be stored"


class FlatFieldDialog(RadiometricReferenceDialog):
    """Choose the uniform target's pixels in a scan and store the flat-field profile.

    Args:
        cube, wavelengths, metadata, saturation_ceiling, parent: As the reference dialog.
        store: ``FlatFieldStore`` to save into; ``None`` disables *Create flat-field profile*.

    After a successful save, ``reference`` is the stored ``FlatFieldProfile`` (also sent with
    ``reference_saved``).
    """

    def __init__(self, cube, wavelengths, metadata=None, store=None, saturation_ceiling=None, parent=None):
        super().__init__(cube, wavelengths, metadata=metadata, store=store,
                         saturation_ceiling=saturation_ceiling, parent=parent, profiles=[])
        self.setWindowTitle("Save as Flat-field Profile")

    # ── layout: the reference dialog without the reflectance table ──────
    def _build_ui(self, profiles):
        super()._build_ui(profiles)
        hidden = {id(self.profile_combo), id(self.avg_window_spin), id(self.flatfield_box)}
        for row in range(self.form.rowCount()):
            field = self.form.itemAt(row, QFormLayout.FieldRole)
            widgets = ([field.widget()] if field.widget() is not None else
                       [field.layout().itemAt(i).widget() for i in range(field.layout().count())])
            if any(id(w) in hidden for w in widgets):
                label = self.form.itemAt(row, QFormLayout.LabelRole)
                for w in widgets + ([label.widget()] if label is not None else []):
                    if w is not None:
                        w.hide()
        self.seam_spin = QDoubleSpinBox()
        self.seam_spin.setRange(0.0, 50.0)
        self.seam_spin.setDecimals(1)
        self.seam_spin.setSingleStep(0.5)
        self.seam_spin.setValue(DEFAULT_SEAM_PERCENT)
        self.seam_spin.setSuffix(" %")
        self.seam_spin.setSpecialValueText("off")
        self.seam_spin.setToolTip("A slit row whose brightness differs more than this from what its neighbours "
                                  "predict is a joint between pieces of the target (or dirt): it is interpolated "
                                  "from the rows around it. A lasting step between two pieces is not found.")
        self.form.addRow("Joint threshold", self.seam_spin)
        self.activate_box.setChecked(False)
        self.activate_box.hide()
        self.create_btn.setText("Create flat-field profile")

    def _draw_spectrum(self):
        """The ROI's brightness along the slit (median 1), instead of its mean spectrum."""
        ax = self.spectrum_ax
        ax.clear()
        mask = self.cleanup.mask
        rows = np.where(mask.any(axis=1))[0]
        if rows.size:
            sums = np.where(mask, self.brightness, 0.0).sum(axis=1)[rows]
            profile = sums / mask.sum(axis=1)[rows]
            ax.plot(rows, profile / np.median(profile), color="#E9A23B", lw=1.2)
            ax.set_xlim(0, self.shape[0] - 1)
        ax.set_title("ROI brightness along the slit (median 1)", fontsize=8, loc="left")
        ax.set_xlabel("slit row", fontsize=7)
        ax.tick_params(labelsize=7)
        self.spectrum_canvas.draw_idle()

    # ── checks ──────────────────────────────────────────────────────────
    def _coverage_items(self) -> List[rt.QcItem]:
        mask = self.cleanup.mask if self.cleanup is not None else None
        if mask is None or not mask.any():
            return []
        rows = np.where(mask.any(axis=1))[0]
        lines = mask.shape[0]
        share = rows.size / lines
        if share < DEFAULT_MIN_COVERAGE:
            return [rt.QcItem("error", f"The ROI covers {rows.size} of {lines} slit rows ({share:.0%}): a flat-field "
                                       f"target must span the slit (at least {DEFAULT_MIN_COVERAGE:.0%})")]
        outside = int(rows[0] + lines - 1 - rows[-1])
        if outside:
            return [rt.QcItem("warn", f"Slit rows outside {rows[0]}-{rows[-1]} ({outside} rows) are not measured: "
                                      "they take the nearest measured row's response")]
        return [rt.QcItem("ok", f"The ROI spans the slit ({rows.size} of {lines} rows)")]

    def problems(self) -> List[str]:
        """Why *Create flat-field profile* is disabled now (empty = it can be pressed)."""
        problems = _key_problems(self.key)
        if self.store is None:
            problems.append(_NO_STORE)
        problems += [i.text for i in self._coverage_items() if i.level == "error"]
        if self.quality is not None:
            problems += self.quality.errors()
        return problems

    def _show_quality(self):
        colors = {"ok": "#2A9D8F", "warn": "#C77D00", "error": "#D1495B"}
        marks = {"ok": "✓", "warn": "!", "error": "✕"}
        items = [rt.QcItem("error", t) for t in _key_problems(self.key)]
        if self.store is None:
            items.append(rt.QcItem("error", _NO_STORE))
        items += self._coverage_items() + list(self.quality.items if self.quality else ())
        html = "".join(f"<div style='margin:2px 0'><b style='color:{colors[i.level]}'>{marks[i.level]}</b>"
                       f"&nbsp;{_escape(i.text)}</div>" for i in items)
        numbers = self.quality.numbers if self.quality else {}
        html += (f"<div style='color:#5E6A7D;margin-top:6px'>ROI {numbers.get('final_px', 0)} px of "
                 f"{numbers.get('drawn_px', 0)} drawn · saturated {numbers.get('saturated_px', 0)} · "
                 f"inset {numbers.get('inset_removed_px', 0)} · other material "
                 f"{numbers.get('spectral_removed_px', 0)} · row gate {numbers.get('gate_removed_px', 0)}</div>")
        self.qc_label.setText(html)
        self.create_btn.setEnabled(not self.problems())

    # ── create ──────────────────────────────────────────────────────────
    def create_profile(self):
        """Fit the flat-field profile on the effective ROI and store it.

        Returns:
            The stored ``FlatFieldProfile``, or ``None`` when nothing was stored.
        """
        if self.cube is None:                       # closed: its arrays are released
            return None
        problems = self.problems()
        if problems:
            self._report_error("Create flat-field profile", "\n".join(problems))
            return None
        mask = self.cleanup.mask.copy()             # saturated pixels are already left out of it
        seam = self.seam_spin.value() / 100.0
        started = self.metadata.get("started_at")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = compute_flatfield(self.cube, mask, seam_threshold=seam if seam > 0 else None)
        except Exception as exc:
            logger.exception("Flat-field fit failed")
            self._report_error("Create flat-field profile", f"The profile could not be fitted: {exc}")
            return None
        finally:
            QApplication.restoreOverrideCursor()
        row_profile = result["row_profile"]
        notes = []
        if result["seam_rows"]:
            notes.append(f"{len(result['seam_rows'])} slit rows look like joints or dirt (rows "
                         f"{_row_ranges(result['seam_rows'])}) and will be interpolated from their neighbours.")
        if result["steps"]:
            notes.append("WARNING: the brightness jumps and stays at " + _step_text(result["steps"]) + ". "
                         "The pieces of the target do not lie alike there (another height or tilt, a shadow, "
                         "or another material). The profile takes the jump for light and will put the "
                         "opposite jump into every scan calibrated with it: level the pieces and scan again.")
        if notes and not self._confirm("Create flat-field profile",
                                       "\n\n".join(notes) + "\n\nStore the profile?"):
            return None
        info = {"source": f"scan {started}" if started else "scan", "roi": self.roi_description(),
                "qc": dict(self.quality.numbers), "seam_threshold": seam if seam > 0 else None,
                "seam_rows": [int(r) for r in result["seam_rows"]],
                "steps": [[int(row), round(float(step), 4)] for row, step in result["steps"]],
                "extrapolated_rows": int(result["extrapolated_rows"]),
                "invalid_bands": int(result["invalid_bands"]),
                "response_min": round(float(row_profile.min()), 4), "response_max": round(float(row_profile.max()), 4)}
        try:
            saved = self.store.save(result["profile"], self.key, mask=mask, coverage=result["coverage"],
                                    info=info, acquired_at=started)
        except Exception as exc:
            logger.exception("Flat-field profile could not be stored")
            self._report_error("Create flat-field profile", f"The profile could not be stored: {exc}")
            return None
        self.reference = saved
        self.reference_saved.emit(saved)
        self.accept()
        return saved


class FlatFieldViewDialog(QDialog):
    """Show a stored flat-field profile: the response along the slit at five bands, and the whole
    ``(slit row, band)`` profile as an image. Rows interpolated as joints are marked in red.

    Args:
        profile: A ``FlatFieldProfile``.
        parent: Parent widget.
    """

    def __init__(self, profile, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Flat-field profile {profile.id}")
        self.profile = profile
        values = profile.profile()
        lines, bands = values.shape
        wl = np.asarray(profile.key.wavelengths, dtype=np.float64)
        joints = [int(r) for r in (profile.info.get("seam_rows") or [])]
        steps = [(int(row), float(step)) for row, step in (profile.info.get("steps") or [])]
        layout = QVBoxLayout(self)
        self.summary_label = QLabel(
            f"{profile.key.label()} · rows {profile.span[0]}-{profile.span[1]} of {lines} measured "
            f"({profile.coverage:.0%}) · response {float(values.min()):.2f}-{float(values.max()):.2f} · "
            f"{len(joints)} joint rows · scanned {profile.acquired_at}"
            + (f"\nWARNING: brightness steps at {_step_text(steps)}: the pieces of the target did not lie "
               "alike, and scans calibrated with this profile carry the opposite steps" if steps else ""))
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.figure = Figure(figsize=(9.5, 6.5), layout="tight")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        top, bottom = self.figure.subplots(2, 1)
        rows = np.arange(lines)
        for b in np.unique(np.linspace(0, bands - 1, 5).round().astype(int)):
            top.plot(rows, values[:, b], lw=0.9, label=f"{wl[b]:.0f} nm")
        for r in joints:
            top.axvline(r, color="#D1495B", lw=0.5, alpha=0.6)
        for row, _ in steps:
            top.axvline(row - 0.5, color="#C77D00", lw=1.2, ls="--")
        top.axhline(1.0, color="#7a8699", lw=0.6, ls=":")
        top.set_xlim(0, lines - 1)
        top.set_xlabel("slit row")
        top.set_ylabel("relative response")
        top.set_title("Response along the slit (median 1 per band; red: rows interpolated as joints"
                      + ("; orange dashes: brightness steps)" if steps else ")"),
                      fontsize=9, loc="left")
        top.legend(fontsize=8, ncol=5)
        top.grid(alpha=0.3)
        image = bottom.imshow(values, aspect="auto", cmap="viridis", interpolation="nearest",
                              extent=(wl[0], wl[-1], lines - 1, 0) if bands > 1 else None)
        bottom.set_xlabel("wavelength (nm)")
        bottom.set_ylabel("slit row")
        self.figure.colorbar(image, ax=bottom, label="relative response")
        self.resize(980, 760)


def _step_text(steps) -> str:
    """``[(212, 0.021), (340, -0.013)]`` as ``"row 212 (+2.1 %), row 340 (-1.3 %)"`` (at most six)."""
    parts = [f"row {int(row)} ({100.0 * float(step):+.1f} %)" for row, step in steps]
    return ", ".join(parts[:6]) + (", ..." if len(parts) > 6 else "")


def _row_ranges(rows) -> str:
    """``[3, 4, 5, 9]`` as ``"3-5, 9"`` (at most six ranges, then ``...``)."""
    rows = sorted(int(r) for r in rows)
    ranges, start, last = [], rows[0], rows[0]
    for r in rows[1:] + [None]:
        if r is not None and r == last + 1:
            last = r
            continue
        ranges.append(str(start) if start == last else f"{start}-{last}")
        if r is not None:
            start = last = r
    return ", ".join(ranges[:6]) + (", ..." if len(ranges) > 6 else "")
