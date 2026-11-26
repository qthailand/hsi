"""``FlatFieldDialog``: save the scan just finished as the camera's flat-field profile.

The user scanned a target that is uniform along the slit and spans it (a white strip), presses
*Save as Flat-field Profile* and gets this dialog. It is the radiometric reference dialog
(:class:`~modules.radiometric.reference_dialog.RadiometricReferenceDialog`: the same image, ROI
tools, clean-up and keyboard) without the reflectance table: *Create flat-field profile* fits
:func:`~modules.flatfield.core.compute_flatfield` on the effective ROI and stores the profile with
the scan's camera setting in a :class:`~modules.flatfield.store.FlatFieldStore`.

The small plot shows the ROI's brightness along the slit: the falloff the profile will remove, and
joints between pieces of the target as narrow dips.

*Combine with the current flat-field profile* (on when the store has one that fits): after the
scan is stored, :class:`FlatFieldCombineDialog` shows the current profile, the new scan and their
combination (:func:`~modules.flatfield.core.combine_flatfields` of every scan behind the current
profile plus the new one) side by side; *Combine* stores the combination as the newest profile.
Moving the target between scans and combining again takes the joints and the target's blemishes
out of the profile, scan after scan.

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
import warnings
from typing import List

import numpy as np
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import (QApplication, QCheckBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QLabel,
                             QPushButton, QVBoxLayout)

from modules.radiometric import roi_tools as rt
from modules.radiometric.reference import scan_temperature_c
from modules.radiometric.reference_dialog import RadiometricReferenceDialog, _escape

from .core import DEFAULT_LAMP_RESIDUAL, DEFAULT_MIN_COVERAGE, compute_flatfield
from .store import FlatFieldStore, _key_problems, next_scan_text, plan_for_result

_GUIDE_COLOR = "#2A9D8F"

logger = logging.getLogger(__name__)

__all__ = ["FlatFieldDialog", "FlatFieldViewDialog", "FlatFieldCombineDialog", "DEFAULT_SEAM_PERCENT"]

_QC_COLORS = {"ok": "#2A9D8F", "warn": "#C77D00", "error": "#D1495B"}
_QC_MARKS = {"ok": "✓", "warn": "!", "error": "✕"}

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
        self.combine_base = self._current_profile()
        self.combine_box = QCheckBox("Combine with the current flat-field profile")
        self.advice = None
        advice_text = ""
        if self.combine_base is not None:
            scans = len(self.combine_base.combined_from) or 1
            self.combine_box.setText(f"Combine with {self.combine_base.id} "
                                     f"({scans} scan{'s' if scans != 1 else ''})")
            self.combine_box.setChecked(True)
            try:
                self.advice = self.store.next_scan(self.combine_base)
            except Exception:
                logger.exception("No next-scan advice")
            level, advice_text = next_scan_text(self.advice, self.combine_base.info.get("leave_one_out"),
                                                combined=bool(self.combine_base.combined_from))
            advice_text = f"\n\nAdvice for this scan (green dashes in the plot): {advice_text}" if advice_text else ""
        self.combine_box.setToolTip(
            "After this scan is stored, show the current profile, this scan and the two combined, side by "
            "side, and store the combination if you choose. Move the target along the slit between scans: "
            "where its joints and blemishes lie elsewhere in the other scans they are taken out. Do not "
            "combine after moving a lamp or changing the working distance." + advice_text)
        self.form.addRow("", self.combine_box)
        self.combine_box.setVisible(self.combine_base is not None)
        self.activate_box.setChecked(False)
        self.activate_box.hide()
        self.create_btn.setText("Create flat-field profile")

    def _current_profile(self):
        """The profile the store would use for this scan's setting now (to combine with), or ``None``."""
        if self.store is None or _key_problems(self.key):
            return None
        try:
            return self.store.find(self.key).profile
        except Exception:
            logger.exception("Cannot read the flat-field profiles")
            return None

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
        advice = getattr(self, "advice", None)
        if advice and advice.get("shift"):
            for row in advice.get("joints_next") or []:      # where the advice wanted this scan's joints
                ax.axvline(row, color=_GUIDE_COLOR, lw=1.0, ls="--")
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
        combine = self.combine_base is not None and self.combine_box.isChecked()
        if result["steps"]:
            notes.append("WARNING: the brightness jumps and stays at " + _step_text(result["steps"]) + ". "
                         "The pieces of the target do not lie alike there (another height or tilt, a shadow, "
                         "or another material). The profile takes the jump for light and will put the "
                         "opposite jump into every scan calibrated with it: "
                         + ("combining it with scans where the pieces lie elsewhere takes the jump out."
                            if combine else "level the pieces and scan again, or move the target along the "
                                            "slit, scan again and combine the scans."))
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
                                    info=info, acquired_at=started, measured=result["measured"],
                                    sensor_temperature_c=scan_temperature_c(self.metadata))
        except Exception as exc:
            logger.exception("Flat-field profile could not be stored")
            self._report_error("Create flat-field profile", f"The profile could not be stored: {exc}")
            return None
        if combine:
            saved = self._combine(saved)
        self.reference = saved
        self.reference_saved.emit(saved)
        self.accept()
        return saved

    def _combine(self, saved):
        """Show the current profile, the scan just stored and their combination; the combined profile
        when the user combines, else ``saved`` (the scan alone)."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            dialog = FlatFieldCombineDialog(self.store, self.combine_base, saved, parent=self)
        except Exception as exc:
            logger.exception("Flat-field scans could not be combined")
            self._report_error("Combine flat-field profiles",
                               f"{saved.id} is stored on its own: the scans could not be combined: {exc}")
            return saved
        finally:
            QApplication.restoreOverrideCursor()
        try:
            self._run_combine(dialog)
            return dialog.combined or saved
        finally:
            dialog.deleteLater()

    def _run_combine(self, dialog):
        """Show the combine dialog modally (replaced in tests)."""
        dialog.exec_()


class FlatFieldCombineDialog(QDialog):
    """Compare the current flat-field profile, a new scan and their combination; store it on *Combine*.

    The scans combined are every scan behind ``current`` (its ``combined_from``, or itself) and
    ``new``. Plots: the broadband response along the slit of the three (the new scan's joints and
    the rows left out of it shaded), how many scans each slit row rests on, and how much
    calibrated values change when the combination replaces the current profile.

    Args:
        store: The ``FlatFieldStore`` holding the profiles.
        current: The profile in use now (single or combined).
        new: The single scan just stored.
        parent: Parent widget.

    Raises:
        ValueError: The scans cannot be combined (:meth:`FlatFieldStore.combine`).

    After *Combine*, ``combined`` is the stored profile (``None`` otherwise).
    """

    def __init__(self, store, current, new, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Combine flat-field profiles")
        self.store, self.current, self.new = store, current, new
        self.scans = [s for s in store.scans_of(current) if s.id != new.id] + [new]
        self.result = store.combine(self.scans)
        self.plan = plan_for_result(self.result, len(self.scans) - 1)
        self.combined = None
        layout = QVBoxLayout(self)
        self.summary_label = QLabel(self._summary())
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.figure = Figure(figsize=(10, 8.2), layout="constrained")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        self._plot()
        self.qc_items = self._checks()
        self.qc_label = QLabel(" &nbsp;·&nbsp; ".join(
            f"<b style='color:{_QC_COLORS[i.level]}'>{_QC_MARKS[i.level]}</b>&nbsp;{_escape(i.text)}"
            for i in self.qc_items))
        self.qc_label.setToolTip(self._details())
        self.qc_label.setWordWrap(True)
        layout.addWidget(self.qc_label)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.alone_btn = QPushButton(f"Use {new.id} alone")
        self.alone_btn.setToolTip("Keep the new scan on its own (it is already stored): e.g. after moving a lamp")
        self.alone_btn.clicked.connect(self.reject)
        self.combine_btn = QPushButton(f"Combine {len(self.scans)} scans")
        self.combine_btn.setDefault(True)
        self.combine_btn.clicked.connect(self.combine)
        buttons.addWidget(self.alone_btn)
        buttons.addWidget(self.combine_btn)
        layout.addLayout(buttons)
        self.resize(1040, 960)

    def _summary(self) -> str:
        lines = []
        for scan in self.scans:
            temp = f", {scan.sensor_temperature_c:.1f} °C" if scan.sensor_temperature_c is not None else ""
            joints = scan.info.get("seam_rows") or []
            lines.append(f"{scan.id} (scanned {scan.acquired_at}{temp}, "
                         f"{'joints ' + _row_ranges(joints) if joints else 'no joint found'})"
                         + ("  ← new" if scan.id == self.new.id else ""))
        return (f"Current profile {self.current.id}"
                + (f" combines {len(self.current.combined_from)} scans" if self.current.combined_from else "")
                + f". Combined with the new scan, {len(self.scans)} scans:\n  " + "\n  ".join(lines))

    def _plot(self):
        current = self.current.profile().astype(np.float64)
        new = self.new.measured().astype(np.float64)
        combined = self.result["profile"].astype(np.float64)
        lines, bands = combined.shape
        wl = np.asarray(self.new.key.wavelengths, dtype=np.float64)
        rows = np.arange(lines)
        top, self.next_ax, mid, bottom = self.figure.subplots(4, 1, height_ratios=[2.0, 0.85, 0.32, 1.7])
        draw_next_scan(self.next_ax, self.plan, lines, enough=not self._needs_more())
        self.next_ax.set_title("Next scan: move the target so its joints sit on the green lines", fontsize=9,
                               loc="left")
        for values, color, width, label in ((current, "#7a8699", 1.0, f"current {self.current.id}"),
                                            (new, "#2F6DB5", 1.0, f"new {self.new.id}"),
                                            (combined, "#D1495B", 1.8, f"combined, {len(self.scans)} scans")):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                top.plot(rows, np.nanmedian(values, axis=1), color=color, lw=width, label=label)
        new_index = len(self.scans) - 1
        for a, b in self.result["joints"][new_index]:
            top.axvspan(a - 0.5, b + 0.5, color="#D1495B", alpha=0.25, lw=0)
        for a, b in _runs(self.result["strays"][new_index]):
            top.axvspan(a - 0.5, b + 0.5, color="#C77D00", alpha=0.18, lw=0)
        guides = self.plan.get("joints_next") if self._needs_more() and self.plan.get("shift") else []
        for k, row in enumerate(guides):
            top.axvline(row, color=_GUIDE_COLOR, lw=1.2, ls="--", label="next scan: joints here" if k == 0 else None)
        top.set_xlim(0, lines - 1)
        top.set_ylabel("relative response")
        top.set_title("Broadband response along the slit; red: the new scan's joints, orange: its strays"
                      + ("; green: the next scan's joints" if guides else ""), fontsize=9, loc="left")
        top.legend(fontsize=8, ncol=3)
        top.grid(alpha=0.3)
        per_row = np.asarray(self.result["scans_per_row"])
        top_count = max(3, int(per_row.max()))
        mid.imshow(per_row[None, :], aspect="auto", cmap="YlGn", vmin=0, vmax=top_count,
                   extent=(-0.5, lines - 0.5, 0, 1), interpolation="nearest")
        for a, b in _runs(np.where(per_row <= 1)[0]):
            mid.axvspan(a - 0.5, b + 0.5, color="#C77D00", alpha=0.8, lw=0)
        for row in guides:
            mid.axvline(row, color=_GUIDE_COLOR, lw=1.2, ls="--")
        mid.set_yticks([])
        mid.set_xlim(-0.5, lines - 0.5)
        mid.set_title("Scans each slit row rests on (darker: more; orange: one or none)", fontsize=9, loc="left")
        change = (combined / current - 1.0) * 100.0
        limit = max(1.0, float(np.percentile(np.abs(change), 99)))
        image = bottom.imshow(change.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-limit, vmax=limit,
                              extent=(-0.5, lines - 0.5, wl[0], wl[-1]) if bands > 1 else None,
                              interpolation="nearest")
        bottom.set_xlabel("slit row")
        bottom.set_ylabel("wavelength (nm)")
        bottom.set_title("Change of calibrated values: combined / current − 1 (%)", fontsize=9, loc="left")
        self.figure.colorbar(image, ax=bottom, label="%")

    def _details(self) -> str:
        """The longer story behind the checks, for the tooltip."""
        result = self.result
        lines = [f"{s.id}: {100 * v:.2f} % RMS left after levelling its pieces"
                 for s, v in zip(self.scans, result["residual_rms"])]
        thin = np.where(np.asarray(result["scans_per_row"]) <= 1)[0]
        if thin.size:
            lines.append(f"Rows on one scan or none: {_row_ranges(thin)}")
        strays = result["strays"][len(self.scans) - 1]
        if strays:
            lines.append(f"Left out of the new scan (blemish, curled edge): rows {_row_ranges(strays)}")
        _level, text = next_scan_text(self.plan, result["leave_one_out"], combined=True)
        if text:
            lines.append(text)
        return "\n".join(lines)

    def _needs_more(self) -> bool:
        """True while another scan with the target moved would still help (see ``next_scan_text``)."""
        return next_scan_text(self.plan, self.result["leave_one_out"], combined=True)[0] == "todo"

    def _checks(self) -> List[rt.QcItem]:
        items = []
        result = self.result
        rms = result["residual_rms"]
        odd = [(s.id, v) for s, v in zip(self.scans, rms) if np.isfinite(v) and v > DEFAULT_LAMP_RESIDUAL]
        if odd:
            items.append(rt.QcItem("warn", "Light changed? " + ", ".join(f"{i} {100 * v:.1f} %" for i, v in odd)
                                   + " off the others: use the new scan alone"))
        else:
            items.append(rt.QcItem("ok", f"Scans agree ({100 * max(rms):.2f} % RMS)"))
        loo = result["leave_one_out"]
        if loo is not None:
            items.append(rt.QcItem("ok" if loo <= 0.01 else "warn", f"One scan out changes it {100 * loo:.2f} %"))
        thin = int(np.count_nonzero(np.asarray(result["scans_per_row"]) <= 1))
        items.append(rt.QcItem("warn" if thin else "ok",
                               f"{thin} rows on one scan or none (orange)" if thin else "Every row on two scans or more"))
        items.append(rt.QcItem("ok", "Then make the radiometric reference again"))
        return items

    def combine(self):
        """Store the combination (the newest profile from now on) and close.

        Returns:
            The stored ``FlatFieldProfile``, or ``None`` when it could not be stored.
        """
        try:
            self.combined = self.store.save_combined(self.scans, self.result)
        except Exception as exc:
            logger.exception("Combined flat-field profile could not be stored")
            self.qc_label.setText(f"<b style='color:{_QC_COLORS['error']}'>✕</b>&nbsp;"
                                  f"The combined profile could not be stored: {_escape(str(exc))}")
            return None
        self.accept()
        return self.combined


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
            + ("" if profile.combined_from else f"{len(joints)} joint rows · ")
            + f"scanned {profile.acquired_at}"
            + (f" at {profile.sensor_temperature_c:.1f} °C" if profile.sensor_temperature_c is not None else "")
            + (f"\nCombined of {len(profile.combined_from)} scans: {', '.join(profile.combined_from)}"
               + (f" · leaving one out changes it by up to {100 * float(profile.info['leave_one_out']):.2f} %"
                  if profile.info.get("leave_one_out") is not None else "")
               + (f" · {profile.info.get('single_rows')} rows rest on one scan"
                  if profile.info.get("single_rows") else "")
               if profile.combined_from else "")
            + _advice_line(profile)
            + (f"\nWARNING: brightness steps at {_step_text(steps)}: the pieces of the target did not lie "
               "alike, and scans calibrated with this profile carry the opposite steps" if steps else ""))
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)
        self.figure = Figure(figsize=(9.5, 6.5), layout="tight")
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, 1)
        plan = _plan_of(profile)
        level = next_scan_text(plan, profile.info.get("leave_one_out"), combined=bool(profile.combined_from))[0]
        if level == "todo" and plan and plan.get("shift"):
            top, self.next_ax, bottom = self.figure.subplots(3, 1, height_ratios=[2.0, 0.7, 2.0])
            draw_next_scan(self.next_ax, plan, lines)
            self.next_ax.set_title("Next scan: move the target so its joints sit on the green lines",
                                   fontsize=9, loc="left")
        else:
            self.next_ax = None
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
        self.resize(980, 860 if self.next_ax is not None else 760)


def _step_text(steps) -> str:
    """``[(212, 0.021), (340, -0.013)]`` as ``"row 212 (+2.1 %), row 340 (-1.3 %)"`` (at most six)."""
    parts = [f"row {int(row)} ({100.0 * float(step):+.1f} %)" for row, step in steps]
    return ", ".join(parts[:6]) + (", ..." if len(parts) > 6 else "")


_PIECE_COLORS = ("#D5DBE3", "#AEB9C7")


def draw_next_scan(ax, plan, lines, enough=False):
    """Draw where to move the flat-field target for the next scan on ``ax``.

    Two bars across the slit: *now* (the latest scan's pieces, red joints) and *next* (the same
    pieces moved by the advised shift, green joints), an arrow with the shift in rows between
    them. With ``enough`` a check mark instead.
    """
    ax.set_xlim(-0.03 * lines, 1.03 * lines)
    ax.set_ylim(0, 2.3)
    ax.set_yticks([])
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    if enough or not plan:
        ax.text(0.5 * lines, 1.15, "✓ Enough scans" if enough else "No advice", ha="center", va="center",
                fontsize=12, color=_QC_COLORS["ok"] if enough else "#7a8699")
        ax.set_xticks([])
        return
    shift = int(plan.get("shift") or 0)
    now = [int(c) for c in plan.get("joints_now") or []]
    reach = abs(shift) + 0.03 * lines

    def bar(y, centres, joint_color):
        edges = [-reach] + sorted(centres) + [lines - 1 + reach]
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            ax.add_patch(Rectangle((a, y), b - a, 0.62, facecolor=_PIECE_COLORS[i % 2], edgecolor="none"))
        for c in centres:
            ax.plot([c, c], [y - 0.05, y + 0.67], color=joint_color, lw=3, solid_capstyle="butt")

    bar(1.45, now, "#D1495B")
    bar(0.2, [c + shift for c in now], _GUIDE_COLOR)
    for x in (-0.5, lines - 0.5):                              # the slit
        ax.plot([x, x], [0.05, 2.2], color="#1f2733", lw=1.2)
    ax.text(-0.025 * lines, 1.76, "now", ha="right", va="center", fontsize=9)
    ax.text(-0.025 * lines, 0.51, "next", ha="right", va="center", fontsize=9, color=_GUIDE_COLOR)
    if shift:
        start = 0.5 * lines - 0.5 * shift
        ax.annotate("", xy=(start + shift, 1.13), xytext=(start, 1.13),
                    arrowprops=dict(arrowstyle="-|>", lw=2.2, color=_GUIDE_COLOR, mutation_scale=16))
        ax.text(0.5 * lines, 1.2, f"{abs(shift)} rows", ha="center", va="bottom", fontsize=10,
                color=_GUIDE_COLOR, fontweight="bold")
    ax.set_xlabel("slit row", fontsize=8)
    ax.tick_params(labelsize=7)


def _plan_of(profile):
    """The next-scan advice for ``profile`` (``FlatFieldStore.next_scan``), or ``None``."""
    try:
        return FlatFieldStore(profile.folder.parent).next_scan(profile)
    except Exception:
        logger.exception("No next-scan advice for %s", profile.id)
        return None


def _advice_line(profile) -> str:
    """``"\\n<next-scan advice>"`` for ``profile``, or ``""``."""
    _level, text = next_scan_text(_plan_of(profile), profile.info.get("leave_one_out"),
                                  combined=bool(profile.combined_from))
    return f"\n{text}" if text else ""


def _runs(rows):
    """``[3, 4, 5, 9]`` as ``[(3, 5), (9, 9)]``."""
    out = []
    for r in sorted(int(r) for r in rows):
        if out and r == out[-1][1] + 1:
            out[-1][1] = r
        else:
            out.append([r, r])
    return [tuple(run) for run in out]


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
