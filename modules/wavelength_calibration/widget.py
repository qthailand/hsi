"""Embeddable calibration panel preserving the legacy interactive workflow.

Qt widget :class:`WavelengthCalibrationWidget`: image/row selection, Savitzky-Golay
smoothing controls, peak finding, an editable (pixel, wavelength) table, Auto Calibration
against the Hg-Ar catalog, fit/residual plots, profile save/load and apply/save-as-default
requests to the host. Fitting itself lives in :mod:`modules.wavelength_calibration.core`.
*Apply* hands the profile to the host, which applies it and stores it for the camera
setting (``modules.contracts.CalibrationStore``) so Smile and FWHM load it automatically.
FWHM / spectral resolution is measured by ``modules.spectral_resolution``.

Data conventions:
    Images are ``(slit rows, spectral columns)``; table pixels are spectral-column indices
    (px) at the image's hardware bin, wavelengths are nm. The hardware bin is read-only
    in the UI and always comes from the camera/host.

Threading:
    GUI thread only; the widget starts no threads of its own.

Example:
    >>> # python -m modules.wavelength_calibration     # stand-alone window
    >>> # w = WavelengthCalibrationWidget(); w.set_frame(frame, camera_bin=1)  # doctest: +SKIP
"""
import numpy as np
import cv2
import pyqtgraph as pg
from PyQt5.QtCore import Qt,pyqtSignal
from PyQt5.QtWidgets import QAbstractSpinBox,QCheckBox,QFormLayout,QFrame,QGroupBox,QScrollArea,QSizePolicy,QWidget,QVBoxLayout,QHBoxLayout,QPushButton,QTableWidget,QTableWidgetItem,QSpinBox,QDoubleSpinBox,QLabel,QFileDialog,QMessageBox,QHeaderView
from modules.contracts.geometry import default_reference_row
from modules.image import png_name,save_png
from .core import WavelengthCalibration,fit_wavelength,row_spectrum,detect_peaks,LEGACY_HGAR,HGAR_LINES,auto_calibrate
from .line_search import load_line_list,search_calibration
from .report import write_report

class WavelengthCalibrationWidget(QWidget):
    """Interactive wavelength-calibration panel (Qt widget, constructible with no arguments).

    Workflow: load or grab a spectral image, pick the slit row, optionally smooth
    (Savitzky-Golay), find peaks, type known wavelengths into the table (or use
    Auto Calibration), press Calculate, then save/apply the profile.

    Host contract (the widget never touches the host application directly):

    * Inputs (methods/slots): ``set_frame``, ``set_bin``, ``set_bin_provider``,
      ``set_host_available``, ``set_calibration_source``, ``begin_grab``, ``accept_snapshot``,
      ``snapshot_failed``, ``calibration_applied``, ``calibration_stored``.
    * Outputs (signals): ``grabRequested()``, ``applyRequested(WavelengthCalibration)``,
      ``saveDefaultRequested(WavelengthCalibration)``.

    Data conventions: images are 2-D ``(slit rows, spectral columns)``; pixel positions in the
    table are spectral-column indices (px) at the image's hardware bin; wavelengths are nm.
    The hardware-bin box is read-only: the bin always comes from the camera
    (``set_frame``, ``set_bin`` or the bin provider), never from the user.

    Threading: all methods must be called on the GUI thread; the widget starts no threads.
    """
    grabRequested=pyqtSignal()
    applyRequested=pyqtSignal(object)
    saveDefaultRequested=pyqtSignal(object)

    def __init__(self,parent=None):
        """Build the controls and plots.

        Args:
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self.frame=None;self.profile=None;self._grab_request_id=None;self._loading=False
        layout=QHBoxLayout(self);plots=QVBoxLayout()
        panel=QWidget();controls=QVBoxLayout(panel);controls.setContentsMargins(0,0,6,0);controls.setSpacing(8)
        scroll=QScrollArea();scroll.setWidget(panel);scroll.setWidgetResizable(True);scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff);scroll.setSizePolicy(QSizePolicy.Fixed,QSizePolicy.Expanding)
        layout.addWidget(scroll);layout.addLayout(plots,1)
        self._scroll=scroll;self._panel=panel
        self.status=QLabel('Load an image or Grab from Scan');self.status.setWordWrap(True);self.status.setMinimumHeight(60)
        self.status.setAlignment(Qt.AlignTop|Qt.AlignLeft);self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)

        # ── 1. Image ────────────────────────────────────────────
        box,form=self._group(controls,'1 · Image')
        row=QHBoxLayout();form.addRow(row)
        self.btn_grab=self._button(row,'Grab from Scan',lambda:self.grabRequested.emit());self.btn_grab.setEnabled(False)
        self._button(row,'Load Image',self.load_image)
        self.btn_export_png=self._button(row,'Export PNG…',self.export_png);self.btn_export_png.setEnabled(False)
        self.btn_export_png.setToolTip('Save the current image (as grabbed) as an 8/16-bit PNG, values unscaled')
        self.row=QSpinBox();self.row.valueChanged.connect(self.update_spectrum);self.row.setToolTip('Slit row whose spectrum is analysed (default: the Smile reference row)')
        self.bin=QSpinBox();self.bin.setRange(1,64);self.bin.valueChanged.connect(self.invalidate)
        self.bin.setReadOnly(True);self.bin.setButtonSymbols(QAbstractSpinBox.NoButtons);self.bin.setFocusPolicy(Qt.NoFocus)
        self.bin.setToolTip('Hardware bin reported by the Scan Controller for this image (not editable)')
        self._pair(form,('Row',self.row),('Bin (camera)',self.bin))
        self.lbl_setting=QLabel('Camera setting: —');self.lbl_setting.setWordWrap(True);form.addRow(self.lbl_setting)
        self.lbl_stored=QLabel('');self.lbl_stored.setWordWrap(True);form.addRow(self.lbl_stored)

        # ── 2. Smoothing ────────────────────────────────────────
        box,form=self._group(controls,'2 · Smoothing')
        self.smooth=QCheckBox('Savitzky–Golay smoothing');self.smooth.setChecked(True);self.smooth.toggled.connect(self.refresh_smoothing);form.addRow(self.smooth)
        self.sg_window=QSpinBox();self.sg_window.setRange(3,201);self.sg_window.setSingleStep(2);self.sg_window.setValue(7);self.sg_window.setToolTip('Window length in pixels (odd)');self.sg_window.valueChanged.connect(self.refresh_smoothing)
        self.sg_order=QSpinBox();self.sg_order.setRange(1,9);self.sg_order.setValue(2);self.sg_order.setToolTip('Polynomial order (must be smaller than the window)');self.sg_order.valueChanged.connect(self.refresh_smoothing)
        self._pair(form,('Window (px)',self.sg_window),('Order',self.sg_order))

        # ── 3. Peaks ────────────────────────────────────────────
        box,form=self._group(controls,'3 · Peaks')
        self.height=QDoubleSpinBox();self.height.setRange(0,100);self.height.setValue(5);self.height.setSuffix(' %');self.height.setToolTip('Ignore peaks lower than this fraction of the strongest one')
        self.distance=QSpinBox();self.distance.setRange(1,10000);self.distance.setValue(5);self.distance.setSuffix(' px');self.distance.setToolTip('Minimum distance between two peaks')
        self._pair(form,('Min height',self.height),('Min distance',self.distance))
        row=QHBoxLayout();form.addRow(row)
        self._button(row,'Find Peaks',self.find_peaks)
        preset=self._button(row,'Hg-Ar preset',self.preset);preset.setToolTip('Fill wavelengths of a Mercury-Argon lamp by position (verify every line)')

        # ── 4. Points ───────────────────────────────────────────
        box=QGroupBox('4 · Points (pixel ↔ wavelength)');inner=QVBoxLayout(box)
        self.table=QTableWidget(0,2);self.table.setHorizontalHeaderLabels(['Pixel','Wavelength (nm)']);self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(22);self.table.setMinimumHeight(170);self.table.itemChanged.connect(self.invalidate)
        inner.addWidget(self.table,1)
        row=QHBoxLayout();inner.addLayout(row)
        self._button(row,'Add point',self.add_point);self._button(row,'Remove selected',self.remove_points)
        controls.addWidget(box,1)

        # ── 5. Calibration ──────────────────────────────────────
        box,form=self._group(controls,'5 · Calibrate')
        self.tolerance=QDoubleSpinBox();self.tolerance.setRange(.05,20);self.tolerance.setDecimals(2);self.tolerance.setValue(1.0);self.tolerance.setSingleStep(.1);self.tolerance.setSuffix(' nm')
        self.tolerance.setToolTip('With 3+ entered wavelengths: a peak is matched to a listed line only if the line is this close to the predicted wavelength')
        form.addRow('Auto match tolerance',self.tolerance)
        self.catalog=HGAR_LINES;self.catalog_name='Hg-Ar (built-in)'
        self.catalog_label=QLabel();self._show_catalog()
        load_list=QPushButton('Load…');load_list.setToolTip('Reference wavelengths (nm) from a text file, one per line');load_list.clicked.connect(self.load_line_list)
        line=QHBoxLayout();line.addWidget(self.catalog_label,1);line.addWidget(load_list);form.addRow('Line list',line)
        self.tolerance_px=QDoubleSpinBox();self.tolerance_px.setRange(.2,10);self.tolerance_px.setDecimals(1);self.tolerance_px.setValue(2.0);self.tolerance_px.setSingleStep(.5);self.tolerance_px.setSuffix(' px')
        self.tolerance_px.setToolTip('Without entered wavelengths: how close (in pixels) a peak must fall to a listed line')
        form.addRow('Search tolerance',self.tolerance_px)
        row=QHBoxLayout();form.addRow(row)
        self.btn_auto=self._button(row,'Auto Calibration',self.auto_calibrate);self.btn_auto.setEnabled(False)
        self.btn_auto.setToolTip('Enter wavelengths for at least 3 peaks; the remaining Hg-Ar lines are identified automatically')
        self._button(row,'Calculate',self.calculate)
        for signal in (self.table.itemChanged,self.table.model().rowsInserted,self.table.model().rowsRemoved,self.table.model().modelReset):signal.connect(self._update_auto_state)

        # ── 6. Profile ──────────────────────────────────────────
        box,form=self._group(controls,'6 · Profile')
        row=QHBoxLayout();form.addRow(row)
        self._button(row,'Save Profile',self.save_profile);self._button(row,'Load Profile',self.load_profile)
        row=QHBoxLayout();form.addRow(row)
        self.apply=self._button(row,'Apply && Save for this setting',self.apply_profile);self.apply.setEnabled(False)
        self.apply.setToolTip('Apply to the live HSI session and store it for this camera setting (camera, bin, orientation, size); Smile and FWHM load it automatically')
        self.save_default=self._button(row,'Save as Default',self.save_as_default);self.save_default.setEnabled(False)
        export=QPushButton('Export Report (.txt, tab-separated)');export.clicked.connect(self.export_report);form.addRow(export)
        export.setToolTip('Fit, points with residuals and the spectrum, as tab-separated text')
        controls.addWidget(self.status)
        self.image_plot=pg.PlotWidget();self.image=pg.ImageItem(axisOrder='row-major');self.image_plot.addItem(self.image)
        self.image_plot.invertY(True)   # row 0 at the top, as in Smile Analysis (matplotlib imshow)
        self.line=pg.InfiniteLine(pos=0,angle=0,movable=True);self.image_plot.addItem(self.line);self.line.sigPositionChanged.connect(lambda:self.row.setValue(int(round(self.line.value()))))
        self.spectrum_plot=pg.PlotWidget();self.spectrum_plot.setLabel('bottom','Pixel');self.spectrum_plot.setLabel('left','Intensity');self.curve=self.spectrum_plot.plot();self.peaks=self.spectrum_plot.plot(pen=None,symbol='o',symbolBrush='r')
        self.fit_plot=pg.PlotWidget();self.fit_plot.setLabel('bottom','Pixel');self.fit_plot.setLabel('left','Wavelength',units='nm')
        self.residual_plot=pg.PlotWidget();self.residual_plot.setLabel('bottom','Pixel');self.residual_plot.setLabel('left','Residual',units='nm')
        for plot in [self.image_plot,self.spectrum_plot,self.fit_plot,self.residual_plot]:plot.setMinimumHeight(120);plots.addWidget(plot)
        for plot in [self.fit_plot,self.residual_plot]:   # always plain nm (no 'knm'/'mnm' SI prefixes)
            for side in ('left','bottom'):plot.getAxis(side).enableAutoSIPrefix(False)
        self._host=False;self._bin_provider=None;self._store=None;self._setting_provider=None;self.setting=None

    def _group(self,parent_layout,title):
        """Add a titled group box with a ``QFormLayout`` to ``parent_layout``; returns ``(box, form)``."""
        box=QGroupBox(title);form=QFormLayout(box);form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignLeft);form.setVerticalSpacing(6);parent_layout.addWidget(box);return box,form

    def _pair(self,form,left,right):
        """Add one form row holding two ``(label, widget)`` pairs side by side."""
        line=QHBoxLayout();line.setSpacing(6)
        for label,widget in (left,right):
            line.addWidget(QLabel(label));line.addWidget(widget,1)
        form.addRow(line)

    def showEvent(self,event):
        """Size the controls column to its contents (font dependent) the first time the widget is shown."""
        super().showEvent(event)
        width=self._panel.minimumSizeHint().width()+self._scroll.verticalScrollBar().sizeHint().width()+8
        self._scroll.setFixedWidth(max(320,width))

    def _button(self,layout,text,callback):
        """Create a ``QPushButton`` with ``text``, connect ``clicked`` to ``callback`` and add it to ``layout``."""
        button=QPushButton(text);button.clicked.connect(callback);layout.addWidget(button);return button

    def set_bin_provider(self,provider):
        """Callable returning the camera's current hardware bin (or None); used for images loaded from disk."""
        self._bin_provider=provider

    def set_calibration_source(self,store,setting_provider):
        """Where calibrations are stored per camera setting, and how to get the camera's current setting.

        Args:
            store: ``modules.contracts.CalibrationStore`` (or None).
            setting_provider: Callable returning the current ``modules.contracts.CameraSetting`` or None.
        """
        self._store=store;self._setting_provider=setting_provider

    def _frame_setting(self,shape,camera_bin):
        """Camera setting of an image of ``shape`` taken at ``camera_bin`` (None without a provider)."""
        try:setting=self._setting_provider() if self._setting_provider else None
        except Exception:setting=None
        return setting.with_frame(shape,camera_bin) if setting is not None else None

    def _show_setting(self):
        """Show the image's camera setting and what is already stored for it."""
        s=self.setting
        if s is None:
            self.lbl_setting.setText('Camera setting: unknown (no camera connected)');self.lbl_stored.setText('');return
        self.lbl_setting.setText('Camera setting: '+s.label())
        if self._store is None:self.lbl_stored.setText('');return
        try:
            stored=self._store.load_wavelength(s)
            if stored is None:text='Stored for this setting: none yet'
            elif not stored.ok:text='Stored file does not match: '+'; '.join(stored.problems)
            else:
                import time
                p=stored.profile;text=(f'Stored for this setting: {stored.path.name} ({time.strftime("%Y-%m-%d %H:%M",time.localtime(stored.saved_at))}, '
                                       f'{len(p.pixels)} lines, RMS {p.rms_nm:.3g} nm) — Apply replaces it')
        except Exception as exc:text=f'Stored file unreadable: {exc}'
        self.lbl_stored.setText(text)

    def calibration_stored(self,profile,path,label):
        """Host callback: ``profile`` was applied and stored at ``path`` for setting ``label``."""
        if profile is self.profile:
            self.status.setText(self.status.text()+f'\nApplied and saved for {label}: {path}')
        self._show_setting()

    def set_bin(self,value):
        """Programmatic bin update (the user cannot edit it)."""
        self.bin.setValue(int(value))

    def set_host_available(self,value):
        """Tell the widget whether a host can serve snapshots and accept profiles.

        Enables "Grab from Scan" and (after a calculation) "Apply to HSI"/"Save as Default".

        Args:
            value: True when the host is connected.
        """
        self._host=bool(value);self.btn_grab.setEnabled(self._host)

    def invalidate(self,*args):
        """Discard the current fit (unless loading) and disable Apply / Save as Default.

        Connected to every edit of the point table or bin; extra signal arguments are ignored.
        """
        if not self._loading:
            had_fit=self.profile is not None
            self.profile=None;self.apply.setEnabled(False);self.save_default.setEnabled(False)
            self._clear_fit(had_fit)

    def _clear_fit(self,announce=True):
        """Empty the fit and residual plots so they never show a fit that is no longer current.

        Args:
            announce: Also replace the fit summary in the status line (only when a fit was shown).
        """
        self.fit_plot.clear();self.residual_plot.clear()
        if announce:self.status.setText('Fit cleared: edit the points, then Calculate')

    def set_frame(self,frame,camera_bin=1,setting=None):
        """Show a spectral image and reset the point table.

        The spectrum row is set to the shared reference row
        (``modules.contracts.geometry.default_reference_row``, clamped at 0) so that it
        matches Smile Analysis.

        Args:
            frame: 2-D array ``(slit rows, spectral columns)``; converted with ``np.asarray``
                (no copy for an ndarray).
            camera_bin: Hardware bin of the image (>= 1).
            setting: Its ``CameraSetting``; by default taken from the setting provider (if any).

        Raises:
            ValueError: ``frame`` is not 2-D.
        """
        a=np.asarray(frame)
        if a.ndim!=2:raise ValueError('Image must be 2-D')
        self.frame=a;self.btn_export_png.setEnabled(a.dtype in (np.uint8,np.uint16));self.bin.setValue(camera_bin);self.row.setMaximum(a.shape[0]-1);self.line.setBounds([0,a.shape[0]-1]);self.image.setImage(a,autoLevels=True)
        self.row.setValue(max(0,default_reference_row(a.shape[0])))   # same rule as Smile Analysis, so the rows match
        self.setting=setting if setting is not None else self._frame_setting(a.shape,camera_bin);self._show_setting()
        self.table.setRowCount(0);self.update_spectrum()

    def update_spectrum(self,*args):
        """Recompute and draw the spectrum of the selected row; clears peaks, points and the current fit."""
        if self.frame is None:return
        self.invalidate();self.table.setRowCount(0);self.peaks.clear();self.line.setValue(self.row.value())
        self.spectrum=self._spectrum();self.curve.setData(self.spectrum)

    def _spectrum(self):
        """Spectrum of the selected row using the current smoothing controls (see ``row_spectrum``)."""
        return row_spectrum(self.frame,self.row.value(),self.sg_window.value(),self.sg_order.value(),self.smooth.isChecked())

    def refresh_smoothing(self,*args):
        """Re-draw the spectrum with the current smoothing; entered points stay valid."""
        enabled=self.smooth.isChecked();self.sg_window.setEnabled(enabled);self.sg_order.setEnabled(enabled)
        if enabled and self.sg_window.value()%2==0:
            self.sg_window.setValue(self.sg_window.value()+1);return   # re-enters with an odd window
        if self.frame is None:return
        self.spectrum=self._spectrum();self.curve.setData(self.spectrum);self.peaks.clear()

    def load_image(self):
        """Ask for a ``.png``/``.tif``/``.tiff``/``.npy`` file and show it.

        The hardware bin is taken from the bin provider (see ``set_bin_provider``) when
        one is set and returns a value; the file is assumed to have been taken at that bin.
        Errors are shown in a message box.
        """
        path,_=QFileDialog.getOpenFileName(self,'Load calibration image','','Images (*.png *.tif *.tiff *.npy)')
        if not path:return
        try:
            a=np.load(path,allow_pickle=False) if path.endswith('.npy') else cv2.imread(path,cv2.IMREAD_UNCHANGED)
            camera_bin=self._bin_provider() if self._bin_provider is not None else None
            self.set_frame(a,camera_bin or self.bin.value())
            note=f'bin {camera_bin} taken from the camera; the file must have been taken at this bin' if camera_bin else 'no camera connected: bin is unchanged'
            self.status.setText(f'Image loaded ({note}); verify line assignments')
        except Exception as exc:self.error(exc)

    def begin_grab(self,token):
        """Mark a snapshot request as pending and disable the Grab button.

        Args:
            token: Opaque request id; only a snapshot carrying the same id is accepted.
        """
        self._grab_request_id=token;self.btn_grab.setEnabled(False);self.status.setText('Waiting for camera frame…')

    def accept_snapshot(self,packet):
        """Receive the frame requested via ``grabRequested``.

        Args:
            packet: Object with ``request_id``, ``image`` (2-D, oriented, before Smile
                correction and binning) and ``camera_bin``. Ignored if ``request_id`` does
                not match the pending request.
        """
        if packet.request_id!=self._grab_request_id:return
        self._grab_request_id=None;self.btn_grab.setEnabled(True);self.set_frame(packet.image,packet.camera_bin);self.status.setText('Grabbed oriented frame before Smile and binning')

    def snapshot_failed(self,token,message):
        """Report a failed snapshot request and re-enable Grab.

        Args:
            token: Id of the failed request (None matches any); ignored if it is not the pending one.
            message: Text shown in the status label.
        """
        if token is not None and token!=self._grab_request_id:return
        self._grab_request_id=None;self.btn_grab.setEnabled(self._host);self.status.setText(message)

    def _table_points(self):
        """(pixel, wavelength) of every row where both cells hold a number."""
        points=[]
        for row in range(self.table.rowCount()):
            a=self.table.item(row,0);b=self.table.item(row,1)
            try:points.append((float(a.text()),float(b.text())))
            except (AttributeError,ValueError):continue
        return points

    def _update_auto_state(self,*args):
        """Enable Auto Calibration whenever an image is loaded (anchored with 3+ wavelengths, else a full search)."""
        self.btn_auto.setEnabled(self.frame is not None)

    def _show_catalog(self):
        """Show the active line list (name, count and range) next to the Load button."""
        c=self.catalog;self.catalog_label.setText(f'{self.catalog_name}: {len(c)} lines, {min(c):.1f}–{max(c):.1f} nm')
        self.catalog_label.setToolTip(', '.join(f'{v:g}' for v in c))

    def load_line_list(self,path=None):
        """Use the reference wavelengths of a text file (one nm value per line) for Auto Calibration.

        Args:
            path: File to read; asks with a file dialog when omitted.
        """
        if not path:
            path,_=QFileDialog.getOpenFileName(self,'Load line list','','Text (*.txt *.csv *.dat);;All files (*)')
            if not path:return
        try:self.catalog=load_line_list(path)
        except Exception as exc:return self.error(exc)
        import os;self.catalog_name=os.path.basename(path);self._show_catalog()
        self.status.setText(f'Line list loaded: {self.catalog_name} ({len(self.catalog)} lines)')

    def auto_calibrate(self):
        """Identify the lamp lines of the active line list and fit.

        With 3 or more table rows that have wavelengths, those rows are anchors
        (``auto_calibrate``, tolerance in nm); otherwise a full search finds the best
        assignment of peaks to listed lines on its own (``search_calibration``, tolerance in px). The smoothed spectrum, peak
        height/distance settings, match tolerance (nm), bin and row come from the controls.
        On success the table is refilled with anchors, auto-matched lines and unmatched
        peaks (blank wavelength) and the fit is shown. Errors go to a message box.
        """
        if self.frame is None:return self.error('Load or Grab an image first')
        points=self._table_points()
        common=dict(height_fraction=self.height.value()/100,distance=self.distance.value(),camera_bin=self.bin.value(),
                    frame_shape=self.frame.shape,selected_row=self.row.value())
        try:
            if len(points)>=3:
                mode='anchored on your points'
                result=auto_calibrate(self.spectrum,[p[0] for p in points],[p[1] for p in points],catalog=self.catalog,
                                      tolerance_nm=self.tolerance.value(),**common)
            else:
                mode='full search, no wavelengths entered'
                result=search_calibration(self.spectrum,self.catalog,tolerance_px=self.tolerance_px.value(),**common)
        except Exception as exc:return self.error(exc)
        self._loading=True
        try:
            self.table.setRowCount(0)
            rows=[(x,y) for x,y in zip(result.pixels,result.wavelengths)]+[(x,None) for x in result.unmatched]
            for i,(x,y) in enumerate(sorted(rows,key=lambda r:r[0])):
                self.table.insertRow(i);self.table.setItem(i,0,QTableWidgetItem(f'{x:.3f}'))
                self.table.setItem(i,1,QTableWidgetItem('' if y is None else f'{y:.3f}'))
        finally:self._loading=False
        self.profile=result.calibration;self.render_profile()
        added=sum(s=='auto' for s in result.sources)
        self.status.setText(self.status.text()+f'\nAuto ({mode}, {self.catalog_name}): {added} lines identified, {len(result.unmatched)} peaks left unmatched ({len(result.pixels)} points used)')
        self._update_auto_state()

    def add_point(self):
        """Append an empty row to the point table and invalidate the fit."""
        self.table.insertRow(self.table.rowCount());self.invalidate()

    def remove_points(self):
        """Delete the selected table rows and invalidate the fit."""
        for row in sorted({i.row() for i in self.table.selectedIndexes()},reverse=True):self.table.removeRow(row)
        self.invalidate()

    def find_peaks(self):
        """Detect peaks in the current spectrum and list them (pixel column only) in the table.

        Uses the "Peak height (% maximum)" and "Peak distance (pixels)" controls.
        """
        if self.frame is None:return self.error('Load or Grab an image first')
        peaks=detect_peaks(self.spectrum,self.height.value()/100,self.distance.value());self.table.setRowCount(0)
        for i,x in enumerate(peaks):self.table.insertRow(i);self.table.setItem(i,0,QTableWidgetItem(str(x)))
        self.peaks.setData(peaks,self.spectrum[peaks]);self.status.setText(f'{len(peaks)} peaks; enter known wavelengths')

    def preset(self):
        """Fill the wavelength column with the positional legacy Hg-Ar preset (``LEGACY_HGAR``).

        Only valid when the detected peaks line up with the preset; every line must be verified.
        """
        for row in range(self.table.rowCount()):
            value=LEGACY_HGAR[row] if row<len(LEGACY_HGAR) else None
            self.table.setItem(row,1,QTableWidgetItem('' if value is None else str(value)))
        self.status.setText('Legacy positional preset: verify every line; missing/extra peaks shift assignments')

    def calculate(self):
        """Fit the table rows that have a wavelength and show the result.

        Rows without a wavelength are skipped. The fit uses the read-only bin, the image
        shape and the selected row. Errors go to a message box.
        """
        try:
            x=[];y=[]
            for row in range(self.table.rowCount()):
                a=self.table.item(row,0);b=self.table.item(row,1)
                if b is None or not b.text().strip():continue
                if a is None or not a.text().strip():raise ValueError(f'Missing pixel on row {row+1}')
                x.append(float(a.text()));y.append(float(b.text()))
            self.profile=fit_wavelength(x,y,self.bin.value(),() if self.frame is None else self.frame.shape,self.row.value());self.render_profile()
        except Exception as exc:
            self.profile=None;self.apply.setEnabled(False);self.save_default.setEnabled(False)
            self._clear_fit(False);self.error(exc)   # never leave the previous fit on screen after a failed Calculate

    def render_profile(self):
        """Draw the fit and residuals (nm) of ``self.profile``, update the status text and enable the host buttons."""
        p=self.profile;self.fit_plot.clear();self.residual_plot.clear();x=np.asarray(p.pixels);y=np.asarray(p.wavelengths);order=np.argsort(x)
        self.fit_plot.plot(x,y,pen=None,symbol='o');self.fit_plot.plot(x[order],(p.m*x+p.c)[order]);self.residual_plot.plot(x,y-(p.m*x+p.c),pen=None,symbol='o')
        self.status.setText(f'λ = {p.m:.8g} × pixel + {p.c:.8g} nm\nR²={p.r2:.8g}, RMS={p.rms_nm:.6g} nm\nHSI base m={p.base_m:.8g} (bin {p.camera_bin})')
        self.apply.setEnabled(self._host)
        self.save_default.setEnabled(self._host)

    def _table_rows(self):
        """Return ``(pixels, wavelengths)`` of every table row with a numeric pixel (wavelength ``None`` if empty)."""
        pixels=[];known=[]
        for row in range(self.table.rowCount()):
            a=self.table.item(row,0);b=self.table.item(row,1)
            try:x=float(a.text())
            except (AttributeError,ValueError):continue
            try:y=float(b.text())
            except (AttributeError,ValueError):y=None
            pixels.append(x);known.append(y)
        return pixels,known

    def export_png(self):
        """Ask for a path and write the current image unchanged as an 8/16-bit PNG (``modules.image.save_png``)."""
        if self.frame is None:return self.error('Load or Grab an image first')
        path,_=QFileDialog.getSaveFileName(self,'Export image',png_name('wavelength',self.bin.value()),'PNG (*.png)')
        if not path:return
        try:save_png(path,self.frame);self.status.setText(f'Image exported: {path} ({self.frame.shape[1]}x{self.frame.shape[0]} px, {self.frame.dtype})')
        except Exception as exc:self.error(exc)

    def export_report(self):
        """Ask for a path and write the calibration report as tab-separated text (see ``report.py``)."""
        if self.frame is None:return self.error('Load or Grab an image first')
        path,_=QFileDialog.getSaveFileName(self,'Export calibration report','wavelength_calibration.txt','Tab-separated text (*.txt *.tsv)')
        if not path:return
        pixels,known=self._table_rows()
        settings={'image_rows':self.frame.shape[0],'image_columns':self.frame.shape[1],'spectrum_row':self.row.value(),
                  'hardware_bin':self.bin.value(),'smoothing':'savitzky-golay' if self.smooth.isChecked() else 'off',
                  'smoothing_window_px':self.sg_window.value(),'smoothing_order':self.sg_order.value(),
                  'peak_min_height_percent':self.height.value(),'peak_min_distance_px':self.distance.value()}
        if self.setting is not None:settings['camera_setting']=self.setting.label()
        try:
            write_report(path,profile=self.profile,pixels=pixels,wavelengths=known,
                         spectrum_raw=row_spectrum(self.frame,self.row.value(),smooth=False),spectrum_smoothed=self.spectrum,settings=settings)
            self.status.setText(f'Report exported: {path}')
        except Exception as exc:self.error(exc)

    def save_profile(self):
        """Ask for a path and write the current profile as JSON (see ``WavelengthCalibration.save``)."""
        if self.profile is None:return self.error('Calculate first')
        path,_=QFileDialog.getSaveFileName(self,'Save wavelength profile','','JSON (*.json)')
        if path:
            try:self.profile.save(path)
            except Exception as exc:self.error(exc)

    def load_profile(self):
        """Ask for a profile JSON, restore its bin and points into the table and show the fit."""
        path,_=QFileDialog.getOpenFileName(self,'Load wavelength profile','','JSON (*.json)')
        if not path:return
        try:
            p=WavelengthCalibration.load(path);self._loading=True
            try:
                self.bin.setValue(p.camera_bin);self.table.setRowCount(len(p.pixels))
                for i,(x,y) in enumerate(zip(p.pixels,p.wavelengths)):
                    self.table.setItem(i,0,QTableWidgetItem(str(x)));self.table.setItem(i,1,QTableWidgetItem(str(y)))
            finally:self._loading=False
            self.profile=p;self.render_profile()
        except Exception as exc:self.error(exc)

    def calibration_applied(self, profile):
        """Host callback: note in the status label that ``profile`` was applied (ignored if it is not the current one).

        Args:
            profile: The ``WavelengthCalibration`` that was emitted by ``applyRequested``.
        """
        if profile is self.profile:
            self.status.setText(self.status.text() + "\nApplied to current HSI session")

    def apply_profile(self):
        """Emit ``applyRequested`` with the current profile (no-op without one)."""
        if self.profile is not None:self.applyRequested.emit(self.profile)

    def save_as_default(self):
        """Emit ``saveDefaultRequested`` with the current profile (no-op without one)."""
        if self.profile is not None:self.saveDefaultRequested.emit(self.profile)

    def error(self,exc):
        """Show ``exc`` (an exception or text) in a warning message box."""
        QMessageBox.warning(self,'Wavelength Calibration',str(exc))
