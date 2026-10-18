"""Qt-independent linear pixel-to-wavelength calibration and profile persistence.

Fits ``wavelength = m * pixel + c`` from matched (pixel, wavelength) points, finds
and refines spectral peaks, extends a few operator-identified peaks to every
detectable Hg-Ar lamp line (:func:`auto_calibrate`) and saves/loads the result as
JSON (:meth:`WavelengthCalibration.save` / :meth:`WavelengthCalibration.load`).

Data conventions:
    * A spectral image is a 2-D array ``(slit rows, spectral columns)``; a *spectrum*
      is one row of it (1-D, length = number of spectral columns).
    * Pixel positions are spectral-column indices (px), zero-based, measured on the
      image at its own hardware bin (``camera_bin``). Wavelengths are in nm.
    * ``m`` is nm per pixel *at that bin*; :attr:`WavelengthCalibration.base_m`
      (``m / camera_bin``) is the unbinned slope that HSI expects.
    * Spectra are returned as float64 copies; inputs are never modified.

Threading:
    Everything here is pure (no global state) and may be called from any thread.
    :class:`WavelengthCalibration` and :class:`AutoCalibrationResult` are immutable.

Example:
    >>> import numpy as np
    >>> from modules.wavelength_calibration import fit_wavelength, detect_peaks, row_spectrum
    >>> cal = fit_wavelength([100, 400, 700], [500.0, 600.0, 700.0], camera_bin=2)
    >>> round(cal.m, 4), round(cal.c, 2), round(cal.base_m, 4)
    (0.3333, 466.67, 0.1667)
    >>> frame = np.zeros((4, 50)); frame[:, 20] = 100.0     # (slit rows, spectral columns)
    >>> detect_peaks(row_spectrum(frame, row=1, smooth=False)).tolist()
    [20]
"""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks, savgol_filter

# Preserves the old tab's positional lamp preset. Users must verify assignments:
# extra/missing detected lines invalidate positional identification.
LEGACY_HGAR = (435.83,546.08,576.96,579.07,None,696.54,None,727.29,738.40,None,None,None,None,794.82,811.53)

from modules.contracts.wavelength import WavelengthCalibration  # the shared profile format lives in contracts

def smooth_spectrum(spectrum,window=7,polyorder=2):
    """Savitzky-Golay smoothing; window is forced odd and limited to the spectrum length.

    Returns the input unchanged when no valid window (>= 3) remains.

    Args:
        spectrum: 1-D intensity array.
        window: Window length in pixels. Even values are reduced by one and the
            window is limited to ``len(spectrum)``.
        polyorder: Polynomial order, clamped into ``[0, window - 1]``.

    Returns:
        numpy.ndarray: Float64 array of the same length (a new array when smoothed).
    """
    spectrum=np.asarray(spectrum,dtype=np.float64)
    size=min(int(window),len(spectrum));size-=1-size%2
    if size<3:return spectrum
    order=max(0,min(int(polyorder),size-1))
    return savgol_filter(spectrum,size,order)

def row_spectrum(frame,row,window=7,polyorder=2,smooth=True):
    """Extract one slit row of a spectral image as a float64 spectrum, optionally smoothed.

    Args:
        frame: 2-D spectral image, shape ``(slit rows, spectral columns)``, any numeric dtype.
        row: Row index; clamped into ``[0, rows - 1]``.
        window: Savitzky-Golay window length in pixels (see :func:`smooth_spectrum`).
        polyorder: Savitzky-Golay polynomial order.
        smooth: If False, return the raw row converted to float64.

    Returns:
        numpy.ndarray: 1-D float64 array of length ``columns`` (a new array; ``frame`` is not modified).

    Raises:
        ValueError: ``frame`` is not 2-D or has fewer than two columns.
    """
    image=np.asarray(frame)
    if image.ndim!=2 or image.shape[1]<2:raise ValueError('A 2-D spectral image is required')
    row=max(0,min(image.shape[0]-1,int(row)))
    spectrum=image[row].astype(np.float64)
    return smooth_spectrum(spectrum,window,polyorder) if smooth else spectrum

def detect_peaks(spectrum,height_fraction=.05,distance=5):
    """Find local maxima of a 1-D spectrum with :func:`scipy.signal.find_peaks`.

    Args:
        spectrum: 1-D intensity array.
        height_fraction: Minimum peak height as a fraction (0..1) of the spectrum maximum.
        distance: Minimum spacing between peaks in pixels (>= 1).

    Returns:
        numpy.ndarray: Integer pixel indices of the peaks, ascending.

    Raises:
        ValueError: ``height_fraction`` outside 0..1 or ``distance`` < 1.
    """
    a=np.asarray(spectrum)
    if not 0<=height_fraction<=1 or distance<1:raise ValueError('Invalid detection settings')
    return find_peaks(a,height=float(np.max(a))*height_fraction,distance=distance)[0]

# Hg and Ar emission lines (nm, air) of a Hg-Ar calibration lamp, used by auto calibration.
HGAR_LINES = (
    404.656, 407.783, 435.833, 546.074, 576.960, 579.067,
    696.543, 706.722, 714.704, 727.294, 738.398, 750.387, 751.465, 763.511,
    772.376, 794.818, 800.616, 801.479, 810.369, 811.531, 826.452, 840.821,
    842.465, 852.144, 866.794, 912.297, 922.450,
)

def refine_peaks(spectrum,peaks):
    """Sub-pixel peak positions by parabolic interpolation of the three samples around each peak.

    Args:
        spectrum: 1-D intensity array.
        peaks: Integer pixel indices of the peaks (e.g. from :func:`detect_peaks`).

    Returns:
        numpy.ndarray: Float64 positions in pixels; each lies within +-0.5 px of its
        input index. Peaks at the first/last sample are returned unrefined.
    """
    a=np.asarray(spectrum,dtype=float);out=[]
    for p in np.asarray(peaks,dtype=int):
        if 0<p<len(a)-1:
            left,mid,right=a[p-1],a[p],a[p+1];denominator=left-2*mid+right
            offset=0.5*(left-right)/denominator if denominator!=0 else 0.0
            out.append(p+float(np.clip(offset,-.5,.5)))
        else:out.append(float(p))
    return np.asarray(out,dtype=float)

@dataclass(frozen=True)
class AutoCalibrationResult:
    """Outcome of :func:`auto_calibrate`.

    Attributes:
        pixels: Sub-pixel positions (px) of all points used in the fit, ascending.
        wavelengths: Wavelength in nm of each point in ``pixels``.
        sources: ``'user'`` for operator-supplied anchors, ``'auto'`` for lines matched to the catalog.
        unmatched: Detected peak positions (px) that matched no catalog line.
        calibration: The final :class:`WavelengthCalibration` fitted to all points.
    """
    pixels: tuple
    wavelengths: tuple
    sources: tuple        # 'user' for the points the operator supplied, 'auto' for matched lamp lines
    unmatched: tuple      # detected peak positions (pixels) that matched no lamp line
    calibration: WavelengthCalibration

def auto_calibrate(spectrum,anchor_pixels,anchor_wavelengths,catalog=HGAR_LINES,tolerance_nm=1.0,
                   height_fraction=.05,distance=5,camera_bin=1,frame_shape=(),selected_row=0,max_iterations=8):
    """Extend a few operator-identified peaks to every detectable lamp line.

    A linear fit through the anchors predicts the wavelength of every other detected peak; a peak
    is accepted when exactly one catalog line lies within ``tolerance_nm`` of that prediction.
    The fit is repeated with the accepted lines until the assignment stops changing.
    The operator's own points are never altered or dropped. Anchors snap to the detected
    peak within 2.5 px of the pixel they were given.

    Args:
        spectrum: 1-D (usually smoothed) intensity array of one slit row.
        anchor_pixels: Pixel positions (px) of at least 3 operator-identified peaks.
        anchor_wavelengths: Their wavelengths in nm.
        catalog: Candidate lamp lines in nm (default :data:`HGAR_LINES`, a Hg-Ar lamp).
        tolerance_nm: Maximum distance (nm) between a predicted wavelength and a
            catalog line for a match (> 0).
        height_fraction: Peak detection threshold as a fraction of the spectrum maximum.
        distance: Minimum peak spacing in pixels.
        camera_bin: Hardware bin of the image, stored in the resulting calibration.
        frame_shape: ``(slit rows, spectral columns)`` stored in the calibration.
        selected_row: Slit row the spectrum came from, stored in the calibration.
        max_iterations: Maximum number of refit/rematch rounds.

    Returns:
        AutoCalibrationResult: all points used, their origin, unmatched peaks and the fit.

    Raises:
        ValueError: fewer than 3 anchors, mismatched/non-finite/duplicate anchors,
            ``tolerance_nm`` <= 0, a catalog with fewer than two lines, anchors that
            give a non-positive dispersion, or two anchors on the same peak.
    """
    anchor_x=np.asarray(anchor_pixels,dtype=float);anchor_y=np.asarray(anchor_wavelengths,dtype=float)
    if anchor_x.ndim!=1 or anchor_x.shape!=anchor_y.shape or len(anchor_x)<3:
        raise ValueError('Auto calibration needs wavelengths for at least 3 peaks')
    if not np.isfinite(anchor_x).all() or not np.isfinite(anchor_y).all():raise ValueError('Points must be finite')
    if len(np.unique(anchor_x))!=len(anchor_x):raise ValueError('Anchor pixels must be distinct')
    if tolerance_nm<=0:raise ValueError('Tolerance must be positive')
    catalog=np.asarray(sorted(catalog),dtype=float)
    if len(catalog)<2:raise ValueError('The line catalog needs at least two lines')

    detected=refine_peaks(spectrum,detect_peaks(spectrum,height_fraction,distance))
    # Anchors snap to the detected peak they were picked from (sub-pixel position).
    anchors=[]
    for x,y in zip(anchor_x,anchor_y):
        if len(detected) and np.min(np.abs(detected-x))<=2.5:x=float(detected[np.argmin(np.abs(detected-x))])
        anchors.append((float(x),float(y)))
    if len({x for x,_ in anchors})!=len(anchors):raise ValueError('Two anchors point at the same peak')
    anchor_set={x for x,_ in anchors}
    candidates=[float(x) for x in detected if x not in anchor_set]
    used_by_anchor=[y for _,y in anchors]

    matched={}                      # pixel -> catalog wavelength
    for _ in range(max_iterations):
        points=anchors+sorted(matched.items())
        m,c=np.polyfit([x for x,_ in points],[y for _,y in points],1)
        if not m>0:raise ValueError('Anchors give a non-positive dispersion; check the assigned wavelengths')
        proposals={}
        for x in candidates:
            predicted=m*x+c;distances=np.abs(catalog-predicted);order=np.argsort(distances)
            best=order[0]
            if distances[best]>tolerance_nm:continue
            if len(order)>1 and distances[order[1]]<=tolerance_nm and distances[order[1]]<1.5*distances[best]:continue  # ambiguous
            line=float(catalog[best])
            if any(abs(line-y)<1e-6 for y in used_by_anchor):continue          # already supplied by the operator
            if line not in proposals or distances[best]<proposals[line][0]:proposals[line]=(float(distances[best]),x)
        new_matched={x:line for line,(_,x) in proposals.items()}
        if new_matched==matched:break
        matched=new_matched

    points=sorted([(x,y,'user') for x,y in anchors]+[(x,y,'auto') for x,y in matched.items()])
    x=[p[0] for p in points];y=[p[1] for p in points]
    calibration=fit_wavelength(x,y,camera_bin,frame_shape,selected_row)
    unmatched=tuple(float(v) for v in candidates if v not in matched)
    return AutoCalibrationResult(tuple(x),tuple(y),tuple(p[2] for p in points),unmatched,calibration)

def fit_wavelength(pixels,wavelengths,camera_bin=1,frame_shape=(),selected_row=0):
    """Fit ``wavelength = m * pixel + c`` by least squares.

    Args:
        pixels: 1-D pixel positions (at the image's own hardware bin).
        wavelengths: Matching wavelengths in nm.
        camera_bin: Hardware bin of the image the pixels were measured on.
        frame_shape: ``(slit rows, spectral columns)`` of the image, or ``()``.
        selected_row: Slit row (0-based) the spectrum came from.

    Returns:
        WavelengthCalibration: slope (nm/px), intercept (nm), R^2 and RMS residual (nm).

    Raises:
        ValueError: fewer than two distinct points, mismatched shapes, non-finite values,
            constant wavelengths, or a non-positive fitted slope (via ``validate``).
    """
    x=np.asarray(pixels,dtype=float);y=np.asarray(wavelengths,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or len(np.unique(x))!=len(x):raise ValueError('Use at least two distinct matched pixels')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Points must be finite')
    if np.ptp(y)==0:raise ValueError('Wavelengths must vary')
    m,c=np.polyfit(x,y,1);res=y-(m*x+c)
    result=WavelengthCalibration(float(m),float(c),float(1-np.sum(res**2)/np.sum((y-y.mean())**2)),float(np.sqrt(np.mean(res**2))),tuple(x),tuple(y),int(camera_bin),tuple(frame_shape),int(selected_row))
    result.validate();return result
