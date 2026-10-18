"""Presentation plots for quality reports; no Qt or global backend changes."""
from pathlib import Path
import numpy as np


def export_quality_plots(report, directory, *, dpi=300):
    """Export wavelength, FWHM, area and correlation plots to four PNG files.

    Matplotlib is imported only on request. Missing measurements remain gaps;
    blended lines are omitted. Returns the written paths for host applications.
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    specs = [('wavelength_error_nm', 'Wavelength error (nm)'),
             ('fwhm_nm', 'FWHM (nm)'),
             ('area_counts_px', 'Background-subtracted area (counts px)'),
             ('correlation_shift_px', 'Relative shift from reference row (px)')]
    paths = []
    for key, label in specs:
        fig = Figure(figsize=(9,5), dpi=100)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        for j, wavelength in enumerate(report.wavelengths_nm):
            if report.metadata['excluded_blends'][j]:
                continue
            if not np.isfinite(report.after[key][:,j]).any():
                continue
            line, = ax.plot(report.rows, report.after[key][:,j], label=f'{wavelength:g} nm after')
            ax.plot(report.rows, report.before[key][:,j], '--', color=line.get_color(), alpha=.5)
        ax.set(xlabel='Slit row (zero-based)', ylabel=label,
               title='Solid: after; dashed: before')
        if ax.lines:
            ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.tight_layout()
        path = directory/(key+'.png')
        fig.savefig(path, dpi=dpi, facecolor='white')
        paths.append(str(path))
    return paths
