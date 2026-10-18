"""CLI for spectral quality validation on a fitted or independent frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .analysis import analyze_smile
from .calibration import load_calibration
from .core import load_frame
from .quality import SpectralReference, analyze_quality, estimate_shift_uncertainty


def main():
    """Generate JSON and bootstrap NPZ diagnostics without importing Qt.

    With --cal, the input can be a separate validation exposure. Without it,
    correction is fitted on the input and results are labelled in-sample.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('--wavelength-cal', required=True,
                        help='JSON with m_nm_per_px, c_nm and standards.')
    parser.add_argument('--cal', help='Previously fitted Smile NPZ model.')
    parser.add_argument('--independent-frame', action='store_true',
                        help='Declare that this exposure was not used to fit --cal.')
    parser.add_argument('--row-step', type=int, default=8)
    parser.add_argument('--resolution-nm', type=float, default=5.)
    parser.add_argument('--interp', choices=['linear', 'cubic', 'lanczos'], default='linear')
    parser.add_argument('--bootstrap', type=int, default=100)
    parser.add_argument('--plots', help='Optional directory for four PNG plots (300 dpi).')
    parser.add_argument('--output', default='smile_quality.json')
    args = parser.parse_args()
    if args.independent_frame and not args.cal:
        parser.error('--independent-frame requires --cal from a different exposure.')
    start = time.perf_counter()
    frame = load_frame(args.image)
    from .quality import normalize_wavelength_config
    config = normalize_wavelength_config(json.loads(Path(args.wavelength_cal).read_text(encoding='utf-8')))
    refs = [SpectralReference(r['wavelength_nm'], r['x_px'],
                             r.get('half_window_px', 32), r.get('blended', False))
            for r in config['standards']]
    analysis = None
    if args.cal:
        model = load_calibration(args.cal)
        model.set_interpolation(args.interp)
        corrected = model.apply(frame)
    else:
        analysis = analyze_smile(frame, interp=args.interp)
        corrected = analysis.frame_corr
    report = analyze_quality(
        frame, corrected, refs, m_nm_per_px=config['m_nm_per_px'], c_nm=config['c_nm'],
        row_step=args.row_step, resolution_nm=args.resolution_nm,
        y_ref=config.get('reference_row_y'), independent_frame=args.independent_frame,
    )
    report.metadata['interpolation'] = args.interp
    report.metadata['elapsed_seconds'] = time.perf_counter()-start
    report.save(args.output)
    if args.plots:
        from .quality_plot import export_quality_plots
        export_quality_plots(report, args.plots)
    if analysis is not None and args.bootstrap:
        rows = report.rows[:, None]
        cols = np.array([r.column_px for r in refs])[None, :]
        uncertainty = estimate_shift_uncertainty(analysis, rows=rows, columns=cols,
                                                  repeats=args.bootstrap)
        np.savez_compressed(Path(args.output).with_suffix('.uncertainty.npz'),
                            rows=report.rows, columns=cols.ravel(), **uncertainty)
    print(json.dumps(report.summary, indent=2))
    print(f'Saved {args.output}; elapsed {time.perf_counter()-start:.3f} s')
    if analysis is None:
        print('Bootstrap omitted: saved maps do not retain raw fitting measurements.')


if __name__ == '__main__':
    main()
