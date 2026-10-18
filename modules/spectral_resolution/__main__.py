"""Stand-alone FWHM / spectral-resolution window: ``python -m modules.spectral_resolution``."""
# Allow running this file directly (IDE 'run file'): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.spectral_resolution"


def main():
    """Open :class:`SpectralResolutionWidget` in its own window (load an image and files by hand)."""
    from modules._support import launch
    from .widget import SpectralResolutionWidget
    launch(SpectralResolutionWidget, "FWHM / Spectral Resolution", (1200, 850))


if __name__ == "__main__":
    main()
