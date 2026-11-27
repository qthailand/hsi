"""Stand-alone use: ``python -m modules.saturation_profile`` opens the calibration window on a
simulated camera (saves into a temporary folder)."""
# Allow running this file directly (IDE 'run file', runfile): make the project importable.
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "modules.saturation_profile"
from modules._support import launch


def main() -> int:
    """Open the window on :class:`~modules.saturation_profile.demo.SimulatedSource`; returns the Qt exit code."""
    import tempfile

    from .demo import SimulatedSession
    from .store import SaturationProfileStore
    from .widget import SaturationMeasureWidget

    def build():
        widget = SaturationMeasureWidget()
        session = SimulatedSession()
        widget.set_camera(session.source.bins(), 1, 1.0, session.source.gain_limits(), 20.0,
                          session.source.exposure_limits_ms(), 65520, "Simulated camera")
        widget.set_session_provider(SimulatedSession)
        widget.set_store(SaturationProfileStore(tempfile.mkdtemp(prefix="saturation_profile_")))
        return widget
    return launch(build, "Saturation level", (900, 600))


if __name__ == "__main__":
    raise SystemExit(main())
