from workers import (
    CubeWorker as _CubeWorker,
    DataThread as _DataThread,
    PreviewCubeThread as _PreviewCubeThread,
    GeoTIFFWorker as _GeoTIFFWorker,
    HypercubeWorker as _HypercubeWorker,
    SpectrogramWorker as _SpectrogramWorker,
    StageThread as _StageThread,
)

__all__ = [
    "StageThread",
    "DataThread",
    "PreviewCubeThread",
    "GeoTIFFWorker",
    "HypercubeWorker",
    "SpectrogramWorker",
    "CubeWorker",
]


_EXPORTS = {
    "StageThread": _StageThread,
    "DataThread": _DataThread,
    "PreviewCubeThread": _PreviewCubeThread,
    "GeoTIFFWorker": _GeoTIFFWorker,
    "HypercubeWorker": _HypercubeWorker,
    "SpectrogramWorker": _SpectrogramWorker,
    "CubeWorker": _CubeWorker,
}


def __getattr__(name):
    if name in _EXPORTS:
        return _EXPORTS[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
