"""Reusable HSI packages; GUI dependencies load only when requested."""

def __getattr__(name):
    if name in {'CalibrationWindow', 'Worker'}:
        from . import radiometric
        value = getattr(radiometric, name)
        globals()[name] = value
        return value
    raise AttributeError(f'{__name__!r} has no attribute {name!r}')
