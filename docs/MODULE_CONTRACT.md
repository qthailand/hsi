# Module contract

Every package under `modules/` is a **self-contained component** that can

1. be **added to the HSI application** (wired by `windows/` and `controllers/`),
2. run **stand-alone** (`python -m modules.<name>`), and
3. be **embedded** in any other PyQt5 application.

The rules below are enforced by `tests/test_module_contract.py`
(`python tools/module_audit.py` prints the same report).

## Layout

```
modules/<name>/
  __init__.py   public API, __all__, __requires__, __widgets__ (Qt is loaded lazily)
  core.py       Qt-free logic, dataclasses, file formats
  widget.py     QWidget(s); optional
  __main__.py   stand-alone launcher (GUI and/or CLI); required if __widgets__ is non-empty
  README.md     overview, stand-alone use, embedding, data formats
```

Other files (`workers.py`, `writers.py`, ...) are fine; keep the split *logic vs Qt*.

## Rules

| # | Rule | Why |
|---|------|-----|
| R1 | `__init__.py` has a docstring, `__all__`, `__requires__` and (if it has widgets) `__widgets__` as literal tuples. | machine-readable surface |
| R2 | Imports of other `modules.*` packages must be listed in `__requires__`; the graph has no cycles. Nothing under `modules/` imports `windows`, `widgets`, `workers`, `controllers`, `utils`, `config`. | modules stay reusable; the app depends on modules, never the reverse |
| R3 | `import modules.<name>` does **not** import PyQt5 (use a lazy `__getattr__` for widget names). | headless use: servers, batch jobs, other GUI toolkits |
| R4 | At least 80 % of public classes/functions/methods have a docstring; every file has a module docstring. | humans can use and extend it |
| R5 | Every class in `__widgets__` is constructible with no arguments, can be shown inside a `QMainWindow` and closed cleanly. | embedding |
| R6 | A module that has widgets ships `__main__.py` using `modules._support.launch`. | stand-alone use |
| R7 | Shared data layouts (profiles, references, cubes) live in `modules.contracts`, not in the module that happens to produce them. | prevents format drift between modules |

`modules/_support` is infrastructure; it may be imported without being declared.

## Widget conventions

* Inputs are **slots/methods** (`set_frame`, `accept_snapshot`), outputs are **signals**
  (`applyRequested`, `analysisComplete`). A widget never reaches into a main window.
* What the host must provide is a small callable or signal, documented in the widget
  docstring (e.g. `set_bin_provider(callable)`), never a reference to the application.
* A widget owns its threads and stops them in `closeEvent`.
* No dependency on the application stylesheet; colours come from `QPalette`.

## Docstring standard (Google style)

Module docstring, in this order:

1. one-line summary;
2. **Data conventions** – array axes (`(slit, scan, wavelength)`), dtypes, units (nm, px, ms);
3. **Threading** – which thread may call what;
4. **Example** – a runnable snippet.

Functions and classes:

```python
def fit_wavelength(pixels, wavelengths, camera_bin=1):
    """Fit ``wavelength = m * pixel + c``.

    Args:
        pixels: 1-D pixel positions (at the image's own hardware bin).
        wavelengths: Matching wavelengths in nm.
        camera_bin: Hardware bin of the image the pixels were measured on.

    Returns:
        WavelengthCalibration: slope, intercept, R^2 and RMS (nm).

    Raises:
        ValueError: fewer than two distinct points, non-finite values, or a flat fit.
    """
```

State **units**, **shape/axes**, **who owns the array** (copy or view) and **thread
safety** wherever they are not obvious. Private helpers (`_name`) need a docstring only
if their behaviour is not clear from the name.

## File formats

Every file a module writes carries `format_version` (and `units` where relevant).
Readers reject unknown versions with a clear message. A format used by more than one
module is defined in `modules.contracts` together with its converters.

## Adding a module

1. Copy `modules/_template/` and rename.
2. Fill in `core.py`, `README.md`, `__init__.py` markers.
3. Run `python tools/module_audit.py` until it is clean; add tests in `tests/`.
4. Register it with the application in `windows/registry.py`.
