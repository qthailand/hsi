# HSI - hyperspectral pushbroom acquisition system

Start the application: `python main.py`.

* **Architecture** (layers, one frame/one scan, where logic lives): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
* **Modules** (what each `modules/*` package is, what it requires, how to run/embed it): [docs/MODULES.md](docs/MODULES.md)
* **Module contract** (rules, docstring standard, how to add a module): [docs/MODULE_CONTRACT.md](docs/MODULE_CONTRACT.md)
* **Configuration** (the `config/*.json` files, wavelength profiles per camera setting): [docs/CONFIGURATION.md](docs/CONFIGURATION.md)
* **History** (the October 2026 patches that led to this layout; not a reference): [docs/HISTORY.md](docs/HISTORY.md)

Checks: `python tools/module_audit.py` (contract report) and
`python -m pytest -q tests modules/smile` (use `QT_QPA_PLATFORM=offscreen` on a machine without a display).
