# Template module

Copy this folder to `modules/<name>/`, rename, and replace the placeholder content. Run
`python tools/module_audit.py` until it is clean (rules in `docs/MODULE_CONTRACT.md`).

## Overview
What the module does and why it exists.

## Requirements
Python packages / SDKs beyond PyQt5 and numpy.

## Stand-alone use
`python -m modules.<name>`

## Embedding in an application
```python
from PyQt5.QtWidgets import QApplication, QMainWindow
from modules.<name> import TemplateWidget
app = QApplication([]); window = QMainWindow()
widget = TemplateWidget(); window.setCentralWidget(widget); window.show()
widget.valueChanged.connect(print)
widget.set_value(21, 2)
```

## Data formats / conventions
Axes, dtypes, units, file layouts (with `format_version`).

## API summary
`scale(values, factor)` - Qt-free logic; `TemplateWidget` - embeddable widget.

## Threading
Which thread may call what.

## Tests
Where the tests are.
