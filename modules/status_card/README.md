# status_card

## Overview

`StatusCard` is a small PyQt5 card (`QFrame`) that shows a title, a large value and a
footer line on a coloured gradient. It accepts arbitrary display values and has no
dependency on HSI services.

## Requirements

* PyQt5 (only when `StatusCard`/`PALETTE` are accessed; `import modules.status_card`
  itself does not import Qt).
* No other `modules.*` package (`__requires__ = ()`).

## Stand-alone use

```
python -m modules.status_card
```

opens a window with a demo card. `StatusCard` needs a `title`, so it is not listed in
`__widgets__` (which holds only no-argument widgets).

## Embedding in an application

```python
from modules.status_card import StatusCard

card = StatusCard("Storage")
card.set_content("42 GB", "Available capacity", color="cyan")
layout.addWidget(card)
```

The directory can also be copied into another project; it only needs PyQt5.

## Data formats / conventions

* `value` and `detail` are converted with `str()`.
* `color` is one of `PALETTE`: `green`, `red`, `orange`, `cyan` (`'#RRGGBB'` strings).
  Anything else raises `ValueError`.
* Footer text longer than 38 characters is shortened to 35 characters plus an
  ellipsis; the tooltip holds the full text.
* Repeating identical `(value, detail, color)` does nothing (no re-styling).

## API summary

* `StatusCard(title, parent=None)`: widget; attributes `title`, `value`, `detail`
  (`QLabel`).
* `StatusCard.set_content(value, detail='', color='cyan')`.
* `PALETTE`: dict of colour name to hex string.

## Threading

Ordinary QWidget: create it and call `set_content` on the GUI thread only. It starts no
threads.

## Tests

`tests/test_correction_status.py` exercises the card through the HSI correction-status
widget; `tests/test_module_contract.py` checks the package contract.
