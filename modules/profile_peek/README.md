# profile_peek

## Overview

A *peek* is a small popup card that a status card opens before the full viewer: the curve that
matters most, two or three facts, a warning when there is one, and **Open ▸**.

| Builder | Shows |
|---|---|
| `dark_peek(dark, wavelength=None, leak=None)` | column profile of the dark (median over the slit); light-leak warning |
| `smile_peek(calibration, path='')` | horizontal shift along the slit at the centre column (the smile); straightness before → after |
| `flatfield_peek(profile)` | response along the slit (median over bands); brightness-step warning |
| `reference_peek(reference)` | radiometric factor per band (median over the slit), log scale |
| `text_peek(title, value, details)` | no curve: the card's value and tooltip lines (e.g. wavelength) |

`ProfilePeek` (`QFrame`) shows a `PeekData`: `set_data(data)`, then `popup(anchor)` places it
below the anchor (above when there is no room) on the anchor's screen. A click outside or Esc
closes it; **Open ▸** (or Enter) closes it and emits `openRequested`.

## Requirements

* NumPy (core); PyQt5 and pyqtgraph for `ProfilePeek` only.
* No other `modules.*` package (`__requires__ = ()`): the builders take the dark dict, Smile
  model, `FlatFieldProfile` and `RadiometricReference` by duck typing.

## Stand-alone use

```
python -m modules.profile_peek
```

opens a peek of a synthetic flat-field profile.

## Embedding in an application

```python
from modules.profile_peek import ProfilePeek, dark_peek

peek = ProfilePeek(parent_window)
peek.set_data(dark_peek(dark, wavelength=(m, c)))
peek.openRequested.connect(open_full_dark_viewer)
peek.popup(dark_card)
```

## Data formats / conventions

* Dark `mean` `(slit, wavelength)` counts; flat-field and reference tables `(slit row, band)`;
  wavelengths in nm. Curves are float64 copies.
* `PeekData.color` is a key of `COLORS` (`green`, `red`, `orange`, `cyan`, `grey`), the status
  card palette.
