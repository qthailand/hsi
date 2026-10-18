# frame_exchange

## Overview

A tiny hub that lets any consumer module (smile, wavelength calibration, ...) ask the
application for **one** camera frame without knowing about cameras. The consumer calls
`FrameExchange.request()`, the application's camera code answers with `publish()` or
`reject()`, and the consumer gets `frame_ready` or `failed`. Nothing is copied or
broadcast unless a request is pending.

## Requirements

* PyQt5 for `FrameExchange` (lazy); `FrameSnapshot` needs only the standard library.
* No other `modules.*` package (`__requires__ = ()`).

## Stand-alone use

No widgets, so there is no `python -m modules.frame_exchange`. It is a headless
`QObject`; see the example below, which works with a bare `QApplication`.

## Embedding in an application

```python
from modules.frame_exchange import FrameExchange, FrameSnapshot

hub = FrameExchange(main_window)
hub.capture_requested.connect(grab_one_frame)      # producer: capture, then
#   hub.publish(FrameSnapshot(request_id, image, time.time(), bin, inverted, dark_on, dark_done))
#   or hub.reject(request_id, "reason")
hub.frame_ready.connect(consumer.accept_snapshot)
hub.failed.connect(consumer.snapshot_failed)
request_id = hub.request()                         # consumer side
```

## Data formats / conventions

* `FrameSnapshot` (frozen dataclass, `snapshot.py`): `request_id`, `image`, `timestamp`
  (s), `camera_bin`, `raw_inverted`, `dark_enabled`, `dark_applied`, `stage`
  (default `'post_dark_hot_pixel_orientation'`).
* `image` is a 2-D `(slit, wavelength)` array, shared and **read-only by convention**
  (not copied by the hub); copy before modifying.
* One pending request at a time; ids increase monotonically; stale or unsolicited
  packets/rejections are ignored; the timeout is 10 s.

## API summary

* `FrameExchange(parent=None)`: `request() -> int`, `publish(snapshot)`,
  `reject(request_id, message)`, `latest`; signals `capture_requested(int)`,
  `frame_ready(object)`, `failed(int, str)`.
* `FrameSnapshot`.

## Threading

`FrameExchange` is a `QObject` that lives in the thread that created it (the GUI thread
in HSI). Call `request` from that thread; producers in worker threads should deliver
`publish`/`reject` via queued signal connections.

## Tests

`tests/test_frame_exchange.py`, `tests/test_module_contract.py`.
