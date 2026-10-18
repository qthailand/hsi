# Latest Smile + shared one-shot Grab

Apply this overlay to HSI stage 1 + the previous Smile integration update.
It includes the latest supplied Smile quality analysis/visualization files and
retains HSI's native uint16 remap adapter.

Modules > Open Smile Analysis > Grab from Scan requests the NEXT normal camera
frame after dark subtraction/hot-pixel correction AND orientation, before
Smile correction and wavelength binning. Dark subtraction may be disabled or
no reference may exist; Grab still works. Active dark-reference capture is
rejected because its raw frames intentionally bypass normal processing.

The worker makes one owned, read-only copy on request only. Normal frames are
not copied for this feature. GUI delivery bypasses the preview FPS throttle.
The Smile widget receives the already-oriented snapshot without flipping again.
Grab supplies the image; click Analyze separately. The host owns QApplication.
Standalone Smile shows a disabled Grab button until a host connects a provider.

## Shared interface for other modules
MainWindow.frame_exchange is the application-owned exchange, not a mutable
module-level global. It exposes:
- request() -> request_id (GUI thread)
- frame_ready(FrameSnapshot)
- failed(request_id, message)
- latest -> last successful snapshot or None
Only one request may be outstanding; new consumers should catch RuntimeError.
Consumers register the returned ID and accept only matching frame_ready packets.
Consumers must not mutate packet.image; copy it if mutation is required.
Connect signals to QObject consumer methods so closing a widget disconnects it.

FrameSnapshot contains image, request_id, timestamp (monotonic seconds),
camera_bin, raw_inverted (orientation already applied), dark_enabled,
dark_applied, and stage='post_dark_hot_pixel_orientation'. The copy's lifetime
is independent of camera buffers, later frames and scan-window closure.
The exchange retains one latest snapshot, not a history. Consumer-held packets
can remain alive independently. Failed requests expire after 10 seconds.

Validation: 51 tests passed, 1 skipped. No physical camera test was performed.
Snapshot copy/delivery adds work only when requested; profile Grab during scans
on the production PC if frame timing is critical. Existing Smile compute cost
and acquisition/recording policy are unchanged by this update.
