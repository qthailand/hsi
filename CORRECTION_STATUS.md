# Correction status cards

Apply this overlay to the latest HSI installation (after the stage failure fix), preserving the hsi/ directory structure. Restart the application/kernel.

Scan Window now contains two cards under the image/focus row. StatusCard is a reusable package at modules/status_card; widgets/correction_status.py composes it for HSI.

Green: Applied. Red: Missing. Orange: Off. Cyan: Waiting or Capturing.
Dark is Applied when an enabled reference matches the processed sensor frame shape; disabled subtraction is Off. Smile is Applied only after successful processing with a loaded model. During raw dark capture, Dark is Capturing and Smile is Waiting because both corrections are bypassed. Indicators describe the most recently processed frame, not a physical shutter status.

Status signals are emitted only on changes. The frame loop compares scalar metadata and shape; no image copies, pixel operations, file reads or GUI rendering are added. No FPS benchmark was performed on acquisition hardware.

Validation: 9 targeted tests passed (2 correction/card tests and 7 stage lifecycle tests). The standalone card was rendered and visually inspected; full camera acquisition was not exercised here.
