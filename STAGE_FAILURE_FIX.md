# Stage failure fix

Apply this overlay to the latest HSI stage 2 installation, preserving the hsi/ directory structure. Restart the Spyder kernel/application before running; do not hot-reload active Qt threads.

Changes:
- Catch Python exceptions inside StageThread.run and emit an explicit error signal.
- Close stage resources in finally and clear references after completion.
- Stop recording and restore Preview on stage failure. A failed scan is not published as a completed cube.
- Prevent another scan until the previous stage thread has finished, including return-home.
- Manual stop requests cooperative cancellation. Closing a window waits up to three seconds; if the stage is still running, closing is deferred instead of forcibly terminating it.
- Report missing XIMC devices clearly and close an opened device if initialization fails.

No camera frame-processing kernels were changed. No FPS benchmark was performed for this patch.

Validation: seven targeted tests passed, including a real QThread with a simulated open failure. Physical XIMC hardware/SDK was unavailable. The reported missing device still requires checking stage power, USB connection, and SDK/device discovery. This patch handles Python failures; it cannot establish or resolve an unobserved native SDK crash.
