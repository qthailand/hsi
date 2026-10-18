Apply over the latest HSI build, including the status cards/layout patches. Restart the kernel/application.

Profiler now separates raw fetch from dark subtraction and hot-pixel repair in the shared backend pipeline. Existing read_frame callers are unchanged. Backend read_frame overrides retain their behavior. Raw dark capture bypasses both corrections. Disabled corrections contribute zero samples rather than retaining earlier active timings. Missing references still time the existing no-op kernel calls.

Typical log field order: fetch, dark, hot_pixel, orient, smile, bin, sharpness, emit. Averages are per-stage samples; periodic sharpness/GUI operations do not execute on every acquired frame. This is measurement instrumentation, not a performance optimization.

Four targeted profile tests passed: result/order equivalence, disabled/empty frames, backend override preservation, raw dark capture bypass. Hardware timing has not been measured here.
