"""Host <-> GPU copy benchmark for the CuPy preprocessing path.

Jetson Orin shares one DRAM between CPU and GPU, yet the pipeline still copies:
a pageable numpy frame is uploaded into a ``cudaMalloc`` array and the binned
result is downloaded back.  This measures, for the frame size of each bin:

* the GPU's memory capabilities (integrated, mapped/pageable access);
* upload from pageable memory (current code) vs from a pinned buffer;
* the real host path: GenICam's ``<< 4`` into a new array then upload (current)
  vs ``<< 4`` written straight into a reused pinned buffer then upload;
* download into pageable memory vs into a pinned buffer;
* zero-copy: one kernel reading a mapped pinned buffer directly vs reading
  device memory (the kernel is the saturating dark subtraction of ``dark16``).

Times are medians in milliseconds over ``--repeat`` runs, host wall time
(``perf_counter``) after ``synchronize``, so they include driver overhead.

``--camera`` adds a real-camera run (GenICam only, settings from the app config;
the ``modules`` code is used read-only, nothing there changes).  Per bin it grabs
``--frames`` frames and, on each real buffer, times the host->GPU path three ways:

* now: GenICam ``_convert`` (new array, ``<< shift``) + pageable upload + kernel;
* A:   ``<< shift`` written into a reused pinned buffer + pinned upload + kernel;
* B:   ``<< shift`` written into a mapped pinned buffer + kernel reading it (zero-copy).

Waiting for the camera is excluded: the clock starts once a buffer is in hand.

Usage:
    python tools/bench_upload.py
    python tools/bench_upload.py --repeat 500
    python tools/bench_upload.py --camera --bins 2,3,4 --frames 200
"""
import argparse
import ctypes
import os
import statistics
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import load_app_config                    # noqa: E402  (project code, used read-only)
from modules.camera import create_backend             # noqa: E402

try:
    import cupy as cp
except Exception as exc:                      # this tool only makes sense where CuPy works
    cp = None
    _CUPY_ERROR = exc

SHAPES = {1: (2048, 2560), 2: (1024, 1280), 3: (682, 848), 4: (512, 640)}   # GenICam (H, W) per bin
BINNED_BINS = 371                                                          # LambdaBinner output width

_DARK = r'''
extern "C" __global__ void dark16(const unsigned short* src, const unsigned short* dark,
 unsigned short* dst, int n) {
 int i=blockDim.x*blockIdx.x+threadIdx.x; if(i>=n)return;
 unsigned short s=src[i], d=dark[i]; dst[i]=s>d?(unsigned short)(s-d):(unsigned short)0;
}
'''

ATTRIBUTES = ('integrated', 'canMapHostMemory', 'unifiedAddressing', 'pageableMemoryAccess',
              'pageableMemoryAccessUsesHostPageTables', 'concurrentManagedAccess',
              'canUseHostPointerForRegisteredMem')


def median_ms(function, repeat):
    """Median host time of ``function()`` followed by a device synchronize, in ms."""
    for _ in range(5):                                   # warm-up: allocations, JIT, caches
        function()
    cp.cuda.Device().synchronize()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        cp.cuda.Device().synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def elapsed_ms(start):
    """Milliseconds since ``start`` after the device has finished its queued work."""
    cp.cuda.Device().synchronize()
    return (time.perf_counter() - start) * 1000.0


def mapped_pinned(shape, dtype):
    """A pinned, device-mapped host buffer: ``(numpy view, cupy view, pointer to free)``."""
    nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
    pointer = cp.cuda.runtime.hostAlloc(nbytes, getattr(cp.cuda.runtime, 'hostAllocMapped', 2))
    host = np.frombuffer((ctypes.c_char * nbytes).from_address(pointer), dtype=dtype).reshape(shape)
    # With unified addressing the host pointer is valid on the device as well.
    memory = cp.cuda.UnownedMemory(pointer, nbytes, None)
    device = cp.ndarray(shape, dtype, cp.cuda.MemoryPointer(memory, 0))
    return host, device, pointer


def pinned_array(shape):
    """A reusable pinned uint16 host array (and the memory object that keeps it alive)."""
    count = int(np.prod(shape))
    memory = cp.cuda.alloc_pinned_memory(count * 2)
    return np.frombuffer(memory, dtype=np.uint16, count=count).reshape(shape), memory


def print_table(header, rows):
    widths = [max(len(str(h)), 8) for h in header]
    print()
    print('  '.join(f'{h:>{w}}' for h, w in zip(header, widths)))
    for row in rows:
        cells = [f'{v:.3f}' if isinstance(v, float) else str(v) for v in row]
        print('  '.join(f'{c:>{w}}' for c, w in zip(cells, widths)))


def synthetic_run(repeat, kernel):
    """Upload/download/zero-copy timings on random frames of each bin's size."""
    rng = np.random.default_rng(0)
    header = ('bin', 'HxW', 'MB', 'up pageable', 'up pinned', 'shift+up now', 'shift>pinned+up',
              'down pageable', 'down pinned', 'kernel devmem', 'kernel zerocopy')
    rows = []
    for bin_value, shape in SHAPES.items():
        raw = rng.integers(0, 1024, shape, dtype=np.uint16)              # 10-bit sensor data
        frame = raw << 4
        dark = cp.asarray(rng.integers(0, 2000, shape, dtype=np.uint16))
        on_device = cp.empty(shape, dtype=cp.uint16)
        out = cp.empty(shape, dtype=cp.uint16)
        pinned_frame, pinned_memory = pinned_array(shape)
        binned = cp.empty((shape[0], BINNED_BINS), dtype=cp.uint16)
        binned_pinned, binned_memory = pinned_array(binned.shape)
        mapped_host, mapped_device, mapped_pointer = mapped_pinned(shape, np.uint16)
        mapped_host[...] = frame
        n = np.int32(frame.size)
        grid = ((frame.size + 255) // 256,)

        def shift_then_upload():                      # current path: new array, pageable upload
            on_device.set(np.left_shift(raw, 4))

        def shift_into_pinned_then_upload():          # proposed: reuse a pinned buffer
            np.left_shift(raw, 4, out=pinned_frame)
            on_device.set(pinned_frame)

        rows.append([bin_value, f'{shape[0]}x{shape[1]}', f'{frame.nbytes / 1e6:.2f}',
                     median_ms(lambda: on_device.set(frame), repeat),
                     median_ms(lambda: on_device.set(pinned_frame), repeat),
                     median_ms(shift_then_upload, repeat),
                     median_ms(shift_into_pinned_then_upload, repeat),
                     median_ms(lambda: binned.get(), repeat),
                     median_ms(lambda: binned.get(out=binned_pinned), repeat),
                     median_ms(lambda: kernel(grid, (256,), (on_device, dark, out, n)), repeat),
                     median_ms(lambda: kernel(grid, (256,), (mapped_device, dark, out, n)), repeat)])
        cp.cuda.runtime.freeHost(mapped_pointer)
    print_table(header, rows)
    print('\nAll times: median ms.  "down" is the binned (H x 371) result, as the pipeline downloads.')


def camera_run(bins, frames, kernel):
    """Grab real GenICam frames and time the host->GPU path now / A / B per bin."""
    config = load_app_config()
    if str(config.camera_backend).lower() != 'genicam':
        print(f'\n--camera supports GenICam only (config camera_backend={config.camera_backend})')
        return
    camera = create_backend(config)
    print(f'\nCamera: genicam, exposure {config.default_exposure_ms} ms, gain {config.default_gain}, '
          f'{frames} frames per bin')
    header = ('bin', 'HxW', 'frames', 'now total', 'A total', 'B total',
              'convert now', 'conv>pinned', 'up pageable', 'up pinned', 'kernel dev', 'kernel zc',
              'copy>mapped', 'shift>mapped')
    rows = []
    try:
        camera.open()
        for bin_value in bins:
            camera.set_bin(bin_value)
            camera.start_stream()
            try:
                row = camera_bin(camera, bin_value, frames, kernel)
                if row is not None:
                    rows.append(row)
            finally:
                camera.stop_stream()
    finally:
        camera.close()
    print_table(header, rows)
    print('\nMedian ms per frame, waiting for the camera excluded.'
          '\n  now = convert now + up pageable + kernel dev'
          '\n  A   = conv>pinned + up pinned + kernel dev'
          '\n  B   = convert into mapped pinned + kernel zc (no upload)'
          '\n  copy>mapped / shift>mapped = camera-buffer hold time: plain copy vs the app\'s copy with shift')


def camera_bin(camera, bin_value, frames, kernel):
    """Median timings for one bin, or ``None`` if the camera keeps failing."""
    names = ('now', 'A', 'B', 'conv', 'conv_pinned', 'up', 'up_pinned', 'kdev', 'kzc', 'copy_mapped', 'shift_mapped')
    times = {name: [] for name in names}
    buffers = None
    misses = 0
    while len(times['now']) < frames:
        with camera._ia.fetch(timeout=1.0) as buffer:              # harvesters takes seconds
            if getattr(buffer, 'is_incomplete', False):
                misses += 1
                if misses > 50:
                    print(f'bin {bin_value}: too many incomplete buffers, skipped')
                    return None
                continue
            component = buffer.payload.components[0]
            data = component.data.reshape(component.height, component.width)
            bits = int(camera._source_bits)
            shift = 16 - bits if 8 < bits < 16 else 0
            if buffers is None or buffers['shape'] != data.shape:
                if buffers is not None:
                    cp.cuda.runtime.freeHost(buffers['mapped_pointer'])
                buffers = camera_buffers(data.shape)
            b = buffers
            n = np.int32(data.size)
            grid = ((data.size + 255) // 256,)

            start = time.perf_counter()                                    # now
            frame = camera._convert(data, bits)
            conv = (time.perf_counter() - start) * 1000.0
            mark = time.perf_counter()
            b['device'].set(frame)
            up = elapsed_ms(mark)
            mark = time.perf_counter()
            kernel(grid, (256,), (b['device'], b['dark'], b['out'], n))
            kdev = elapsed_ms(mark)

            start = time.perf_counter()                                    # A
            np.left_shift(data, shift, out=b['pinned'], casting='unsafe')
            conv_pinned = (time.perf_counter() - start) * 1000.0
            mark = time.perf_counter()
            b['device'].set(b['pinned'])
            up_pinned = elapsed_ms(mark)
            mark = time.perf_counter()
            kernel(grid, (256,), (b['device'], b['dark'], b['out'], n))
            total_a = conv_pinned + up_pinned + elapsed_ms(mark)

            start = time.perf_counter()                                    # B
            np.left_shift(data, shift, out=b['mapped_host'], casting='unsafe')
            kernel(grid, (256,), (b['mapped_device'], b['dark'], b['out'], n))
            total_b = elapsed_ms(start)
            mark = time.perf_counter()
            kernel(grid, (256,), (b['mapped_device'], b['dark'], b['out'], n))
            kzc = elapsed_ms(mark)

            # How long the camera buffer is held: plain copy vs the app's copy with shift.
            start = time.perf_counter()
            np.copyto(b['mapped_host'], data, casting='unsafe')
            copy_mapped = (time.perf_counter() - start) * 1000.0
            start = time.perf_counter()
            np.left_shift(data, np.uint16(shift), out=b['mapped_host'], dtype=np.uint16)
            shift_mapped = (time.perf_counter() - start) * 1000.0

            for name, value in zip(names, (conv + up + kdev, total_a, total_b, conv, conv_pinned,
                                           up, up_pinned, kdev, kzc, copy_mapped, shift_mapped)):
                times[name].append(value)
    cp.cuda.runtime.freeHost(buffers['mapped_pointer'])
    median = {name: statistics.median(values[5:] or values) for name, values in times.items()}  # skip warm-up
    h, w = buffers['shape']
    return [bin_value, f'{h}x{w}', len(times['now']), median['now'], median['A'], median['B'],
            median['conv'], median['conv_pinned'], median['up'], median['up_pinned'],
            median['kdev'], median['kzc'], median['copy_mapped'], median['shift_mapped']]


def camera_buffers(shape):
    """Device, pinned and mapped buffers for one frame size."""
    pinned, pinned_memory = pinned_array(shape)
    mapped_host, mapped_device, mapped_pointer = mapped_pinned(shape, np.uint16)
    return {'shape': shape, 'pinned': pinned, 'pinned_memory': pinned_memory,
            'device': cp.empty(shape, dtype=cp.uint16), 'out': cp.empty(shape, dtype=cp.uint16),
            'dark': cp.zeros(shape, dtype=cp.uint16),
            'mapped_host': mapped_host, 'mapped_device': mapped_device, 'mapped_pointer': mapped_pointer}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repeat', type=int, default=200, help='timed runs per synthetic measurement')
    parser.add_argument('--camera', action='store_true', help='also measure with real GenICam frames')
    parser.add_argument('--bins', default='2,3,4', help='camera bins for --camera, e.g. 1,2,3,4')
    parser.add_argument('--frames', type=int, default=200, help='real frames per bin for --camera')
    args = parser.parse_args(argv)
    if cp is None:
        print(f'CuPy is not available: {_CUPY_ERROR}')
        return 1

    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties.get('name', b'?')
    print(f"GPU: {name.decode() if isinstance(name, bytes) else name}   CuPy {cp.__version__}   "
          f"CUDA runtime {cp.cuda.runtime.runtimeGetVersion()}")
    for attribute in ATTRIBUTES:
        print(f'  {attribute:42s} {properties.get(attribute, "n/a")}')
    kernel = cp.RawKernel(_DARK, 'dark16')

    synthetic_run(args.repeat, kernel)
    if args.camera:
        camera_run([int(v) for v in args.bins.split(',') if v.strip()], args.frames, kernel)
    return 0


if __name__ == '__main__':
    sys.exit(main())
