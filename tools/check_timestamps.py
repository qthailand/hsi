"""Check whether the GenICam camera gives per-frame timestamps and frame IDs.

The app would use them to drop frames taken before the stage starts moving.
The camera is used exactly as the app uses it: settings come from the app
config and ``open()`` / ``start_stream()`` run the app's usual tuning.  No stage
command is sent, and no camera node is written beyond that, except with
``--latch``, which runs the timestamp-latch command once.  The latch only copies
the camera clock into a register and does not change acquisition.

Reports:
* the buffer attributes found (timestamp, frame ID) and their values for the first frames;
* the camera timestamp rate against host time (1.0 means nanoseconds);
* the delivered frame rate from camera timestamps, and frame-ID gaps (lost frames);
* which timestamp-latch / tick-frequency nodes the camera has.

``--list-nodes`` instead lists every node whose name matches ``--match`` (default
time/stamp/latch) in all node maps (camera, GenTL device, interface, system), to find
a latch under a vendor-specific name.  It is read-only: the camera is not opened the
app's way (no tuning, no stream), no command is executed and no node is written;
only readable values are read.  Close the app first, it holds the camera.

Usage:
    python tools/check_timestamps.py
    python tools/check_timestamps.py --frames 200 --latch
    python tools/check_timestamps.py --list-nodes
    python tools/check_timestamps.py --list-nodes --match .      # every node
"""
import argparse
import os
import re
import statistics
import sys
import time

from harvesters.core import Harvester

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from config import load_app_config                    # noqa: E402  (project code, used read-only)
from modules.camera import create_backend             # noqa: E402

TIMESTAMP_ATTRIBUTES = ('timestamp_ns', 'timestamp', 'timestamp_us')
FRAME_ID_ATTRIBUTES = ('frame_id', 'frame_count', 'image_id')
LATCH_NODES = ('TimestampLatch', 'GevTimestampControlLatch', 'TimestampControlLatch')
LATCH_VALUE_NODES = ('TimestampLatchValue', 'GevTimestampValue', 'TimestampValue')
TICK_NODES = ('GevTimestampTickFrequency', 'TimestampTickFrequency')
#: Node maps listed by --list-nodes: label -> ImageAcquirer attribute.
NODE_MAP_OWNERS = (('camera', 'remote_device'), ('GenTL device', 'device'),
                   ('GenTL interface', 'interface'), ('GenTL system', 'system'))
#: GenApi EAccessMode values.
ACCESS_MODES = {0: 'NI', 1: 'NA', 2: 'WO', 3: 'RO', 4: 'RW'}
#: Node types whose value is read (commands are never executed, categories have none).
VALUE_TYPES = ('IInteger', 'IFloat', 'IString', 'IEnumeration', 'IBoolean')


def buffer_value(buffer, names):
    """First readable attribute among ``names`` on the harvesters buffer or its GenTL buffer."""
    for owner in (buffer, getattr(buffer, '_buffer', None)):
        if owner is None:
            continue
        for name in names:
            try:
                value = getattr(owner, name)
            except Exception:
                continue
            if value is not None:
                return name, value
    return None, None


def node_value(node_map, name):
    """``(present, value)`` of a node; the value is ``None`` for commands or unreadable nodes."""
    if not hasattr(node_map, name):
        return False, None
    try:
        return True, getattr(node_map, name).value
    except Exception:
        return True, None


def collect(camera, frames):
    """Fetch ``frames`` complete buffers: list of (host_ns, timestamp, frame_id) plus attribute names."""
    rows, names, incomplete = [], (None, None), 0
    while len(rows) < frames:
        with camera._ia.fetch(timeout=1.0) as buffer:         # harvesters takes seconds
            host_ns = time.perf_counter_ns()
            if getattr(buffer, 'is_incomplete', False):
                incomplete += 1
                if incomplete > 50:
                    raise RuntimeError('too many incomplete buffers')
                continue
            ts_name, ts = buffer_value(buffer, TIMESTAMP_ATTRIBUTES)
            id_name, frame_id = buffer_value(buffer, FRAME_ID_ATTRIBUTES)
            names = (ts_name, id_name)
            rows.append((host_ns, ts, frame_id))
    return rows, names, incomplete


def report(rows, names, incomplete):
    ts_name, id_name = names
    print(f'\nBuffer timestamp attribute: {ts_name or "NOT FOUND"}')
    print(f'Buffer frame-ID attribute:  {id_name or "NOT FOUND"}')
    print(f'Incomplete buffers skipped: {incomplete}')
    print('\nFirst frames (host ns since first, camera timestamp, frame ID):')
    host0 = rows[0][0]
    for host_ns, ts, frame_id in rows[:5]:
        print(f'  {host_ns - host0:>12d}  {ts!s:>22}  {frame_id!s:>10}')

    host_span = rows[-1][0] - rows[0][0]
    if ts_name is not None and all(isinstance(r[1], int) for r in rows):
        cam_span = rows[-1][1] - rows[0][1]
        print(f'\nCamera / host time ratio over {len(rows)} frames: {cam_span / host_span:.4f}'
              '   (about 1.0 = nanoseconds; otherwise divide by the tick frequency)')
        steps = [b[1] - a[1] for a, b in zip(rows, rows[1:])]
        print(f'Camera frame interval: median {statistics.median(steps):.0f}, '
              f'min {min(steps)}, max {max(steps)} (camera units)')
        if cam_span > 0:
            print(f'Delivered rate from camera time: {(len(rows) - 1) / cam_span * 1e9:.1f} fps (if ns)')
        if any(s <= 0 for s in steps):
            print('WARNING: timestamps are not strictly increasing')
    print(f'Delivered rate from host time:   {(len(rows) - 1) / host_span * 1e9:.1f} fps')

    if id_name is not None and all(isinstance(r[2], int) for r in rows):
        gaps = [b[2] - a[2] for a, b in zip(rows, rows[1:])]
        lost = sum(g - 1 for g in gaps if g > 1)
        print(f'Frame-ID gaps: {sum(g > 1 for g in gaps)} gaps, {lost} frames not received '
              f'(of {rows[-1][2] - rows[0][2] + 1} sent)')
        if any(g <= 0 for g in gaps):
            print('WARNING: frame IDs are not strictly increasing')


def check_nodes(node_map, latch):
    print('\nTimestamp nodes on the camera:')
    for name in TICK_NODES + LATCH_VALUE_NODES + LATCH_NODES:
        present, value = node_value(node_map, name)
        print(f'  {name:28s} {"present" if present else "-":8s} {"" if value is None else value}')
    if not latch:
        print('  (run with --latch to execute the latch command once and read the latched value)')
        return
    for command in LATCH_NODES:
        if hasattr(node_map, command):
            before = time.perf_counter_ns()
            getattr(node_map, command).execute()
            for name in LATCH_VALUE_NODES:
                present, value = node_value(node_map, name)
                if present:
                    print(f'  latched with {command}: {name} = {value} '
                          f'(host perf_counter_ns {before})')
                    return
            print(f'  {command} executed, but no latch value node was readable')
            return
    print('  no latch command node found')


def node_rows(node_map, pattern):
    """``(name, type, access, value, tooltip)`` of the nodes whose name matches ``pattern``.

    Only readable values of value nodes are read; commands are not executed.
    """
    rows = []
    for wrapper in node_map.nodes:
        node = wrapper.node
        if not re.search(pattern, node.name, re.IGNORECASE):
            continue
        kind = type(wrapper).__name__
        try:
            access = ACCESS_MODES.get(int(node.get_access_mode()), '?')
        except Exception:
            access = '?'
        value = ''
        if kind in VALUE_TYPES and access in ('RO', 'RW'):
            try:
                value = str(wrapper.value)
            except Exception as exc:
                value = f'<unreadable: {exc}>'
        rows.append((node.name, kind[1:], access, value, ' '.join(str(node.tooltip or '').split())))
    return rows


def list_nodes(acquirer, pattern):
    """Print the matching nodes of every node map the acquirer has."""
    for label, owner_name in NODE_MAP_OWNERS:
        try:
            node_map = getattr(acquirer, owner_name).node_map
            rows = node_rows(node_map, pattern)
        except Exception as exc:
            print(f'\n{label} node map: not available ({exc})')
            continue
        print(f'\n{label} node map: {len(rows)} nodes matching {pattern!r}')
        for name, kind, access, value, tooltip in rows:
            print(f'  {name:36s} {kind:12s} {access:3s} {value:>20s}  {tooltip[:70]}')


def run_list_nodes(camera, pattern):
    """Open the device only far enough to read its node maps (no tuning, no stream)."""
    harvester = Harvester()
    try:
        harvester.add_file(camera.CTI_PATH)
        harvester.update()
        if not harvester.device_info_list:
            print('No GenICam device found (is the app still running?)')
            return 1
        acquirer = harvester.create(camera.camera_index)
        try:
            list_nodes(acquirer, pattern)
        finally:
            acquirer.destroy()
    finally:
        harvester.reset()
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--frames', type=int, default=100, help='frames to collect')
    parser.add_argument('--latch', action='store_true', help='execute the timestamp-latch command once')
    parser.add_argument('--list-nodes', action='store_true',
                        help='only list matching nodes of all node maps (read-only)')
    parser.add_argument('--match', default='time|stamp|latch',
                        help='regular expression on node names for --list-nodes (case-insensitive)')
    args = parser.parse_args(argv)

    config = load_app_config()
    if str(config.camera_backend).lower() != 'genicam':
        print(f'GenICam only (config camera_backend={config.camera_backend})')
        return 1
    camera = create_backend(config)
    if args.list_nodes:
        return run_list_nodes(camera, args.match)
    print(f'Camera: genicam, bin {config.default_bin}, exposure {config.default_exposure_ms} ms, '
          f'gain {config.default_gain}; collecting {args.frames} frames')
    try:
        camera.open()
        camera.start_stream()
        try:
            rows, names, incomplete = collect(camera, args.frames)
        finally:
            camera.stop_stream()
        report(rows, names, incomplete)
        check_nodes(camera._nm, args.latch)
    finally:
        camera.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
