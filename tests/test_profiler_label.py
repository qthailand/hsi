"""The profile log line names the preprocessing backend and drops stale stages on a change."""
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.acquisition.profiler import StageProfiler


def test_label_in_log_and_stale_stages_cleared(caplog):
    backend = ['opencv_cuda']
    prof = StageProfiler(log_every=2, label=lambda: f'backend={backend[0]}')
    with caplog.at_level(logging.INFO, logger='modules.acquisition.profiler'):
        for _ in range(2):
            prof._record('gpu_preprocess', 15.0)
            prof.tick()
        backend[0] = 'cpu'                          # runtime fell back
        for _ in range(2):
            prof._record('orient_smile', 2.0)
            prof.tick()
    lines = [r.getMessage() for r in caplog.records]
    assert lines[0].startswith('[profile backend=opencv_cuda fps=') and 'gpu_preprocess' in lines[0]
    assert lines[1].startswith('[profile backend=cpu fps=')
    assert 'gpu_preprocess' not in lines[1] and 'orient_smile=2.00ms' in lines[1]


def test_without_label_keeps_old_format(caplog):
    prof = StageProfiler(log_every=1)
    with caplog.at_level(logging.INFO, logger='modules.acquisition.profiler'):
        prof._record('fetch', 1.0)
        prof.tick()
    message = caplog.records[0].getMessage()
    assert message.startswith('[profile fps=') and message.endswith('] fetch=1.00ms')


def test_detail_sampled_one_frame_in_ten_and_lines_do_not_carry_stale_stages(caplog):
    prof = StageProfiler(log_every=20, detail_every=10)
    detail = []
    with caplog.at_level(logging.INFO, logger='modules.acquisition.profiler'):
        for frame in range(40):
            detail.append(prof.want_detail())
            prof._record('fetch', 1.0)
            if frame < 20:
                prof._record('dark', 0.0)               # e.g. recorded only during dark capture
            prof.tick()
    assert sum(detail) == 4 and detail[0] and detail[10]
    first, second = (r.getMessage() for r in caplog.records)
    assert 'dark=0.00ms' in first and 'dark' not in second
    assert not StageProfiler(enabled=False).want_detail()


def test_fps_and_shown_rates_in_header(caplog, monkeypatch):
    import re
    from modules.acquisition import profiler as module
    clock = [100.0]
    monkeypatch.setattr(module.time, 'perf_counter', lambda: clock[0])
    prof = StageProfiler(log_every=10, label=lambda: 'backend=cupy_cuda cam=121.6Hz', shown_stage='emit')
    with caplog.at_level(logging.INFO, logger='modules.acquisition.profiler'):
        for frame in range(20):
            clock[0] += 0.01                                   # 100 loop frames per second
            prof._record('fetch', 5.0)
            if frame % 2 == 0:
                prof._record('emit', 0.1)                      # every other frame reaches the screen
            prof.tick()
    header = re.match(r'\[profile (.*?)\]', caplog.records[1].getMessage()).group(1)   # steady period
    assert header.startswith('backend=cupy_cuda cam=121.6Hz fps=')
    fps = float(re.search(r'fps=([0-9.]+)', header).group(1))
    shown = float(re.search(r'shown=([0-9.]+)', header).group(1))
    assert abs(fps - 100) < 0.5 and abs(shown - 50) < 0.5
