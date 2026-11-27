"""System sampler for the acceptance suite: CPU, GPU, RAM, temperatures and clocks while a test runs.

Sources, best available per platform (the report names the one used, or ``n/a``):

* CPU load and clocks, RAM, the test process's RSS and CPU: ``psutil`` (Windows and Linux);
* temperatures: ``psutil.sensors_temperatures`` or ``/sys/class/thermal`` (Linux / Jetson); on
  Windows usually none (reported as unavailable, not as zero);
* GPU: ``tegrastats`` (Jetson: ``GR3D_FREQ`` load and frequency, the SoC temperatures), else
  the Jetson sysfs GPU load / frequency, else ``nvidia-smi`` (desktop NVIDIA), else none.

Threading:
    :class:`SystemMonitor` samples on its own daemon thread between ``start`` and ``stop``.
"""
from __future__ import annotations

import glob
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    import psutil
except ImportError:                                            # reported, the suite still runs
    psutil = None

_GPU_LOAD_GLOBS = ("/sys/devices/gpu.0/load", "/sys/devices/platform/gpu.0/load",
                   "/sys/devices/platform/*.gpu/load", "/sys/devices/platform/bus@0/*.gpu/load",
                   "/sys/devices/platform/*ga10b*/load", "/sys/devices/platform/bus@0/*ga10b*/load")
_GPU_FREQ_GLOBS = ("/sys/devices/gpu.0/devfreq/*/cur_freq", "/sys/devices/platform/*gpu*/devfreq/*/cur_freq",
                   "/sys/devices/platform/bus@0/*gpu*/devfreq/*/cur_freq",
                   "/sys/class/devfreq/*gpu*/cur_freq", "/sys/class/devfreq/*ga10b*/cur_freq")


def _first(patterns) -> Optional[str]:
    for pattern in patterns:
        found = sorted(glob.glob(pattern))
        if found:
            return found[0]
    return None


def _read_number(path: Optional[str]) -> Optional[float]:
    if not path:
        return None
    try:
        return float(Path(path).read_text().strip().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def thermal_zones() -> Dict[str, float]:
    """``{sensor: degC}`` from psutil or ``/sys/class/thermal`` (``{}`` when there is none)."""
    temps: Dict[str, float] = {}
    if psutil is not None and hasattr(psutil, "sensors_temperatures"):
        try:
            for name, entries in (psutil.sensors_temperatures() or {}).items():
                for i, entry in enumerate(entries):
                    if entry.current is not None:
                        temps[f"{name}{'' if len(entries) == 1 else i}"] = float(entry.current)
        except Exception:                                      # noqa: BLE001 - optional
            pass
    if not temps:
        for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            kind = (Path(zone) / "type")
            value = _read_number(str(Path(zone) / "temp"))
            if value is not None:
                try:
                    label = kind.read_text().strip()
                except OSError:
                    label = Path(zone).name
                temps[label] = value / 1000.0 if value > 1000 else value
    return temps


class _Tegrastats:
    """``tegrastats`` streamed in the background (Jetson); the newest parsed line."""

    def __init__(self, interval_ms: int):
        self.latest: Dict = {}
        self.proc = subprocess.Popen(["tegrastats", "--interval", str(int(interval_ms))], stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            parsed = {}
            gr3d = re.search(r"GR3D_FREQ (\d+)%(?:@\[?(\d+)\]?)?", line)
            if gr3d:
                parsed["gpu_load_pct"] = float(gr3d.group(1))
                if gr3d.group(2):
                    parsed["gpu_freq_mhz"] = float(gr3d.group(2))
            ram = re.search(r"RAM (\d+)/(\d+)MB", line)
            if ram:
                parsed["ram_used_mb"], parsed["ram_total_mb"] = float(ram.group(1)), float(ram.group(2))
            temps = {name: float(value) for name, value in re.findall(r"(\w+)@(-?[\d.]+)C", line)}
            if temps:
                parsed["temps"] = temps
            cpus = re.search(r"CPU \[([^\]]+)\]", line)
            if cpus:
                freqs = [float(f) for f in re.findall(r"@(\d+)", cpus.group(1))]
                if freqs:
                    parsed["cpu_freq_mhz_tegra"] = sum(freqs) / len(freqs)
            self.latest = parsed

    def stop(self):
        try:
            self.proc.terminate()
        except Exception:                                      # noqa: BLE001
            pass


class SystemMonitor:
    """Samples the system every ``interval_s`` (see the module docstring).

    Args:
        interval_s: Sampling period.
        pid: The test process to follow (RSS, CPU %), or ``None``.
    """

    def __init__(self, interval_s: float = 1.0, pid: Optional[int] = None):
        self.interval_s = float(interval_s)
        self.pid = pid
        self.samples: List[Dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tegra: Optional[_Tegrastats] = None
        self._gpu_load = _first(_GPU_LOAD_GLOBS)
        self._gpu_freq = _first(_GPU_FREQ_GLOBS)
        self._nvidia_smi = shutil.which("nvidia-smi")
        self.sources = {"cpu": "psutil" if psutil else "n/a", "gpu": "n/a", "temperature": "n/a"}

    def start(self, pid: Optional[int] = None) -> "SystemMonitor":
        """Start sampling (``pid``: the process to follow)."""
        self.pid = pid if pid is not None else self.pid
        if shutil.which("tegrastats"):
            try:
                self._tegra = _Tegrastats(int(self.interval_s * 1000))
                self.sources["gpu"] = "tegrastats"
            except OSError:
                self._tegra = None
        if self.sources["gpu"] == "n/a":
            self.sources["gpu"] = ("jetson sysfs" if self._gpu_load else "nvidia-smi" if self._nvidia_smi else "n/a")
        if psutil is not None:
            psutil.cpu_percent(None)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _process(self):
        if psutil is None or self.pid is None:
            return None
        try:
            return psutil.Process(self.pid)
        except Exception:                                      # noqa: BLE001 - it ended
            return None

    def _run(self):
        process = self._process()
        if process is not None:
            try:
                process.cpu_percent(None)
            except Exception:                                  # noqa: BLE001
                process = None
        while not self._stop.wait(self.interval_s):
            self.samples.append(self._sample(process))

    def _sample(self, process) -> Dict:
        s: Dict = {"t": time.time()}
        if psutil is not None:
            s["cpu_pct"] = psutil.cpu_percent(None)
            freq = psutil.cpu_freq()
            if freq is not None:
                s["cpu_freq_mhz"] = float(freq.current)
                s["cpu_freq_max_mhz"] = float(freq.max or 0) or None
            vm = psutil.virtual_memory()
            s["ram_used_mb"], s["ram_pct"] = (vm.total - vm.available) / 2**20, vm.percent
            if process is not None:
                try:
                    s["proc_rss_mb"] = process.memory_info().rss / 2**20
                    s["proc_cpu_pct"] = process.cpu_percent(None)
                except Exception:                              # noqa: BLE001 - it ended
                    pass
        temps = thermal_zones()
        if self._tegra is not None and self._tegra.latest:
            latest = self._tegra.latest
            for key in ("gpu_load_pct", "gpu_freq_mhz"):
                if key in latest:
                    s[key] = latest[key]
            temps = {**temps, **latest.get("temps", {})}
        elif self._gpu_load:
            load = _read_number(self._gpu_load)
            if load is not None:
                s["gpu_load_pct"] = load / 10.0                # Jetson sysfs: 0..1000
            freq = _read_number(self._gpu_freq)
            if freq is not None:
                s["gpu_freq_mhz"] = freq / 1e6
        elif self._nvidia_smi:
            try:
                out = subprocess.run([self._nvidia_smi, "--query-gpu=utilization.gpu,clocks.gr,temperature.gpu",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
                load, clock, temp = (float(v) for v in out.stdout.strip().splitlines()[0].split(","))
                s.update(gpu_load_pct=load, gpu_freq_mhz=clock)
                temps["gpu"] = temp
            except Exception:                                  # noqa: BLE001 - optional
                pass
        if temps:
            s["temps"] = temps
            self.sources["temperature"] = "tegrastats" if self._tegra is not None else "sensors"
        return s

    def stop(self) -> Dict:
        """Stop sampling; returns :meth:`summary`."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.interval_s * 3 + 1)
        if self._tegra is not None:
            self._tegra.stop()
        return self.summary()

    def summary(self) -> Dict:
        """Mean / max of every sampled quantity, the sources, and the samples."""
        out: Dict = {"sources": dict(self.sources), "samples": len(self.samples)}
        keys = sorted({k for s in self.samples for k in s if k not in ("t", "temps")})
        for key in keys:
            values = [s[key] for s in self.samples if s.get(key) is not None]
            if values:
                out[f"{key}_mean"] = sum(values) / len(values)
                out[f"{key}_max"] = max(values)
        names = sorted({n for s in self.samples for n in s.get("temps", {})})
        out["temps_max"] = {n: max(s["temps"][n] for s in self.samples if n in s.get("temps", {})) for n in names}
        rss = [s["proc_rss_mb"] for s in self.samples if s.get("proc_rss_mb") is not None]
        if len(rss) >= 2:
            out["proc_rss_growth_mb"] = rss[-1] - rss[0]
        out["series"] = self.samples
        return out


__all__ = ["SystemMonitor", "thermal_zones"]
