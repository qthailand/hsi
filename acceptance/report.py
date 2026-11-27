"""Acceptance report: ``report.json`` (everything), ``summary.csv`` (one row per test), ``summary.html``."""
from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Dict, List

from common import FAIL, PASS, SKIPPED, write_json

CSV_METRICS = ("count_mean_ms", "count_p95_ms", "count_p99_ms", "count_max_ms", "fps", "fps_off", "fps_ratio",
               "dropped_frames", "frame_latency_mean_ms", "frame_latency_max_ms", "cpu_pct_mean", "proc_cpu_pct_mean",
               "gpu_load_pct_mean", "ram_used_mb_max", "proc_rss_mb_max", "proc_rss_growth_mb", "temp_max_c",
               "cpu_freq_mhz_mean", "gpu_freq_mhz_mean", "frames_compared", "frames_identical")


def write_reports(folder: Path, report: Dict) -> Dict[str, Path]:
    """Write the three files into ``folder``; returns their paths."""
    folder = Path(folder)
    paths = {"json": folder / "report.json", "csv": folder / "summary.csv", "html": folder / "summary.html"}
    write_json(paths["json"], report)
    with open(paths["csv"], "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["test_id", "category", "status", "reason", "duration_s", *CSV_METRICS])
        for t in report["tests"]:
            m = t.get("metrics", {})
            writer.writerow([t["test_id"], t["category"], t["status"], t["reason"], f"{t.get('duration_s', 0):.1f}",
                             *[_fmt(m.get(k)) for k in CSV_METRICS]])
    paths["html"].write_text(_html(report), encoding="utf-8")
    return paths


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return value


def _table(rows: List[List], head: List[str], classes=None) -> str:
    out = ["<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in head) + "</tr></thead><tbody>"]
    for i, row in enumerate(rows):
        cls = f' class="{classes[i]}"' if classes and classes[i] else ""
        out.append(f"<tr{cls}>" + "".join(f"<td>{html.escape(str(_fmt(c)))}</td>" for c in row) + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def _kv(data: Dict) -> str:
    return _table([[k, json.dumps(v, default=str) if isinstance(v, (dict, list)) else v] for k, v in data.items()],
                  ["", ""])


def _html(report: Dict) -> str:
    tests = report["tests"]
    counts = {s: sum(1 for t in tests if t["status"] == s) for s in (PASS, FAIL, SKIPPED)}
    verdict = report["verdict"]
    backend = report.get("backend", {})
    banner = ""
    if backend.get("cpu_fallback"):
        banner = (f'<div class="warn">Processing backend: requested <b>{html.escape(str(backend.get("requested")))}'
                  f'</b>, ran on <b>{html.escape(str(backend.get("used")))}</b> (CPU fallback)'
                  f'{" - " + html.escape(backend.get("reason") or "") if backend.get("reason") else ""}</div>')
    test_rows = [[t["test_id"], t["category"], t["status"], t["reason"],
                  ", ".join(f"{k}={_fmt(v)}" for k, v in t.get("metrics", {}).items() if v is not None)]
                 for t in tests]
    classes = [t["status"].lower() for t in tests]
    readiness = report.get("profile_readiness", {})
    ready_rows = [[k, v["state"], v["reason"], json.dumps(v.get("revision") or {}, default=str)]
                  for k, v in readiness.items()]
    ready_classes = ["pass" if v["state"] == "READY" else ("skipped" if v["state"] == "NOT USED" else "fail")
                     for v in readiness.values()]
    integrity = report.get("integrity", {})
    style = """
    :root{--bg:#fff;--fg:#1d2228;--muted:#5b6672;--line:#d8dde3;--pass:#e5f6e8;--fail:#fde7e6;--skip:#f1f2f4;
          --warn:#fff4d6;--passfg:#1b7f37;--failfg:#b3261e}
    @media (prefers-color-scheme: dark){:root{--bg:#15181c;--fg:#e6e9ee;--muted:#9aa4b2;--line:#2c3138;
          --pass:#12301b;--fail:#3a1614;--skip:#20242a;--warn:#3a2f12;--passfg:#56d364;--failfg:#ff7b72}}
    body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,Segoe UI,sans-serif;margin:24px;max-width:1300px}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:16px;margin:28px 0 8px;border-bottom:1px solid var(--line)}
    .muted{color:var(--muted)} table{border-collapse:collapse;width:100%;margin:6px 0}
    th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top;font-size:13px}
    tr.pass td{background:var(--pass)} tr.fail td{background:var(--fail)} tr.skipped td{background:var(--skip)}
    .verdict{font-size:18px;font-weight:600;padding:8px 12px;border-radius:6px;display:inline-block;margin:8px 0}
    .verdict.PASS{background:var(--pass);color:var(--passfg)} .verdict.FAIL{background:var(--fail);color:var(--failfg)}
    .warn{background:var(--warn);padding:8px 12px;border-radius:6px;margin:8px 0}
    """
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
             f"<title>HSI acceptance</title><style>{style}</style></head><body>",
             f"<h1>HSI acceptance report</h1><div class='muted'>{html.escape(report['started_at'])} - "
             f"{html.escape(report['platform'].get('node', ''))} ({html.escape(report['platform'].get('system', ''))} "
             f"{html.escape(report['platform'].get('machine', ''))}) - mode {html.escape(report['options']['mode'])}, "
             f"camera {html.escape(report['options']['camera'])}</div>",
             f"<div class='verdict {verdict}'>{verdict}: {counts[PASS]} passed, {counts[FAIL]} failed, "
             f"{counts[SKIPPED]} skipped</div>", banner,
             "<h2>Tests</h2>", _table(test_rows, ["test", "category", "status", "reason", "metrics"], classes),
             "<h2>Profile readiness</h2>", _table(ready_rows, ["profile", "state", "reason", "revision"], ready_classes),
             "<h2>Integrity</h2>", _kv(integrity),
             "<h2>Camera</h2>", _kv(report.get("camera_identity", {})),
             "<h2>Configuration</h2>", _kv(report.get("configuration", {})),
             "<h2>Software</h2>", _kv(report.get("software", {})),
             "<h2>Platform</h2>", _kv(report.get("platform", {})),
             "</body></html>"]
    return "\n".join(parts)


__all__ = ["write_reports"]
