"""Crash log for the test suite: Qt critical/fatal messages and faulthandler dumps go to a file.

A Qt ``qFatal`` or a native crash ends the process with exit code 3 (abort) before pytest can
report anything, and pytest's output capture swallows the message. The log file survives that;
its path is printed in the pytest header.
"""
import faulthandler
import os
import sys
import tempfile

from PyQt5.QtCore import QtCriticalMsg, QtFatalMsg, qInstallMessageHandler

CRASH_LOG = os.path.join(tempfile.gettempdir(), "hsi_pytest_crash.log")
_log = None
_current = "(no test running)"


def _qt_message(mode, context, message):
    """Print like Qt's default handler; critical and fatal messages also go to the crash log."""
    sys.stderr.write(message + "\n")
    if mode in (QtCriticalMsg, QtFatalMsg):
        _log.write(f"[{_current}] Qt {'fatal' if mode == QtFatalMsg else 'critical'}: {message}\n")
        _log.flush()
        os.fsync(_log.fileno())


def pytest_configure(config):
    """Open the crash log, point faulthandler at it and install the Qt message handler."""
    global _log
    _log = open(CRASH_LOG, "w", encoding="utf-8")
    faulthandler.enable(file=_log, all_threads=True)
    qInstallMessageHandler(_qt_message)


def pytest_report_header(config):
    return f"crash log (Qt fatal messages, faulthandler dumps): {CRASH_LOG}"


def pytest_runtest_setup(item):
    """Remember the running test; re-point faulthandler (modules/camera/genicam.py re-enables it on stderr)."""
    global _current
    _current = item.nodeid
    faulthandler.enable(file=_log, all_threads=True)
