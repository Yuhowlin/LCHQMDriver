"""``scripts/check_real_config.py`` stays runnable — the end-to-end rot alarm.

The script predated the roster threading of ``QMDeviceModel`` and crashed at
step [2/5] without anything noticing: no test imported it, and its manual runs
were piped through ``grep | tail``, whose exit code masked the failure. So this
asserts the FINAL PASS line and the exit code, never a fragment of the output.

One subprocess against the repo ``quam_state/`` (hermetic — the same folder the
live-state backend tests load), one qubit to bound the cost (~25 s): the value
here is the SHAPE of the pipeline (roster derivation -> device model -> Session
runs -> writeback -> save/reload), not target breadth. A failure prints the
script's own step log, which says which stage broke.
"""

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("quam_config")
pytest.importorskip("qm")

REPO = Path(__file__).resolve().parents[1]


def test_check_real_config_passes_against_the_live_quam_state():
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "check_real_config.py"),
         str(REPO / "quam_state"), "--qubits", "q1"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=600,
    )
    tail = proc.stdout[-4000:] + "\n--- stderr ---\n" + proc.stderr[-2000:]
    assert proc.returncode == 0, tail
    assert "\nPASS - scqo works against this real state" in proc.stdout, tail
