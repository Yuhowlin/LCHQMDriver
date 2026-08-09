"""The session door (`_lib.qm_job`) and its failure surface.

Every trap here was OBSERVED on 5Q4C before the door existed (run-index rows,
2026-08-04 .. 2026-08-09) and misattributed to active reset, which the same
index later exonerated (thermal runs died identically; the active-reset ramsey
passed at 2026-08-09 23:13 once the gateway was cleared):

* vendor ``qm_session`` re-raised non-busy open failures as a BARE
  ``Exception`` — run records showed ``Exception: `` (empty) while the real
  gateway text (``DEADLINE_EXCEEDED``) sat unprinted in ``__cause__``;
* nothing bounded execute/fetch — a zombie job hung the CLI forever;
* an operator kill left the RUNNING job holding the QOP, so following runs
  died at the QOP-free timeout.

The fakes stub the vendor surface only (qmm/qm/job/results); everything under
test is the real ``scqo_qm.experiments._lib`` code.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import scqo_qm.experiments._lib as _lib


# --------------------------------------------------------------------------
# fakes: the narrowest vendor surface qm_job touches
# --------------------------------------------------------------------------

BUSY_TEXT = "Resources already locked"  # the OPX1000 multi-user busy signature
WEDGE_TEXT = "PHYSICAL CONFIG ERROR gateway gave up: DEADLINE_EXCEEDED"


class FakeJob:
    def __init__(self):
        self.halted = False

    def halt(self):
        self.halted = True


class FakeQM:
    def __init__(self, execute=None):
        self.closed = False
        self.executed = []
        self._execute = execute

    def execute(self, prog):
        self.executed.append(prog)
        if self._execute is not None:
            return self._execute(prog)
        return FakeJob()

    def close(self):
        self.closed = True


class FakeQMM:
    """open_qm behavior is a callable so each test scripts its own gateway."""

    def __init__(self, open_qm):
        self._open_qm = open_qm
        self.open_calls = 0

    def open_qm(self, config, close_other_machines):
        assert close_other_machines is False, "the door must never steal the QOP"
        self.open_calls += 1
        return self._open_qm(self.open_calls)

    def list_open_quantum_machines(self):
        return ["qm-stale-1"]


class FakeMachine:
    network = {"host": "10.9.9.9"}

    def __init__(self, qmm):
        self._qmm = qmm

    def connect(self):
        return self._qmm

    def generate_config(self):
        return {}


def fast_polls(monkeypatch, *, qop_poll=0.01, execute_budget=0.25, stall=0.05):
    monkeypatch.setattr(_lib, "_QOP_FREE_POLL_S", qop_poll)
    monkeypatch.setattr(_lib, "_EXECUTE_BUDGET_S", execute_budget)
    monkeypatch.setattr(_lib, "_STALL_BUDGET_S", stall)


# --------------------------------------------------------------------------
# the census: one door, no vendor session helper anywhere in the package
# --------------------------------------------------------------------------

def test_no_module_imports_the_vendor_session_helper():
    """``qualang_tools.multi_user`` appears NOWHERE under scqo_qm/ — its
    ``qm_session`` is the bare-Exception / swallowed-KeyboardInterrupt trap
    this module replaces, and a future experiment must not quietly reopen the
    hole. (Text scan, same style as test_reset_method's .reset-literal scan.)"""
    import re

    pkg = Path(_lib.__file__).resolve().parents[1]
    importing = re.compile(r"^\s*(from|import)\s+qualang_tools\.multi_user\b")
    offenders = [
        str(p.relative_to(pkg))
        for p in pkg.rglob("*.py")
        if any(importing.match(line) for line in
               p.read_text(encoding="utf-8").splitlines())
    ]
    assert offenders == [], (
        f"qualang_tools.multi_user imported outside the _lib door: {offenders}"
    )


# --------------------------------------------------------------------------
# open: multi-user etiquette kept, everything else loud
# --------------------------------------------------------------------------

def test_busy_then_free_opens_and_cleans_up(monkeypatch):
    fast_polls(monkeypatch)
    qm = FakeQM()

    def gateway(call):
        if call < 3:
            raise Exception(BUSY_TEXT)
        return qm

    machine = FakeMachine(FakeQMM(gateway))
    with _lib.qm_job(machine, "prog", timeout=5.0) as job:
        assert isinstance(job, FakeJob)
    assert qm.executed == ["prog"]
    assert job.halted, "halt-on-exit is unconditional (no-op on a finished job)"
    assert qm.closed


def test_busy_timeout_names_holders_and_playbook(monkeypatch):
    fast_polls(monkeypatch)
    machine = FakeMachine(FakeQMM(lambda call: (_ for _ in ()).throw(Exception(BUSY_TEXT))))
    with pytest.raises(TimeoutError) as err:
        with _lib.qm_job(machine, "prog", timeout=0.05):
            pass
    text = str(err.value)
    assert "qm-stale-1" in text, "the operator must see WHO holds the QOP"
    assert "http://10.9.9.9:9504/cluster" in text


def test_open_failure_surfaces_the_buried_gateway_text(monkeypatch):
    """The regression for the empty ``Exception:`` run records: a bare
    exception whose real text hides in __cause__ (exactly what the vendor
    helper manufactured) must come out with the cause text AND the wedge
    playbook in the message."""
    fast_polls(monkeypatch)

    def gateway(call):
        try:
            raise RuntimeError(WEDGE_TEXT)
        except RuntimeError as cause:
            raise Exception() from cause  # noqa: TRY002 — the vendor's exact shape

    machine = FakeMachine(FakeQMM(gateway))
    with pytest.raises(RuntimeError) as err:
        with _lib.qm_job(machine, "prog", timeout=1.0):
            pass
    text = str(err.value)
    assert "DEADLINE_EXCEEDED" in text
    assert "http://10.9.9.9:9504/cluster" in text


# --------------------------------------------------------------------------
# execute: bounded, and errors keep their words
# --------------------------------------------------------------------------

def test_execute_stall_dies_bounded_and_closes(monkeypatch):
    fast_polls(monkeypatch)
    qm = FakeQM(execute=lambda prog: time.sleep(30))
    machine = FakeMachine(FakeQMM(lambda call: qm))
    t0 = time.monotonic()
    with pytest.raises(TimeoutError) as err:
        with _lib.qm_job(machine, "prog", timeout=1.0):
            pass
    assert time.monotonic() - t0 < 5.0, "must not wait out the fake's 30s sleep"
    assert "gateway compile/start stall" in str(err.value)
    assert qm.closed


def test_execute_error_carries_wedge_hint(monkeypatch):
    fast_polls(monkeypatch)

    def boom(prog):
        raise RuntimeError(WEDGE_TEXT)

    qm = FakeQM(execute=boom)
    machine = FakeMachine(FakeQMM(lambda call: qm))
    with pytest.raises(RuntimeError) as err:
        with _lib.qm_job(machine, "prog", timeout=1.0):
            pass
    assert "DEADLINE_EXCEEDED" in str(err.value)
    assert "http://10.9.9.9:9504/cluster" in str(err.value)
    assert qm.closed


# --------------------------------------------------------------------------
# unwind: the job is halted on EVERY exit, including an operator kill
# --------------------------------------------------------------------------

def test_operator_kill_halts_job_then_propagates(monkeypatch):
    """THE stale-QM regression: Ctrl+C during the fetch loop must halt the
    running job (freeing the QOP even when close() is ignored) and then
    PROPAGATE — the vendor helper swallowed it and left the job running,
    which is why the next runs died at the QOP-free timeout."""
    fast_polls(monkeypatch)
    qm = FakeQM()
    machine = FakeMachine(FakeQMM(lambda call: qm))
    with pytest.raises(KeyboardInterrupt):
        with _lib.qm_job(machine, "prog", timeout=1.0) as job:
            raise KeyboardInterrupt
    assert job.halted
    assert qm.closed


def test_body_error_halts_job(monkeypatch):
    fast_polls(monkeypatch)
    qm = FakeQM()
    machine = FakeMachine(FakeQMM(lambda call: qm))
    with pytest.raises(ValueError):
        with _lib.qm_job(machine, "prog", timeout=1.0) as job:
            raise ValueError("fetch went wrong")
    assert job.halted
    assert qm.closed


# --------------------------------------------------------------------------
# the fetch watchdog and the record-only wait
# --------------------------------------------------------------------------

class FrozenFetcher:
    """A fetcher whose shot counter never advances: the zombie job."""

    t_start = 0.0

    def __init__(self, job, axes):
        self.dataset = object()

    def __iter__(self):
        while True:
            time.sleep(0.005)
            yield self.dataset

    def get(self, key, default=None):
        return 3 if key == "n" else default


def test_acquire_watchdog_halts_a_silent_job(monkeypatch):
    fast_polls(monkeypatch)
    monkeypatch.setattr(_lib, "XarrayDataFetcher", FrozenFetcher)
    monkeypatch.setattr(_lib, "progress_counter", lambda *a, **k: None)
    qm = FakeQM()
    machine = FakeMachine(FakeQMM(lambda call: qm))
    with pytest.raises(TimeoutError) as err:
        _lib.acquire(machine, "prog", {}, num_shots=10, timeout=1.0)
    assert "stuck at shot 3/10" in str(err.value)
    assert qm.closed


class FakeResults:
    def __init__(self, behavior):
        self._behavior = behavior

    def wait_for_all_values(self, timeout=None):
        assert timeout is not None, "the record-only wait must be bounded"
        return self._behavior()


def test_wait_all_streams_wraps_timeout(monkeypatch):
    machine = FakeMachine(FakeQMM(lambda call: FakeQM()))

    def never_done():
        raise TimeoutError("Job was not done in time")

    with pytest.raises(TimeoutError) as err:
        _lib.wait_all_streams(machine, FakeResults(never_done))
    assert "zombie-job stall" in str(err.value)
    assert "http://10.9.9.9:9504/cluster" in str(err.value)


def test_wait_all_streams_names_a_dead_job():
    machine = FakeMachine(FakeQMM(lambda call: FakeQM()))
    with pytest.raises(RuntimeError, match="failed or was cancelled"):
        _lib.wait_all_streams(machine, FakeResults(lambda: False))


def test_wait_all_streams_passes_a_finished_job():
    machine = FakeMachine(FakeQMM(lambda call: FakeQM()))
    assert _lib.wait_all_streams(machine, FakeResults(lambda: True)) is None
