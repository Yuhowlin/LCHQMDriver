"""Shared plumbing for probes: target selection and the execute-and-fetch half.
No qualibrate imports.

Flux amplitude/rail validation lives in ``_flux_limits.py`` — a probe asking
"may this port emit these volts?" imports from there, not here.

THE SESSION DOOR (``qm_job``). Every hardware execution in this driver — the
shared ``acquire`` below and the three record-only fetchers (tomography, the
two T1 trackers) — reaches the QOP through ``qm_job``, never through
``qualang_tools.multi_user.qm_session`` (``tests/test_acquire_guard.py`` scans
for the import). The vendor helper has three defects, each observed on 5Q4C
(2026-08-04 .. 2026-08-09; the run-index rows are the evidence) and each one a
debugging trap:

1. Any ``open_qm`` failure that is NOT the multi-user busy signature is
   re-raised as ``raise Exception from e`` — a BARE exception whose message is
   EMPTY, with the real gateway error (``DEADLINE_EXCEEDED``, the OPX1000
   wedge signature) hidden in ``__cause__`` where no run record prints it.
   Three 5Q4C qubit_ramsey runs died as ``Exception: `` on 2026-08-04 and were
   misread as an active-reset bug; the reset was exonerated by the run index
   (thermal single_shot_readout died identically at 20:57 the same evening).
2. Nothing bounds execute-and-fetch: a job the gateway accepts but never
   drives spins the fetch loop FOREVER. The 2026-08-09 22:45 qubit_ramsey
   hang — killed by hand after many minutes, leaving an empty run folder —
   was this.
3. An operator kill (KeyboardInterrupt) is swallowed and the still-RUNNING job
   is left holding the QOP: ``close()`` on a wedged gateway is silently
   ignored, so every following run waits the full QOP-free timeout and dies
   (``TimeoutError ... 120s``: 2026-08-05 11:31/11:34/13:50, 2026-08-09
   23:03). ``qm_job`` HALTS its job on every exit path before closing — a
   halted job frees the QOP even when the close itself is ignored.

The multi-user etiquette is preserved: ``open_qm(close_other_machines=False)``
polled until the QOP frees, bounded by the caller's ``timeout`` — this door
never steals the instrument from a live run, it only refuses to hang on a dead
one.
"""

import logging
import threading
import time
from contextlib import contextmanager
from typing import Callable, List, Optional

import xarray as xr
from qualang_tools.results import progress_counter
from qualibration_libs.core import BatchableList
from qualibration_libs.data import XarrayDataFetcher

_log = logging.getLogger(__name__)

#: The two "QOP is legitimately busy" signatures ``open_qm`` raises, matched
#: version-agnostically (OPX1000 / legacy wording — same strings the vendor
#: ``qm_session`` matches, minus its capability-flag bug).
_QOP_BUSY_OPX1000 = "Resources already locked"
_QOP_BUSY_LEGACY = "cannot be used because it isn't shareable in other QM."

#: Substrings that identify the OPX1000 gateway wedge in a vendor error text.
_WEDGE_MARKERS = ("DEADLINE_EXCEEDED", "PHYSICAL CONFIG ERROR")

#: Gateway compile-and-start budget. The gateway applies its own ~30 s
#: deadline when it still answers at all; a blocked ``execute()`` beyond this
#: is the wedge, not a slow compile.
_EXECUTE_BUDGET_S = 180.0

#: Fetch-loop no-progress watchdog: with no new shot for this long the job is
#: declared a zombie and halted. Every ``acquire`` program saves the shot
#: counter ``n`` once per shot, and the slowest legitimate shot in the roster
#: is orders of magnitude shorter than this.
_STALL_BUDGET_S = 300.0

#: Cap on ``wait_for_all_values`` in the record-only fetchers (tomography, T1
#: trackers): their streams arrive only when the job ENDS, so progress cannot
#: be watched — a generous absolute ceiling stands in (longest real tracker
#: run to date: ~34 s).
_RECORD_WAIT_BUDGET_S = 3600.0

_QOP_FREE_POLL_S = 0.5

#: The OPX1000 cluster ADMIN panel port (the gRPC gateway itself is on the
#: port wiring.json names; the admin page where stale QMs are cleared sits on
#: 9504 across the lab's clusters).
_ADMIN_PANEL_PORT = 9504


def select_qubits(machine, names: Optional[List[str]] = None, *, multiplexed: bool = False) -> BatchableList:
    """Node-free replacement for `qualibration_libs.parameters.get_qubits(node)`.

    Selects qubits from the machine by name (or `machine.active_qubits` when
    `names` is None/empty) and wraps them in the same `BatchableList` the
    qualibrate helper produces, so probes can iterate `qubits.batch()` /
    `qubits.get_names()` identically in both shells.
    """
    if not names:
        qubits = machine.active_qubits
    else:
        qubits = [machine.qubits[q] for q in names]
    if multiplexed:
        batched_groups = [list(range(len(qubits)))]
    else:
        batched_groups = [[i] for i in range(len(qubits))]
    return BatchableList(qubits, batched_groups)


def select_qubit_pairs(machine, names: Optional[List[str]] = None, *,
                       multiplexed: bool = False) -> BatchableList:
    """Node-free replacement for `qualibration_libs.parameters.get_qubit_pairs(node)`:
    selects pairs from the machine by name (or `machine.active_qubit_pairs` when
    `names` is None/empty) and wraps them in the same `BatchableList`."""
    if not names:
        pairs = machine.active_qubit_pairs
    else:
        pairs = [machine.qubit_pairs[p] for p in names]
    if multiplexed:
        batched_groups = [list(range(len(pairs)))]
    else:
        batched_groups = [[i] for i in range(len(pairs))]
    return BatchableList(pairs, batched_groups)


def _error_chain_text(exc: BaseException) -> str:
    """The exception and its causes as one line, so a vendor ``raise Exception
    from e`` can never hide the gateway's actual words again."""
    parts: List[str] = []
    seen = set()
    e: Optional[BaseException] = exc
    while e is not None and id(e) not in seen and len(parts) < 6:
        seen.add(id(e))
        text = str(e).strip()
        parts.append(f"{type(e).__name__}: {text}" if text else type(e).__name__)
        e = e.__cause__ if e.__cause__ is not None else e.__context__
    return " <- ".join(parts)


def _is_wedge_text(text: str) -> bool:
    return any(marker in text for marker in _WEDGE_MARKERS)


def cluster_hint(machine) -> str:
    """The operator playbook line for a wedged gateway, with the admin URL
    derived from the machine's network block when it has one."""
    host = None
    network = getattr(machine, "network", None)
    if network is not None:
        try:
            host = network.get("host") if hasattr(network, "get") else getattr(network, "host", None)
        except Exception:
            host = None
    where = (f"http://{host}:{_ADMIN_PANEL_PORT}/cluster" if host
             else f"the cluster admin page (http://<gateway-host>:{_ADMIN_PANEL_PORT}/cluster)")
    return (
        "If no other run is genuinely live, this is the OPX1000 stale-QM gateway wedge "
        "(a dead run's job still holds resources and close() is silently ignored) — "
        f"clear/restart the cluster at {where}, then re-run."
    )


def _open_machine_ids(qmm) -> str:
    # list_open_qms is the 1.2+ name; the pre-1.2 spelling is the fallback.
    lister = getattr(qmm, "list_open_qms", None) or getattr(qmm, "list_open_quantum_machines", None)
    if lister is None:
        return "unavailable"
    try:
        ids = lister()
        return ", ".join(str(i) for i in ids) if ids else "none reported"
    except Exception:
        return "unavailable"


def _open_qop(qmm, config, *, timeout: float, hint: str):
    """``open_qm(close_other_machines=False)`` with the multi-user busy-wait,
    but loud on everything else: the real vendor error text survives, the
    wedge signature is named, and a busy timeout reports who holds the QOP."""
    qm_logger = logging.getLogger("qm")
    saved_level = qm_logger.level
    t_start = time.monotonic()
    warned = False
    try:
        while True:
            try:
                return qmm.open_qm(config, close_other_machines=False)
            except Exception as exc:
                text = _error_chain_text(exc)
                busy = (_QOP_BUSY_OPX1000 in text) or (_QOP_BUSY_LEGACY in text)
                if not busy:
                    wedge = _is_wedge_text(text)
                    raise RuntimeError(
                        f"open_qm failed: {text}"
                        + (f" — the OPX1000 gateway-wedge signature. {hint}" if wedge else "")
                    ) from exc
                if time.monotonic() - t_start >= timeout:
                    raise TimeoutError(
                        f"QOP still busy after {timeout:.0f}s "
                        f"(open quantum machines: {_open_machine_ids(qmm)}). "
                        f"Another operator's run may genuinely hold it — check before clearing. {hint}"
                    ) from exc
                if not warned:
                    _log.warning("QOP busy — waiting up to %.0fs for it to free ...", timeout)
                    # open_qm retries log a vendor ERROR line each poll; hush
                    # the qm logger for the wait only (restored in finally).
                    qm_logger.setLevel(logging.CRITICAL)
                    warned = True
                time.sleep(_QOP_FREE_POLL_S)
    finally:
        qm_logger.setLevel(saved_level)


def _execute_bounded(qm, prog, *, hint: str):
    """``qm.execute`` in a worker thread with a wall-clock budget: a gateway
    that accepts the connection but never returns from compile-and-start is
    reported as the wedge instead of hanging the CLI forever."""
    box: dict = {}

    def _run() -> None:
        try:
            box["job"] = qm.execute(prog)
        except BaseException as exc:  # noqa: BLE001 — carried to the main thread
            box["error"] = exc

    worker = threading.Thread(target=_run, name="scqo-qm-execute", daemon=True)
    worker.start()
    worker.join(_EXECUTE_BUDGET_S)
    if worker.is_alive():
        raise TimeoutError(
            f"qm.execute() still blocked after {_EXECUTE_BUDGET_S:.0f}s — "
            f"gateway compile/start stall. {hint}"
        )
    if "error" in box:
        exc = box["error"]
        if isinstance(exc, KeyboardInterrupt):
            raise exc
        text = _error_chain_text(exc)
        raise RuntimeError(
            f"qm.execute() failed: {text}"
            + (f" — the OPX1000 gateway-wedge signature. {hint}" if _is_wedge_text(text) else "")
        ) from exc
    return box["job"]


def _halt_quietly(job) -> None:
    """Best-effort job halt. Halting a finished job is a no-op; halting a
    zombie is what actually frees the QOP when close() is being ignored."""
    if job is None:
        return
    for name in ("halt", "cancel"):
        fn = getattr(job, name, None)
        if fn is None:
            continue
        try:
            fn()
        except Exception as exc:
            _log.debug("job %s() raised while cleaning up: %s", name, exc)
        return


def _close_quietly(qm) -> None:
    try:
        qm.close()
    except Exception as exc:
        _log.warning("qm.close() failed (gateway may be wedged): %s",
                     _error_chain_text(exc))


@contextmanager
def qm_job(machine, prog, *, timeout: float, config: Optional[dict] = None):
    """Connect, open the QOP politely, execute ``prog`` and yield the running
    job; on EVERY exit — success, error, or operator kill — halt the job, then
    close the QM. ``timeout`` bounds only the wait for the QOP to free, the
    same meaning it has always had."""
    hint = cluster_hint(machine)
    qmm = machine.connect()
    config = config if config is not None else machine.generate_config()
    qm = _open_qop(qmm, config, timeout=timeout, hint=hint)
    job = None
    try:
        job = _execute_bounded(qm, prog, hint=hint)
        yield job
    finally:
        _halt_quietly(job)
        _close_quietly(qm)


def wait_all_streams(machine, results) -> None:
    """Bounded stand-in for ``results.wait_for_all_values()`` in the
    record-only fetchers, whose streams complete only when the job ends: a
    zombie job surfaces as a named TimeoutError (and ``qm_job`` halts it on
    unwind) instead of blocking forever; a failed/cancelled job is reported as
    such rather than fetched as silently-short arrays."""
    try:
        ok = results.wait_for_all_values(timeout=_RECORD_WAIT_BUDGET_S)
    except TimeoutError as exc:
        raise TimeoutError(
            f"result streams not complete after {_RECORD_WAIT_BUDGET_S:.0f}s — "
            f"zombie-job stall; the job will be halted. {cluster_hint(machine)}"
        ) from exc
    if not ok:
        raise RuntimeError(
            "job failed or was cancelled before its result streams completed "
            "(the qm log on stderr carries the gateway's reason)."
        )


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    The execute-and-fetch half is identical for every swept experiment, so all
    probes share this one implementation. `config` defaults to
    `machine.generate_config()`; pass an explicit config when the program needs a
    pre-built one (e.g. a baked config carrying baking ops the fresh config lacks).

    A no-progress watchdog rides the fetch loop: every program here saves the
    shot counter ``n`` once per shot, so ``n`` frozen for ``_STALL_BUDGET_S``
    means the job is running-but-dead — it is halted (by ``qm_job``'s unwind)
    and named, never waited on forever.
    """
    hint = cluster_hint(machine)
    with qm_job(machine, prog, timeout=timeout, config=config) as job:
        data_fetcher = XarrayDataFetcher(job, sweep_axes)
        last_n = None
        last_progress_t = time.monotonic()
        for dataset in data_fetcher:
            n_now = data_fetcher.get("n", 0)
            progress_counter(
                n_now,
                num_shots,
                start_time=data_fetcher.t_start,
            )
            if last_n is None or n_now != last_n:
                last_n = n_now
                last_progress_t = time.monotonic()
            elif time.monotonic() - last_progress_t > _STALL_BUDGET_S:
                raise TimeoutError(
                    f"job made no progress for {_STALL_BUDGET_S:.0f}s "
                    f"(stuck at shot {n_now}/{num_shots}); halting it. {hint}"
                )
        # Expose possible runtime errors. execution_report is a method on some QM
        # API versions and a property on others — tolerate both.
        if log:
            rep = getattr(job, "execution_report", None)
            if callable(rep):
                log(rep())
            elif rep is not None:
                log(rep)
    return dataset
