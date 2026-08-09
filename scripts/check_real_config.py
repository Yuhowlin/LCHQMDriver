"""Self-test the scqo stack against a REAL QUAM state (OPX1000) — no hardware needed.

    python scripts/check_real_config.py D:\\qpu_data\\SQ_demo\\QM_OPX1000_config
    python scripts/check_real_config.py <state_dir> --qubits q1 q2

Loads your ``state.json`` + ``wiring.json``, then runs the full scqo pipeline with
SIMULATED data over the REAL device tree: read neutral fields -> run experiments ->
fit -> write results back -> save (to an explicit scratch path ONLY — the live
quam_state is never targeted) -> reload and compare. Everything happens on a
temporary copy; your original files are never opened for writing.

Needs the QM environment (lab: ``conda activate LCHQM_test``).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root for `customized`


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("state_dir", help="folder holding state.json + wiring.json")
    parser.add_argument("--qubits", nargs="+", help="qubits to exercise (default: all in the state)")
    args = parser.parse_args()

    source = Path(args.state_dir)
    for fname in ("state.json", "wiring.json"):
        if not (source / fname).is_file():
            raise SystemExit(f"{fname} not found in {source}")
    work = Path(tempfile.mkdtemp(prefix="scqo_qm_selftest_"))
    shutil.copy(source / "state.json", work / "state.json")
    shutil.copy(source / "wiring.json", work / "wiring.json")
    print(f"sandbox: {work}")
    print("  (temporary self-test copies + throwaway run data: your originals and your")
    print("   real data_root are NOT touched; real measurements use `scqo run`)")

    try:
        from quam_config import Quam
    except ModuleNotFoundError as err:
        raise SystemExit(
            f"missing package: {err.name}\n"
            "This self-test needs the QM stack (quam/qm + this repo installed). "
            "Run it in the lab's QM environment:  conda activate LCHQM_test"
        )

    machine = Quam.load(str(work))
    print(f"[1/5] loaded QUAM | qubits: {list(machine.qubits)}")

    import scqo_qm.experiments  # noqa: F401
    from scqo_qm.backend.qm_backend import QMDeviceModel
    from scqo import Session
    from scqo.testing import SimulatedBackend

    dm = QMDeviceModel(machine)
    snap = dm.snapshot()
    for name, fields in snap.items():
        print(f"      {name}: {fields}")
    unreadable = [q for q, f in snap.items() if any(v is None for v in f.values())]
    qubits = args.qubits or [q for q in snap if q not in unreadable]
    print(f"[2/5] snapshot OK | testing qubits: {qubits}"
          + (f" (skipping uncalibrated: {unreadable})" if unreadable else ""))

    sess = Session(SimulatedBackend(dm), data_root=work / "data", device_name="selftest")
    before = {q: dict(v) for q, v in sess.device_state().items()}
    failures = []
    for experiment in ("resonator_spectroscopy", "qubit_power_rabi"):
        # update="apply": this self-test exists to exercise writeback (scqo v0.6.0
        # defaults to suggest-only, which would leave the QUAM objects untouched).
        result = sess.run(experiment, {"qubits": qubits}, update="apply", tags=["selftest"])
        ok = all(result["outcomes"].get(q) == "successful" for q in qubits) and not result.get("error")
        print(f"[3/5] {experiment}: {result['outcomes']}" + (f" error={result['error']}" if result.get("error") else ""))
        if not ok:
            failures.append(experiment)

    after = sess.device_state()
    moved = [q for q in qubits if after[q] != before[q]]
    print(f"[4/5] writeback reached the real QUAM objects for: {moved or 'NONE'}")
    if set(moved) != set(qubits):
        failures.append("writeback")

    saved = work / "saved_state"
    machine.save(path=saved)  # explicit scratch path ONLY — never the default quam_state
    reloaded = Quam.load(str(saved))
    dm2 = QMDeviceModel(reloaded)
    round_trip = all(
        abs(dm2.snapshot()[q]["readout_freq"] - after[q]["readout_freq"]) < 1e-3 for q in qubits
    )
    print(f"[5/5] QUAM save/reload round-trip (scratch path): {'OK' if round_trip else 'MISMATCH'}")
    if not round_trip:
        failures.append("save-roundtrip")

    print(f"\nruns saved + indexed under {work / 'data'}: "
          f"{[r['run_id'] for r in sess.find_runs(tag='selftest')]}")
    print("\nPASS - scqo works against this real state" if not failures
          else f"\nFAIL - problems in: {', '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
