"""Power-Rabi update policy: pure fit -> amplitude decision + minimal QUAM writeback.

The shell decides *which* qubits to update (drive_qubit selection, failed
outcomes); this module decides *what* the new amplitude is and writes it.
"""

from dataclasses import dataclass
from typing import Dict

from scqo_qm import quam_fields


@dataclass(frozen=True)
class PowerRabiUpdate:
    opt_amp: float  # new pulse amplitude for the swept operation


def compute_update(fit: Dict, current_amplitude: float) -> PowerRabiUpdate:
    """Pure decision: scale the current pulse amplitude by the fitted prefactor."""
    return PowerRabiUpdate(opt_amp=fit["opt_amp_prefactor"] * current_amplitude)


def apply_update(qubit, operation: str, upd: PowerRabiUpdate, *, update_x90: bool) -> None:
    """Write the calibrated amplitude onto the QUAM qubit (call inside the
    shell's `record_state_updates()` when GUI approval is wanted).

    When the swept operation is x180 and `update_x90` is set, x90 is ALSO set to
    half the pi-pulse amplitude. That halving is this NODE's policy, exposed as its
    own GUI parameter — `quam_fields.set_pi_amp` has no mode flag for it, because
    pi and pi/2 are independent knobs (`pi_amp` / `pi_amp_x90`) and a derived x90
    would silently overwrite a real `qubit_deterministic_benchmarking` calibration.
    Power Rabi only measures the pi, so halving is a convenience, not a measurement.
    """
    quam_fields.set_pi_amp(qubit, upd.opt_amp, operation=operation)
    if update_x90 and operation == "x180":
        quam_fields.set_pi_amp(qubit, upd.opt_amp / 2.0, operation="x90")
