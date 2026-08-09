"""Resonator-spectroscopy-vs-power update policy: pure fit -> (optimal power,
frequency shift) decision + minimal QUAM writeback.

Mirrors the official 02b writeback: set the resonator output power to the fitted
optimal power, and shift the readout frequency (both ``RF_frequency`` and
``f_01``, kept in step) by the dip shift measured at that power. Success gating
stays in the shell (it owns `node.outcomes`).
"""

from dataclasses import dataclass
from typing import Dict

from scqo_qm import quam_fields


@dataclass(frozen=True)
class ReadoutPowerUpdate:
    optimal_power: float  # dBm; passed to resonator.set_output_power
    frequency_shift: float  # Hz; added to the resonator readout frequency
    max_amp: float  # readout amplitude cap for set_output_power


def compute_update(fit: Dict, max_amp: float) -> ReadoutPowerUpdate:
    """Pure decision: the estimator's optimal power and the dip shift at that power."""
    return ReadoutPowerUpdate(
        optimal_power=float(fit["optimal_power"]),
        frequency_shift=float(fit["frequency_shift"]),
        max_amp=float(max_amp),
    )


def apply_update(qubit, upd: ReadoutPowerUpdate) -> None:
    """Write the optimal readout power and shift the readout frequency onto the
    QUAM resonator (call inside the shell's `record_state_updates()` when GUI
    approval is wanted)."""
    qubit.resonator.set_output_power(power_in_dbm=upd.optimal_power, max_amplitude=upd.max_amp)
    # Shift both RF_frequency and f_01 by the same delta, keeping them in step (02b).
    quam_fields.shift_readout_freq(qubit, upd.frequency_shift)
    if getattr(qubit.resonator, "f_01", None) is not None:
        qubit.resonator.f_01 = float(qubit.resonator.f_01) + float(upd.frequency_shift)
