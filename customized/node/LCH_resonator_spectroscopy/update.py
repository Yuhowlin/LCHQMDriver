"""Resonator-spectroscopy update policy: fitted resonance -> resonator frequency writeback.

The estimator's absolute resonance frequency is written to both
`q.resonator.f_01` and `q.resonator.RF_frequency`. Success gating stays in the
shell (it owns `node.outcomes`).
"""

from dataclasses import dataclass
from typing import Dict

from scqo_qm import quam_fields


@dataclass(frozen=True)
class ResonatorSpecUpdate:
    frequency: float  # Hz; the fitted resonator resonance frequency


def compute_update(fit: Dict) -> ResonatorSpecUpdate:
    """Pure decision: the estimator's fitted resonance is the new resonator frequency."""
    return ResonatorSpecUpdate(frequency=float(fit["frequency"]))


def apply_update(qubit, upd: ResonatorSpecUpdate) -> None:
    """Write the resonance frequency onto the QUAM resonator (call inside the
    shell's `record_state_updates()` when GUI approval is wanted)."""
    quam_fields.set_readout_freq(qubit, upd.frequency)
