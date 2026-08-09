"""QM resonator spectroscopy vs ABSOLUTE power (amplitude sweep) for scqo - supplies
only ``probe()``.

Parameters, the punchout analysis and the readout_power_dbm/readout_freq_hz writeback
are inherited from ``scqo.experiments.ResonatorSpectroscopyPowerAmp``. scqo sweeps
``(power_dbm, detuning_hz)``; the LCHQM probe builds the same sweep on coords
``(power, detuning)``, which the backend's ``_to_canonical`` renames positionally.

Power convention: the core ``run()`` already solved the chain for the window top
(``readout_power_dbm = max_power_dbm``), so the probe's ``amps`` prefactors are
relative to THAT top — ``10**((power_dbm - max_power_dbm)/20)``, top point exactly
1.0 (``amplitude_scale`` scales the readout pulse the setter just parked at ~0.5 of
full scale; every prefactor is <= 1, well inside QUA's range). Same realization as
the qualibrate ``LCH_resonator_spectroscopy_power`` node (set-top -> sweep down ->
revert), whose shared probe module is reused unchanged. Loop order is amplitude ->
averages -> frequency (the probe's contract); the ring-down wait is the governed
depletion time — the per-run ``readout_depletion_ns`` override, else the readout
channel's calibrated ``readout_depletion_s`` knob, which on this backend IS the
resonators' ``depletion_time`` (None = never calibrated, keep the QUAM values).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import ResonatorSpectroscopyPowerAmp
from scqo.experiments._depletion import depletion_wait_ns


@register
class QMResonatorSpectroscopyPowerAmp(ResonatorSpectroscopyPowerAmp):
    """Build a multiplexed punchout QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import resonator_spectroscopy_power as power_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        power_dbm = np.asarray(self.sweep_axes["power_dbm"])
        # prefactors relative to the window top run() solved the chain for
        amps = 10.0 ** ((power_dbm - self.params.max_power_dbm) / 20.0)
        return power_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            amps=amps,
            power_dbm=power_dbm,  # absolute dBm axis
            num_shots=self.params.num_averages,
            # Per-run override, else the calibrated readout_depletion_s knob —
            # resolved by scqo's ONE precedence helper, so this probe and the
            # Qblox one cannot come to disagree about what the wait is. None
            # (never calibrated) keeps the probe's per-qubit QUAM values, which
            # on this backend IS the same field the knob writes.
            depletion_time_ns=depletion_wait_ns(self, self.params.targets[0]),
        )
