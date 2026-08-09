"""QM chain-stepped absolute punchout for scqo - supplies only ``probe()``.

Parameters, the per-point chain-stepping run lifecycle (python loop: re-solve
full_scale_power_dbm + amplitude per power point, one 1D acquisition each,
boundary-recorded set/revert), the punchout analysis and the
readout_power_dbm/readout_freq_hz proposals are inherited from
``scqo.experiments.ResonatorSpectroscopyPowerChain``.

By the time ``probe()`` runs, the core loop has already pushed THIS point's chain
into QUAM (and the shared acquire regenerates the QUA config every cycle), so the
probe is literally the plain 1D resonator-spectroscopy program measured at the
current readout amplitude (~0.5 full scale) — ``scqo_qm/probes/
resonator_spectroscopy.build_program`` is reused UNCHANGED, including its
``rr.wait(depletion_time)`` between shots. ``self.sweep_axes`` holds only the
detuning axis during each per-point call (the run loop swaps it in).
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ResonatorSpectroscopyPowerChain


@register
class QMResonatorSpectroscopyPowerChain(ResonatorSpectroscopyPowerChain):
    """Per-point 1D resonator-spectroscopy QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments import _resonator_spectroscopy as res_spec_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        return res_spec_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            num_shots=self.params.num_averages,
        )
