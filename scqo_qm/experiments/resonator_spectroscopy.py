"""QM resonator spectroscopy for scqo - supplies only ``probe()``.

Parameters, the Lorentzian-dip fit and the readout_freq_hz writeback are inherited
from ``scqo.experiments.ResonatorSpectroscopy``. scqo sweeps ``detuning_hz``;
the LCHQM probe builds the same sweep on coord ``detuning``, which the backend's
``_to_canonical`` renames to ``detuning_hz``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ResonatorSpectroscopy


@register
class QMResonatorSpectroscopy(ResonatorSpectroscopy):
    """Build a multiplexed resonator-spectroscopy QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments import _resonator_spectroscopy as resonator_spec_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return resonator_spec_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            num_shots=self.params.num_averages,
        )
