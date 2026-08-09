"""QM qubit tomography for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitTomography``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitTomography


@register
class QMQubitTomography(QubitTomography):
    """Build a multiplexed qubit tomography QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_tomography as tomo_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, sweep_axes = tomo_probe.build_program(
            machine,
            qubits,
            qubit_configs=self.params.qubit_configs,
            gate_counts=list(self.params.gate_counts),
            num_shots=int(self.params.num_averages),
            num_training_shots=int(self.params.num_training_shots),
            symmetrized_readout=self.params.symmetrized_readout,
            reset_type=check_reset_method(self),
        )
        return prog, sweep_axes, tomo_probe
