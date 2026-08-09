"""QM Deterministic Benchmarking for scqo - supplies only ``probe()``."""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitDeterministicBenchmarking


@register
class QMQubitDeterministicBenchmarking(QubitDeterministicBenchmarking):
    """Build a multiplexed Deterministic Benchmarking QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_deterministic_benchmarking as db_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        repetitions = self.params.get_repetitions()

        return db_probe.build_program(
            machine,
            qubits,
            num_shots=int(self.params.num_averages),
            repetitions=repetitions,
            # from sweep_axes, not the Parameters: run() has already resolved the
            # window (explicit list / single-point mode / linspace) into the axis
            amp_prefactors=list(self.sweep_axes["amp_prefactor"]),
            target_gate=self.params.target_gate,
            reset_type=check_reset_method(self),
            use_state_discrimination=self.params.use_state_discrimination,
        )


