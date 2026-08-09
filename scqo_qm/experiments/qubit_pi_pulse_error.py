"""QM Pi-pulse error amplification for scqo - supplies only ``probe()``."""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitPiPulseError


@register
class QMQubitPiPulseError(QubitPiPulseError):
    """Build a multiplexed Pi-pulse error amplification QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_pi_pulse_error as pi_pulse_error_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # self.sweep_axes, not a second define_sweep() call: run() has already
        # built the axes, and rebuilding them here is a way for the emitted sweep
        # and the declared one to drift apart
        return pi_pulse_error_probe.build_program(
            machine,
            qubits,
            amp_prefactors=list(self.sweep_axes["amp_prefactor"]),
            gate_counts=list(self.sweep_axes["gate_count"]),
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=False,
        )
