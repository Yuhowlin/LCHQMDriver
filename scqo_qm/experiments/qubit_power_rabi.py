"""QM qubit power Rabi for scqo - supplies only ``probe()``.

Parameters, the cosine fit, opt_amp_prefactor extraction and the pi_amp writeback are
inherited from ``scqo.experiments.QubitPowerRabi``. scqo's ``amp_prefactor`` is already
a factor of the current pi pulse, which is exactly the LCHQM probe's
``amplitude_scale``, so the sweep passes straight through — and since the probe emits
that same axis NAME, ``_to_canonical`` matches it by name instead of by position.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitPowerRabi


@register
class QMQubitPowerRabi(QubitPowerRabi):
    """Build a multiplexed power-Rabi QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_power_rabi as power_rabi_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return power_rabi_probe.build_program(
            machine,
            qubits,
            amps=self.sweep_axes["amp_prefactor"],
            operation="x180",
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            drive_qubit=None,
        )
