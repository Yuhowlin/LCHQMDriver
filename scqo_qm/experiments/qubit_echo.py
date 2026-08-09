"""QM qubit echo (Hahn) for scqo - supplies only ``probe()``.

Parameters, exponential-envelope fit and reporting are inherited from
``scqo.experiments.QubitEcho``. scqo sweeps ``wait_time_ns`` (the TOTAL echo idle
time tau); the LCHQM probe sweeps the per-arm wait in clock cycles (two arms of
tau/2, 4 ns per cycle -> cycles = tau_ns / 8) and builds the same sweep on coord
``idle_time``, which the backend's ``_to_canonical`` renames back.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitEcho


@register
class QMQubitEcho(QubitEcho):
    """Build a multiplexed Hahn-echo QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_echo as echo_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # Total idle time tau (ns) -> per-arm clock cycles: tau/2 per arm, 4 ns per
        # cycle. QUA wait() needs >= 1 cycle, hence the floor.
        wait_ns = self.sweep_axes["wait_time_ns"]
        arm_cycles = np.maximum(1, np.round(np.asarray(wait_ns) / 8)).astype(int)

        return echo_probe.build_program(
            machine,
            qubits,
            idle_times_cycles=arm_cycles,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
