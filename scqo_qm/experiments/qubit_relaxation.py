"""QM qubit relaxation (T1) for scqo - supplies only ``probe()``.

Parameters, exponential fit and reporting are inherited from
``scqo.experiments.QubitRelaxation``. scqo sweeps ``wait_time_ns``; the LCHQM probe
builds the same sweep on coord ``idle_time``, which the backend's ``_to_canonical``
renames back.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitRelaxation


@register
class QMQubitRelaxation(QubitRelaxation):
    """Build a multiplexed T1 QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_relaxation as t1_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        wait_ns = self.sweep_axes["wait_time_ns"]
        wait_cycles = np.maximum(1, np.round(wait_ns / 4)).astype(int)

        return t1_probe.build_program(
            machine,
            qubits,
            wait_times_cycles=wait_cycles,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
