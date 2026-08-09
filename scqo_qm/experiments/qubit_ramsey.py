"""QM Ramsey for scqo - supplies only ``probe()``.

Parameters, the decaying-cosine fit, T2*/detuning extraction and the drive_freq_hz
writeback are all inherited from ``scqo.experiments.QubitRamsey``. This class only
compiles the scqo sweep into a QUA program via the shared LCHQM Ramsey probe.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from scqo import register
from scqo.experiments import QubitRamsey


@register
class QMQubitRamsey(QubitRamsey):
    """Build a multiplexed Ramsey QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    #: Ramsey is also the sensitive test of the settle: residual readout photons
    #: Stark-shift the first y90 and surface as a fitted-detuning error.
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_ramsey as ramsey_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # scqo sweeps idle time in ns; the QUA program sweeps in clock cycles (4 ns).
        idle_ns = self.sweep_axes["idle_time_ns"]
        idle_times_cycles = np.maximum(1, np.round(idle_ns / 4)).astype(int)

        return ramsey_probe.build_program(
            machine,
            qubits,
            idle_times_cycles=idle_times_cycles,
            detuning_hz=int(self.params.frequency_detuning_hz),
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
