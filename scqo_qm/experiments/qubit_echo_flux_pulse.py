"""QM qubit echo vs flux PULSE (T2 echo spectrum) for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from
``scqo.experiments.QubitEchoFluxPulse``.

PULSE CONTRACT: as in ``qubit_relaxation_flux_pulse`` - the z bias is a ``const``
PULSE (played in BOTH echo arms) riding on the standing offset, so
``flux_bias_v`` is an excursion FROM ``idle_flux``. The probe MODULE keeps its
historical name (shared with the qualibrate shell path).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import QubitEchoFluxPulse


@register
class QMQubitEchoFluxPulse(QubitEchoFluxPulse):
    """Build a multiplexed T2 Echo vs flux-PULSE QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_echo_flux as echo_flux_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        sweeps = self.define_sweep()
        flux_bias = list(sweeps["flux_bias_v"])
        wait_ns = sweeps["wait_time_ns"]
        wait_cycles = np.maximum(1, np.round((wait_ns / 2) / 4)).astype(int)

        return echo_flux_probe.build_program(
            machine,
            qubits,
            wait_times_cycles=wait_cycles,
            flux_amps_v=flux_bias,
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
