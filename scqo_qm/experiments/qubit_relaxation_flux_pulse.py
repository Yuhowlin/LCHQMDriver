"""QM qubit relaxation vs flux PULSE (T1 spectrum) for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from
``scqo.experiments.QubitRelaxationFluxPulse``.

PULSE CONTRACT: this probe conforms to the ``_pulse`` name by construction - the
z bias is a ``const`` PULSE played during the idle delay
(``scqo_qm/probes/qubit_relaxation_flux.py``), which the DAC adds to the
standing offset ``initialize_qpu`` applies, so ``flux_bias_v`` is an excursion
FROM ``idle_flux`` and 0 V means "stay parked". ``flux_point`` is passed
explicitly for the reason the probe documents: the bias that was rail-validated
must be the bias that plays. The probe MODULE keeps its historical name (it is
shared with the qualibrate ``LCH_T1_spectrum`` shell).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import QubitRelaxationFluxPulse


@register
class QMQubitRelaxationFluxPulse(QubitRelaxationFluxPulse):
    """Build a multiplexed T1 vs flux-PULSE QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_relaxation_flux as t1_flux_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        sweeps = self.define_sweep()
        flux_bias = list(sweeps["flux_bias_v"])
        wait_ns = sweeps["wait_time_ns"]
        wait_cycles = np.maximum(1, np.round(wait_ns / 4)).astype(int)

        return t1_flux_probe.build_program(
            machine,
            qubits,
            wait_times_cycles=wait_cycles,
            flux_amps_v=flux_bias,
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
