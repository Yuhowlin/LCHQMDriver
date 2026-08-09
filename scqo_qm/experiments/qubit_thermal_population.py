"""QM residual-population measurement for scqo - supplies only ``probe()``.

Parameters, the pinned-center fit and the ``n_th`` writeback are inherited from
``scqo.experiments.QubitThermalPopulation``. The program is the readout-fidelity
one with a single prepared state: reset, measure, repeat — no drive pulse at all,
so the cloud is whatever the passive wait leaves behind.

Same PER-SHOT contract and axis-order note as ``single_shot_readout``; the
streams are ``buffer(1).buffer(num_shots)`` and the length-1 prepared_state axis
is kept so the dataset shape matches the rest of the family.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitThermalPopulation


@register
class QMQubitThermalPopulation(QubitThermalPopulation):
    """Build a multiplexed per-shot |g>-only readout QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments import _readout_fidelity as fidelity_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return fidelity_probe.build_program(
            machine,
            qubits,
            operation="readout",
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
            prepared_states=[0],
        )
