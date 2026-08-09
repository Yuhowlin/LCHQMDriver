"""QM DRAG alternating calibration for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitDragAlternating``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitDragAlternating


@register
class QMQubitDragAlternating(QubitDragAlternating):
    """Build a multiplexed DRAG alternating QUA program on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = "it executes and fetches inside probe()"

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_drag_alternating as alternating_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        
        sweeps = self.define_sweep()
        beta_array = list(sweeps["beta"])
        nb_pulses_array = [int(x) for x in sweeps["nb_of_pulses"]]

        prog, sweep_axes = alternating_probe.build_program(
            machine,
            qubits,
            num_shots=int(self.params.num_averages),
            beta_array=beta_array,
            nb_pulses_array=nb_pulses_array,
            reset_type=check_reset_method(self),
            use_state_discrimination=False,
            target_gate=getattr(self.params, "target_gate", "x180"),
        )

        params = self.params
        shots = getattr(params, "num_averages", None) or getattr(params, "num_shots", 1)

        return alternating_probe.acquire(
            machine,
            prog,
            sweep_axes,
            num_shots=int(shots),
            timeout=self.backend._timeout,
        )

