"""QM DRAG equator calibration for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitDragEquator``.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitDragEquator


@register
class QMQubitDragEquator(QubitDragEquator):
    """Build a multiplexed DRAG equator QUA program on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it generates a reference-alpha config and "
                           "fetches against it inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from customized.probes._lib import select_qubits
        from customized.probes import qubit_drag_equator as equator_probe

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        sweeps = self.define_sweep()
        beta_array = list(sweeps["beta"])

        # build_program temporarily sets QUAM alpha = ref_alpha and generates the QM
        # config while that modified alpha is active (so the waveform baked into the
        # hardware config encodes the correct DRAG Q amplitude for fixed-point scaling).
        # It then restores the original alpha and returns (prog, sweep_axes, config).
        prog, sweep_axes, config = equator_probe.build_program(
            machine,
            qubits,
            num_shots=int(self.params.num_averages),
            beta_array=beta_array,
            pulse_repetitions=int(self.params.pulse_repetitions),
            reset_type=check_reset_method(self),
            use_state_discrimination=False,
            target_gate=getattr(self.params, "target_gate", "x180"),
        )




        params = self.params
        shots = getattr(params, "num_averages", None) or getattr(params, "num_shots", 1)
        # Pass the pre-built config so _lib.acquire uses the waveform with ref_alpha,
        # not the now-restored (original alpha) machine state.
        return equator_probe.acquire(
            machine,
            prog,
            sweep_axes,
            num_shots=int(shots),
            timeout=self.backend._timeout,
            config=config,
        )
