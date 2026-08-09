"""QM ADE T1 tracking for scqo - supplies only ``probe()``.

Parameters, the trace summary and reporting are inherited from
``scqo.experiments.QubitT1Ade``. The probe streams heterogeneous shapes
(per-block gamma/sigma, per-shot states, a timestamp stream), so ``probe()``
returns the 3-tuple ``(program, sweep_axes, probe_module)`` and the backend
uses the probe module's OWN ``acquire()`` instead of ``_lib.acquire``.

The probe always discriminates (the on-FPGA closed form consumes the state
bit), so a qubit without a calibrated readout threshold is refused BY NAME
before any QUA is built — regardless of reset_method.
"""

from __future__ import annotations

from typing import Any, ClassVar

from scqo import register
from scqo.experiments import QubitT1Ade


def discriminator_problems(machine, names: list[str]) -> list[str]:
    """One message per qubit lacking a readout threshold (empty = ready)."""
    problems = []
    for name in names:
        qubit = machine.qubits[name]
        threshold = getattr(qubit.resonator.operations["readout"], "threshold", None)
        if threshold is None:
            problems.append(f"{name} has no readout threshold")
    return problems


@register
class QMQubitT1Ade(QubitT1Ade):
    """Build the ADE T1-tracking QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_t1_ade as ade_probe

        machine = self.backend.machine  # type: ignore[attr-defined]

        problems = discriminator_problems(machine, list(self.params.targets))
        if problems:
            raise ValueError(
                "qubit_t1_ade always discriminates (the on-FPGA closed form "
                "consumes the state bit): " + "; ".join(problems) + ". Run "
                "single_shot_readout and accept its readout_threshold first."
            )

        qubits = select_qubits(machine, self.params.targets, multiplexed=False)

        prog, sweep_axes = ade_probe.build_program(
            machine,
            qubits,
            num_blocks=int(self.params.num_blocks),
            n_avg=int(self.params.num_averages),
            t0_cycles=max(4, int(round(self.params.t0_ns / 4))),
            dt_cycles=max(4, self.initial_dt_ns() // 4),
            adaptive_dt=bool(self.params.adaptive_dt),
            dt_factor=float(self.params.dt_factor),
            min_dt_cycles=max(4, int(self.params.min_dt_ns) // 4),
            max_dt_cycles=max(8, int(self.params.max_dt_ns) // 4),
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
        )
        return prog, sweep_axes, ade_probe
