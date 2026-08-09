"""QM adaptive Bayesian T1 for scqo - supplies only ``probe()``.

Parameters, the credible-interval analysis and reporting are inherited from
``scqo.experiments.QubitT1Bayesian``. The probe streams heterogeneous shapes,
so ``probe()`` returns the 3-tuple ``(program, sweep_axes, probe_module)`` and
the backend uses the probe module's OWN ``acquire()``.

Two vendor-side prerequisites, each refused BY NAME before any QUA is built:

* a calibrated readout threshold (the sequence discriminates every shot);
* QUAM's ``resonator.confusion_matrix`` — the SPAM-aware likelihood reads
  alpha = P(read 0 | prep 1) and beta = P(read 1 | prep 0) from it. The matrix
  is written by the 07_iq_blobs qualibrate node; it is deliberately dead to
  SCQO's neutral surface (placement rule), so the probe reads it straight off
  the QUAM tree and this shell owns the refusal.
"""

from __future__ import annotations

from typing import Any, ClassVar

import xarray as xr

from scqo import register
from scqo.experiments import QubitT1Bayesian

from .qubit_t1_ade import discriminator_problems


def confusion_matrix_problems(machine, names: list[str]) -> list[str]:
    """One message per qubit lacking a usable 2x2 confusion matrix."""
    problems = []
    for name in names:
        qubit = machine.qubits[name]
        matrix = getattr(qubit.resonator, "confusion_matrix", None)
        if (matrix is None or len(matrix) != 2
                or any(len(row) != 2 for row in matrix)):
            problems.append(f"{name} has no 2x2 resonator.confusion_matrix")
    return problems


@register
class QMQubitT1Bayesian(QubitT1Bayesian):
    """Build the adaptive Bayesian T1 QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_t1_bayesian as bayes_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)

        problems = discriminator_problems(machine, targets)
        problems += confusion_matrix_problems(machine, targets)
        if problems:
            raise ValueError(
                "qubit_t1_bayesian needs a calibrated discriminator AND a "
                "measured confusion matrix (the SPAM-aware likelihood reads "
                "alpha/beta from it): " + "; ".join(problems) + ". Run "
                "single_shot_readout (accept readout_threshold), then the "
                "07_iq_blobs qualibrate node to store the confusion matrix."
            )

        qubits = select_qubits(machine, targets, multiplexed=False)
        lin_wait_cycles = self.lin_wait_ns() // 4

        prog, sweep_axes = bayes_probe.build_program(
            machine,
            qubits,
            num_blocks=int(self.params.num_blocks),
            num_probes=int(self.params.num_probes),
            c_adaptive=float(self.params.adaptive_c),
            k0=float(self.params.k0),
            t1_prior_s={t: self._t1_prior_s[t] for t in targets},
            t1_min_s=float(self.params.t1_min_s),
            t1_max_s=float(self.params.t1_max_s),
            k_min=float(self.params.k_min),
            k_max=float(self.params.k_max),
            interleaved=bool(self.params.interleaved_validation),
            lin_wait_cycles=lin_wait_cycles,
            active_reset_per_probe=bool(self.params.active_reset_per_probe),
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
        )
        if self.params.interleaved_validation:
            # side-channel for the probe's acquire(): the validation grid the
            # program compiled in, so lin_wait_s lands in the dataset
            sweep_axes["lin_wait_cycles"] = xr.DataArray(lin_wait_cycles)
        return prog, sweep_axes, bayes_probe
