"""QM three-state single-shot readout for scqo - supplies only ``probe()``.

Parameters, the three-Gaussian-mixture fit and reporting are inherited from
``scqo.experiments.SingleShotReadoutGEF``. Same PER-SHOT contract as the
two-state probe (streams ``buffer(3).buffer(num_shots)``, no ``.average()``)
and the same axis-order note: the QUA loops nest the shot loop OUTER over the
prepared-state loop, so the raw array is (shot_idx, prepared_state) and the
backend's name-based ``_to_canonical`` path handles it.

Preparing |f> needs a calibrated EF pi pulse — the QUAM operation ``EF_x180``
on the qubit's xy channel — and the qubit's ``anharmonicity``, which is how far
below f_01 that pulse plays. Both are refused BY NAME here rather than
compiling a program that silently prepares |e> three times: the fit would find
two blobs, the third weight would collapse and the failure would look like a
readout problem instead of a missing calibration. Calibrate them with the
vendored ``12_Qubit_Spectroscopy_E_to_F`` and ``13_power_rabi_ef`` nodes.

No ``update()`` override: unlike the two-state experiment this proposes no
discriminator, because a scalar threshold on one rotated quadrature cannot
separate three blobs.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import SingleShotReadoutGEF


def _ef_surface_problems(machine, names: list[str]) -> list[str]:
    """One message per qubit lacking the EF drive surface (empty = ready)."""
    problems = []
    for name in names:
        qubit = machine.qubits[name]
        missing = []
        if "EF_x180" not in getattr(qubit.xy, "operations", {}):
            missing.append("the xy operation 'EF_x180'")
        if not getattr(qubit, "anharmonicity", None):
            missing.append("anharmonicity")
        if missing:
            problems.append(f"{name} is missing {' and '.join(missing)}")
    return problems


@register
class QMSingleShotReadoutGEF(SingleShotReadoutGEF):
    """Build a multiplexed per-shot |g>/|e>/|f> readout QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.experiments import _readout_fidelity as fidelity_probe

        machine = self.backend.machine  # type: ignore[attr-defined]

        problems = _ef_surface_problems(machine, list(self.params.targets))
        if problems:
            raise ValueError(
                "cannot prepare |f>: " + "; ".join(problems) + ". Calibrate the 1->2 "
                "transition first (qualibrate nodes 12_Qubit_Spectroscopy_E_to_F for "
                "the anharmonicity, then 13_power_rabi_ef for the EF_x180 amplitude), "
                "or run single_shot_readout for a two-state measurement."
            )

        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        return fidelity_probe.build_program(
            machine,
            qubits,
            operation="readout",
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
            prepared_states=[0, 1, 2],
            readout_freq_shift_hz=self.params.readout_freq_shift_hz,
        )
