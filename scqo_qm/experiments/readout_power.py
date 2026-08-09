"""QM readout-power (fidelity vs readout amplitude) for scqo - supplies only ``probe()``.

Parameters, the per-amplitude two-Gaussian-mixture fit and the ``readout_amp``
writeback are inherited from ``scqo.experiments.ReadoutPower``. PER-SHOT
contract: every readout shot's I/Q point is recorded individually — the probe's
streams are ``buffer(2).buffer(len(amps)).buffer(num_shots)`` with NO
``.average()``. The swept ``amp_prefactor`` values are applied as QUA
``amplitude_scale`` on the readout pulse (prefactor x current readout_amp).

AXIS-ORDER NOTE: the probe's QUA loops nest shot (outer) over amplitude over
prepared-state (inner), so the raw per-qubit array is shaped
(shot_idx, amp_prefactor, prepared_state) — a permutation of scqo's declared
sweep order (amp_prefactor, prepared_state, shot_idx). The probe's sweep_axes
already carry exactly the canonical names ``shot_idx``/``amp_prefactor``/
``prepared_state`` in that raw nesting order, so the backend's ``_to_canonical``
takes its name-based path (no positional rename — which would scramble the axes)
and ``estimate()`` transposes by name.

SHOT-INDEX VALUES NOTE: the probe's ``shot_idx`` coord is ``arange(1, n+1)``
while scqo declares ``arange(n)``. This offset is acceptable: the name-based
``_to_canonical`` path asserts SIZES, not values, and ``estimate()`` uses the
coord only for transposing/slicing, never its numeric values (same situation as
the single_shot_readout / readout_fidelity pair).
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ReadoutPower


@register
class QMReadoutPower(ReadoutPower):
    """Build a multiplexed per-shot readout-amplitude-scan QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import readout_power as power_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return power_probe.build_program(
            machine,
            qubits,
            amps=self.sweep_axes["amp_prefactor"],
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
        )
