"""QM readout-frequency (fidelity vs readout detuning) for scqo - supplies only ``probe()``.

Parameters, the per-frequency two-Gaussian-mixture fit and the ``readout_freq_hz``
writeback are inherited from ``scqo.experiments.ReadoutFrequency``. PER-SHOT
contract: every readout shot's I/Q point is recorded individually — the probe's
streams are ``buffer(2).buffer(len(dfs)).buffer(num_shots)`` with NO
``.average()``. scqo sweeps ``detuning_hz``; the LCHQM probe applies each value
relative to the resonator's current IF (``df + intermediate_frequency`` — same
convention as the resonator_spectroscopy wrapper) but names its axis
``frequency``, which we re-key below.

AXIS-ORDER NOTE (do not "fix" the order below): the probe's QUA loops nest shot
(outer) over detuning over prepared-state (inner), so the raw per-qubit array is
shaped (shot_idx, frequency, prepared_state) — a permutation of scqo's declared
sweep order (detuning_hz, prepared_state, shot_idx). We re-key the probe's
sweep_axes with the canonical scqo names IN RAW NESTING ORDER (only ``frequency``
-> ``detuning_hz`` actually changes; the values already ARE the detunings), so
the backend's ``_to_canonical`` takes its name-based path (per-name size checks,
no positional rename — which would scramble the axes) and ``estimate()``
transposes by name.

SHOT-INDEX VALUES NOTE: the probe's ``shot_idx`` coord is ``arange(1, n+1)``
while scqo declares ``arange(n)``. This offset is acceptable: the name-based
``_to_canonical`` path asserts SIZES, not values, and ``estimate()`` uses the
coord only for transposing/slicing, never its numeric values (same situation as
the single_shot_readout / readout_fidelity pair).
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import ReadoutFrequency


@register
class QMReadoutFrequency(ReadoutFrequency):
    """Build a multiplexed per-shot readout-frequency-scan QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import readout_frequency as freq_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, axes = freq_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
        )
        # Canonical names in RAW nesting order (shot outer, detuning, prepared
        # state inner — see module docstring); the DataArray values are reused
        # (the probe's "frequency" values already are the detunings in Hz).
        sweep_axes = {
            "qubit": axes["qubit"],
            "shot_idx": axes["shot_idx"],
            "detuning_hz": axes["frequency"],
            "prepared_state": axes["prepared_state"],
        }
        return prog, sweep_axes
