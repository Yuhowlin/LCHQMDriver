"""QM qubit spectroscopy for scqo - supplies only ``probe()``.

Parameters, peak fitting and the drive_freq_hz writeback are inherited from
``scqo.experiments.QubitSpectroscopy``. scqo sweeps ``detuning_hz``; the LCHQM probe
builds the same sweep on coord ``detuning``, which the backend's ``_to_canonical``
renames back.

Drive power contract: the core ``run()`` already solved the drive chain for
``drive_power_dbm`` (recorded set -> acquire -> revert), parking the exact
amplitude on the saturation op — so the probe plays it at ``amplitude_scale=1.0``
(exact in QUA fixed point). The shared probe keeps its ``operation_amp`` argument
for the qualibrate node, a separate consumer with its own explicit amps.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitSpectroscopy


@register
class QMQubitSpectroscopy(QubitSpectroscopy):
    """Build a multiplexed two-tone spectroscopy QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_spectroscopy as spec_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return spec_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            operation="saturation",
            operation_len=int(self.params.drive_len_ns) if self.params.drive_len_ns else None,
            operation_amp=1.0,  # run() parked the exact amplitude on the saturation op
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
        )
