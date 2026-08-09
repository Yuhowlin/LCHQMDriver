"""QM long-time (spectroscopy) cryoscope for scqo — supplies only ``probe()``.

Parameters, the spectroscopy-cryoscope estimator and the paired-fact writeback
are inherited from ``scqo.experiments.QubitSpectroscopyCryoscope``. scqo sweeps
the drive DETUNING (``detuning_hz``) x the (log-spaced) WAIT time
(``wait_time_ns``) into a parked flux pulse; the LCHQM probe realizes it with a
held ``const`` flux, a wait, and a fixed ``x180`` spectroscopy pulse.

Unlike the Ramsey cryoscope, there is NO baking — a plain stretched ``const`` plus
a fixed pulse — so ``probe()`` returns ``(program, sweep_axes)`` and the backend's
shared fetch path runs it (no per-call baked config to thread through).

Single target: the probe refuses more than one qubit (one parked drive + one flux
line + one wait axis); reset is resolved through ``_reset.check_reset_method`` so
``reset_method="active"`` is refused by name. The drive is centered on the
arch-predicted parked detuning ``resolved_center_offset_hz`` (no LO shift —
band-limited v1).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scqo import register
from scqo.experiments import QubitSpectroscopyCryoscope


@register
class QMQubitSpectroscopyCryoscope(QubitSpectroscopyCryoscope):
    """Build the long-time cryoscope spectroscopy sweep on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.probes import qubit_spectroscopy_cryoscope as spec_cryo_probe
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        from ._reset import check_reset_method

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        target = str(self.params.targets[0])
        # ns (on the 4 ns grid) -> clock cycles for the QUA wait loop
        wait_ns = self.sweep_axes["wait_time_ns"]
        wait_cycles = np.maximum(4, np.round(wait_ns / 4)).astype(int)

        return spec_cryo_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            wait_cycles=wait_cycles,
            flux_amp_v=float(self.params.flux_pulse_amp_v),
            center_offset_hz=self.resolved_center_offset_hz(target),
            operation_len_ns=int(self.params.drive_len_ns),
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
