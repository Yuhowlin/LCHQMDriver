"""QM ramsey cryoscope for scqo — supplies only ``probe()``.

Parameters, the step-response estimator and the paired-fact writeback are
inherited from ``scqo.experiments.QubitRamseyCryoscope``. scqo sweeps the flux-pulse
DURATION (``duration_ns``, every ns) x the closing-pulse FRAME (turns); the LCHQM
probe realizes the 1 ns duration resolution with baking (<=16 ns baked, a
stretched ``const`` plus a baked remainder above) and plays the flux pulse
idle-relative at the governed flux point.

Like the swap chevron, ``probe()`` ACQUIRES and returns a ready ``xr.Dataset``:
the program only runs against the probe's own baked config (which carries the
``flux_pulse1..16`` operations), and the backend's shared fetch path would
regenerate a config without them. The probe's ``sweep_axes`` are already the
canonical scqo names in raw nesting order (``duration_ns`` outer, ``frame``
inner), so ``_to_canonical`` takes its name-based path.

Single target: the probe refuses more than one qubit (the sequence is built per
qubit); reset is resolved through ``_reset.check_reset_method`` so ``reset_method
="active"`` is refused by name rather than silently thermalized.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scqo import register
from scqo.experiments import QubitRamseyCryoscope


@register
class QMQubitRamseyCryoscope(QubitRamseyCryoscope):
    """Build, run and fetch the ramsey cryoscope phase-tomography sequence on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the phase "
                           "tomography and fetches against it inside probe()")

    def probe(self) -> Any:
        from customized.probes import qubit_ramsey_cryoscope as ramsey_cryoscope_probe
        from customized.probes._lib import select_qubits
        from customized.quam_fields import GOVERNED_FLUX_POINT

        from ._reset import check_reset_method

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        durations_ns = np.round(self.sweep_axes["duration_ns"]).astype(int)

        prog, sweep_axes, baked_config = ramsey_cryoscope_probe.build_program(
            machine,
            qubits,
            durations_ns=durations_ns,
            frames=self.sweep_axes["frame"],
            flux_amp_v=float(self.params.flux_pulse_amp_v),
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
        # Acquire here: the baked config is per-call and cannot be reached through
        # the backend's (program, axes, module) shape.
        return ramsey_cryoscope_probe.acquire(
            machine, prog, sweep_axes,
            num_shots=int(self.params.num_averages),
            timeout=self.backend._timeout,
            config=baked_config,
        )
