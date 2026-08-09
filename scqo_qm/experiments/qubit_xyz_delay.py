"""QM XY-Z delay for scqo — supplies ``probe()``.

Parameters, the triangle-peak fit and the ``flux_delay_s`` writeback are inherited
from ``scqo.experiments.QubitXyzDelay``; this adapter only builds and runs the
baked QM program.

Like ``pair_swap_chevron`` (and unlike the other QM adapters), ``probe()``
ACQUIRES and returns a ready ``xr.Dataset``: the program only runs against the
probe's own baked config (which carries the per-segment ``x180`` + ``flux_pulse``
ops), and the backend's shared fetch path would regenerate a config without them.
The probe already labels its axes with the scqo names (``prepared_state`` /
``relative_time_ns``), so only ``qubit`` -> ``target`` is left to the backend's
canonicalization.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitXyzDelay


@register
class QMQubitXyzDelay(QubitXyzDelay):
    """Build, run and fetch the multiplexed XY-Z delay scan on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the slid pulses "
                           "and fetches against it inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_xyz_delay as xyz_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, sweep_axes, baked_config = xyz_probe.build_program(
            machine,
            qubits,
            half_scan_ns=int(self.params.half_scan_ns),
            z_pulse_amp_v=float(self.params.z_pulse_amp_v),
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            flux_point=GOVERNED_FLUX_POINT,
        )
        # Acquire here: the baked config is per-call and cannot be reached through
        # the backend's (program, axes, module) shape.
        return xyz_probe.acquire(
            machine, prog, sweep_axes,
            num_shots=int(self.params.num_averages),
            timeout=self.backend._timeout,  # type: ignore[attr-defined]
            config=baked_config,
        )
