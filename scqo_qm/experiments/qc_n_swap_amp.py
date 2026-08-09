"""QM N-swap x flux-amplitude error-amplification map for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.QcNSwapAmp``. The vendor probe
(``scqo_qm/probes/qc_N_swap_amp``) excites the pair's CONTROL qubit and plays
every swap through ``pair.macros[swap_operation].apply(ctrl_amp=...)`` — the
control-side flux amplitude in absolute volts, exactly scqo's ``flux_amp_v``
axis — and keeps EVERY shot's per-qubit discriminated state. This adapter:

* refuses unless ``drive_side``/``flux_side`` resolve to the vendor CONTROL
  member (the probe has no target-side mode; a silent role mismatch would
  mislabel the prepared/transfer panels);
* orders the per-shot states into the readout schema's (high, low) ``member``
  axis and, in ``readout_mode="average"``, reduces them to ``joint_population``
  via scqo's shared ``states_to_joint_population`` — ``"shot"`` returns the
  per-member ``state`` form as-is (the full-information trade).

Unlike the FPGA-averaged pair adapters this one does not use
``JointPopulationMixin``: the reduction happens here from per-shot data, with
the SAME helper the simulated backend uses.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import QcNSwapAmp, states_to_joint_population


def _member_states(state: xr.DataArray, high_side: str) -> xr.DataArray:
    """The probe's per-shot states, reordered onto the schema's axes.

    ``state`` arrives ``(qubit, shot, qubit_amplitude, round)`` with the qubit
    axis in READOUT order — ``[control, target]``, fixed by the adapter's
    measure list. The member axis is ROLE-ordered (high, low), so the vendor
    order flips when the roster's high member is the vendor target."""
    order = [0, 1] if high_side == "control" else [1, 0]
    da = state.isel(qubit=order).rename({
        "qubit": "member", "shot": "shot_idx",
        "qubit_amplitude": "flux_amp_v", "round": "swap_count",
    })
    return da.assign_coords(member=["high", "low"])


@register
class QMQcNSwapAmp(QcNSwapAmp):
    """Build, run and fetch the N-swap amplitude map on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it executes one program per swap pair in a "
                           "Python loop inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair
        from scqo_qm.probes import qc_N_swap_amp as swap_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time. `high` fixes the member ordering.
        high_side = role_side(self, "high", field="targets")
        drive = role_side(self, self.params.drive_side, field="drive_side")
        flux = role_side(self, self.params.flux_side, field="flux_side",
                         needs_flux=True)
        if drive != "control" or flux != "control":
            raise ValueError(
                f"qc_n_swap_amp on QM excites AND flux-drives the vendor pair's "
                f"CONTROL member (the swap macro's ctrl_amp is the only swept "
                f"knob), but drive_side={self.params.drive_side!r} / flux_side="
                f"{self.params.flux_side!r} resolve to (drive={drive}, "
                f"flux={flux}). Select the control-side role for both — or run "
                f"pair_swap_chevron, which honors either side.")

        # One door for the reset method: refuses what QM cannot honour (this
        # shell does not opt into active reset — default DENY — so 'active'
        # refuses by name until it has hardware evidence for this sequence).
        reset = check_reset_method(self)

        amps = np.asarray(self.sweep_axes["flux_amp_v"], dtype=float)
        counts = np.asarray(self.sweep_axes["swap_count"]).astype(int)
        shots = int(self.params.num_averages)
        gap_ns = int(self.params.operation_gap_ns)

        # One program per pair (the probe takes a single swap pair); the two
        # measured qubits are exactly the pair's members, control first.
        per_pair = []
        for pair in self.params.targets:
            qp = vendor_pair(self, pair)
            prog, sweep_axes = swap_probe.build_program(
                machine, [qp.qubit_control, qp.qubit_target], qp,
                swap_operation=self.params.swap_operation,
                rounds_array=counts,
                qubit_amplitudes=amps,
                num_shots=shots,
                reset_type=reset,
                use_state_discrimination=True,
                operation_gap_ns=gap_ns,
            )
            raw = swap_probe.acquire(machine, prog, sweep_axes,
                                     num_shots=shots,
                                     timeout=self.backend._timeout)
            per_pair.append(_member_states(raw["state"], high_side))

        state = xr.concat(per_pair, dim="qubit_pair").assign_coords(
            qubit_pair=list(self.params.targets))
        if self.params.readout_mode == "shot":
            return state.to_dataset(name="state")
        jp = states_to_joint_population(state, member_dim="member",
                                        shot_dim="shot_idx")
        return jp.to_dataset()
