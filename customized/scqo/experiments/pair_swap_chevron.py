"""QM single-excitation swap chevron for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are
inherited from ``scqo.experiments.PairSwapChevron``; the joint-population
reduction comes from :class:`JointPopulationMixin`. scqo sweeps
``(flux_amp_v, swap_time_ns)`` in absolute volts and whole nanoseconds; the
LCHQM probe sweeps a QUA amplitude scale x a 1 ns-granular pulse duration
(baked below 4 ns), so this adapter drives the probe in its ``amp_mode
="absolute"`` mode and maps the neutral high/low roles onto the vendor's
control/target.

Unlike every other QM adapter here, ``probe()`` ACQUIRES and returns a ready
``xr.Dataset``: the program only runs against the probe's own baked config
(which carries the ``flux_pulse1..16`` operations), and the backend's shared
fetch path would regenerate a config without them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import PairSwapChevron

from ._pair_roles import JointPopulationMixin


@register
class QMPairSwapChevron(JointPopulationMixin, PairSwapChevron):
    """Build, run and fetch the multiplexed swap chevron on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the sub-17 ns "
                           "branch and fetches against it inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair_name
        from customized.probes import pair_qq_chevron as chevron_probe
        from customized.probes._lib import select_qubit_pairs

        machine = self.backend.machine  # type: ignore[attr-defined]
        # targets are ROSTER composite names; the probe helper selects by QUAM
        # pair key (QM names its pairs after the coupler), so translate first —
        # order preserved, which is what the axis relabel below relies on.
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)

        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time. `high` fixes the reduce_raw
        # orientation; the two selectors are independent of each other.
        self._high_side = role_side(self, "high", field="targets")
        drive_role = role_side(self, self.params.drive_side, field="drive_side")
        flux_role = role_side(self, self.params.flux_side, field="flux_side",
                              needs_flux=True)

        # The probe's `times_cycles` argument is ns despite its name (it plays
        # 1..16 ns baked segments below the 4 ns clock), and it must be an
        # integer array — from_array loops it into an int QUA variable.
        times_ns = np.round(self.sweep_axes["swap_time_ns"]).astype(int)
        prog, axes, baked_config = chevron_probe.build_program(
            machine,
            pairs,
            amplitudes=self.sweep_axes["flux_amp_v"],
            times_cycles=times_ns,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            amp_mode="absolute",
            flux_role=flux_role,
            drive_role=drive_role,
        )
        # The canonical time axis is the probe's REAL grid: re-declare it so
        # sizes and values match the raw data exactly.
        self.sweep_axes["swap_time_ns"] = axes["time"].values.astype(float)
        sweep_axes = {
            # The probe labels its target axis with VENDOR pair keys; scqo's
            # dataset (and estimate()) key on the ROSTER composite names the
            # operator asked for. Same order by construction (vendor_names was
            # built from targets), so relabel rather than rename downstream.
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            "flux_amp_v": axes["amplitude"],
            "swap_time_ns": axes["time"],
        }
        # Acquire here: the baked config is per-call and cannot be reached
        # through the backend's (program, axes, module) shape.
        shots = getattr(self.params, "num_averages", None) or getattr(
            self.params, "num_shots", 1)
        return chevron_probe.acquire(
            machine, prog, sweep_axes,
            num_shots=int(shots), timeout=self.backend._timeout,
            config=baked_config,
        )
