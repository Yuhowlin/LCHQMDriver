"""QM residual-ZZ vs coupler bias for scqo - supplies ``probe()`` + the raw
joint-state reduction.

Parameters, the per-bias echo-fringe fit and the writeback (the decouple point
as ``idle_flux`` on the COUPLER MODE's own flux channel, plus the residual
``zz_hz`` fact on the pair) are inherited from
``scqo.experiments.PairZZCoupler``. scqo sweeps
``(coupler_bias_v, idle_time_ns)``; the LCHQM probe sweeps ``amplitudes`` (V on
the pair's tunable coupler) x ``durations`` (interaction time, clock cycles) with
a Hahn echo + virtual detuning on ONE pair member and joint two-qubit state
readout. The neutral ``measure`` role (high/low, roster-declared) is mapped onto
the vendor's control/target here; ``reduce_raw`` turns the joint populations into
the canonical ``signal`` (the measured qubit's excited-state probability).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import PairZZCoupler


@register
class QMPairZZCoupler(PairZZCoupler):
    """Build the multiplexed ZZ-vs-coupler QUA program on the QM OPX (QCQ pairs)."""

    def _measure_side(self, machine: Any) -> str:
        """Map the neutral ``measure`` role (high/low) onto vendor control/target.

        The mapping itself is shared with the pair swap maps — see
        ``_vendor.role_side`` for why the roster's declared roles are the only
        source and why a mixed mapping refuses. ``machine`` is accepted (and
        unused) so the call site reads like the probe helpers around it."""
        from ._vendor import role_side

        return role_side(self, self.params.measure, field="measure")

    def _vendor_pair(self, machine: Any, name: str) -> Any:
        """The QUAM qubit_pair behind a ROSTER composite name.

        ``machine`` is accepted (and unused) so the call site reads like the
        probe helpers around it; the resolution itself is the backend's, which
        joins roster composite -> QUAM pair by MEMBERSHIP."""
        from ._vendor import vendor_pair

        return vendor_pair(self, name)

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.probes import pair_qcq_zz_coupler_freq as zz_probe
        from scqo_qm.experiments._lib import select_qubit_pairs

        from ._vendor import vendor_pair_name

        machine = self.backend.machine  # type: ignore[attr-defined]
        # targets are ROSTER composite names; the probe helper selects by QUAM
        # pair key (QM names its pairs after the coupler), so translate first —
        # order preserved, which is what the axis relabel below relies on.
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)
        self._side = self._measure_side(machine)

        # Canonical idle times (ns) -> clock cycles; the raw time axis is the
        # QUANTIZED grid (durations*4 ns), which estimate() reads from coords.
        cycles = np.unique(np.clip(
            np.round(self.sweep_axes["idle_time_ns"] / 4).astype(int), 4, None))
        amplitudes = self.sweep_axes["coupler_bias_v"]

        prog, axes = zz_probe.build_program(
            machine,
            pairs,
            amplitudes=amplitudes,
            durations=cycles,
            detuning_hz=int(self.params.detuning_hz),
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            measure_qubit=self._side,
        )
        # The canonical time axis is the probe's REAL quantized grid: re-declare
        # it so sizes and values match the raw data exactly.
        self.sweep_axes["idle_time_ns"] = axes["time"].values.astype(float)
        sweep_axes = {
            # The probe labels its target axis with VENDOR pair keys; scqo's
            # dataset (and estimate()) key on the ROSTER composite names the
            # operator asked for. Same order by construction (vendor_names was
            # built from targets), so relabel rather than rename downstream.
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            "coupler_bias_v": axes["amp"],
            "idle_time_ns": axes["time"],
        }
        return prog, sweep_axes

    def reduce_raw(self, raw: xr.Dataset) -> xr.Dataset:
        """Joint two-qubit populations -> the measured qubit's excited-state
        probability (the canonical ``signal``). First digit = control."""
        if "state_ee" in raw.data_vars:
            sig = (raw["state_eg"] + raw["state_ee"] if self._side == "control"
                   else raw["state_ge"] + raw["state_ee"])
        else:  # IQ fallback (no state discrimination): fit the I fringe
            sig = raw["I_control"] if self._side == "control" else raw["I_target"]
        return sig.to_dataset(name="signal")
