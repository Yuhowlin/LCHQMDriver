"""QM fixed-time coupler-flux x qubit-flux swap map for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.PairSwapFluxMap``; the joint-population reduction comes
from :class:`JointPopulationMixin`. scqo sweeps ``(qubit_flux_v,
coupler_flux_v)`` in absolute volts — which is exactly the LCHQM probe's own
``amp_mode="absolute"``, so the adapter's work is the role mapping, the neutral
pulse SHAPE -> QUAM operation name, and quantizing the fixed duration onto the
4 ns clock.

The probe's debug-only paths (``swap_via_macro``, ``amp_mode="prefactor"``) are
deliberately not reachable from here: they exist to isolate a vendor macro
against a direct play, which is a qualibrate-side question.
"""

from __future__ import annotations

from typing import Any

import xarray as xr
from scqo import register
from scqo.experiments import PairSwapFluxMap

from ._pair_roles import JointPopulationMixin

#: neutral waveform shape -> the QUAM operation name it is played as. The op
#: must exist on BOTH the coupler and the selected member's z line; register
#: the shaped one with `python quam_config/register_flattop_cosine.py` before
#: selecting it (the probe refuses with the available list if it is missing).
_SHAPE_OPS = {"square": "const", "flattop_cosine": "flattop_cosine"}

#: QM plays flux pulses on a 4 ns clock, with a 16 ns floor (4 cycles).
_CLOCK_NS = 4
_MIN_FLUX_NS = 16


@register
class QMPairSwapFluxMap(JointPopulationMixin, PairSwapFluxMap):
    """Build the multiplexed fixed-time 2D swap map on the QM OPX (QCQ pairs)."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair_name
        from scqo_qm.probes import pair_qcq_fixed_time as map_probe
        from scqo_qm.experiments._lib import select_qubit_pairs

        machine = self.backend.machine  # type: ignore[attr-defined]
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)

        self._high_side = role_side(self, "high", field="targets")
        drive_role = role_side(self, self.params.drive_side, field="drive_side")
        flux_role = role_side(self, self.params.flux_side, field="flux_side",
                              needs_flux=True)

        operation = _SHAPE_OPS[self.params.flux_pulse_shape]
        flux_time = None
        if self.params.swap_time_ns is not None:
            flux_time = max(
                _MIN_FLUX_NS,
                int(round(self.params.swap_time_ns / _CLOCK_NS)) * _CLOCK_NS)
        # what the instrument ACTUALLY plays — estimate() records this, not the
        # unquantized request (None = each operation's own native length).
        self._flux_time_ns = float(flux_time) if flux_time is not None else None

        prog, axes = map_probe.build_program(
            machine,
            pairs,
            coupler_amplitudes=self.sweep_axes["coupler_flux_v"],
            qubit_amplitudes=self.sweep_axes["qubit_flux_v"],
            coupler_operation=operation,
            qubit_operation=operation,
            flux_time=flux_time,
            flux_role=flux_role,
            amp_mode="absolute",
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            drive_role=drive_role,
        )
        return prog, {
            # roster composite names, not the vendor pair keys the probe used
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            # raw nesting order: the qubit amplitude is the OUTER loop
            "qubit_flux_v": axes["qubit_amplitude"],
            "coupler_flux_v": axes["coupler_amplitude"],
        }
