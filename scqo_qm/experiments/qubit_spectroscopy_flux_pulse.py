"""QM qubit spectroscopy vs PULSED flux for scqo - supplies only ``probe()``.

Parameters, the transmon-arch fit and reporting are inherited from
``scqo.experiments.QubitSpectroscopyFluxPulse``. scqo sweeps ``(flux_bias_v,
detuning_hz)``; the LCHQM probe sweeps ``dcs`` (flux, V) and ``dfs`` (drive
detuning, Hz) with each qubit fluxing/driving its OWN lines (z/xy source = None,
as in the official 03b node).

PULSE CONTRACT: this probe conforms to the ``_pulse`` name by construction — the
flux is a z PULSE played only alongside the saturation drive
(``scqo_qm/probes/qubit_spectroscopy_flux.py``: ``qubit.z.play(...,
duration=operation_duration)`` then ``align()`` then ``measure``), so every
readout happens at idle flux and the neutral ``estimate()`` reduces the map
against ONE global IQ reference. The DAC adds that pulse to the standing offset
``initialize_qpu`` applies, which is what makes ``flux_bias_v`` an excursion FROM
``idle_flux`` (0 V = stay parked) and why the neutral ``estimate()``
re-references the fitted sweet spot to an absolute set-point before writing
``flux_offset``. ``flux_point`` is passed explicitly so the bias the probe
rail-validated is the bias that plays. The probe MODULE keeps its historical name
(it is shared with the qualibrate shell path).

AXIS-ORDER NOTE (do not "fix" the order below): the probe's QUA loops nest df
(outer) over dc (inner) and its streams are ``buffer(len(dcs)).buffer(len(dfs))``,
so the raw per-qubit array is shaped (detuning, flux) — the OPPOSITE of scqo's
declared sweep order (flux_bias_v, detuning_hz). A positional rename in
``_to_canonical`` would therefore swap the axes (silently, whenever the two sweep
lengths happen to be equal). Instead we re-key the probe's sweep_axes with the
canonical scqo names IN RAW NESTING ORDER; ``_to_canonical`` then takes its
name-based path (no rename, per-name size checks) and ``estimate()`` transposes
by name.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitSpectroscopyFluxPulse


@register
class QMQubitSpectroscopyFluxPulse(QubitSpectroscopyFluxPulse):
    """Build a multiplexed 2D (pulsed-flux x detuning) spectroscopy QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_spectroscopy_flux as flux_probe
        from scqo_qm.quam_fields import GOVERNED_FLUX_POINT

        from ._vendor import flux_source_name

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # A foreign flux source is a ROSTER entity; this probe plays the flux as
        # a z PULSE on a QUAM qubit, so _vendor refuses a coupler here with the
        # reason (qubit_only) instead of failing inside the QUA build.
        z_source = (None if self.params.flux_component is None
                    else flux_source_name(self, self.params.flux_component,
                                          qubit_only=True))

        prog, axes = flux_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            dcs=self.sweep_axes["flux_bias_v"],
            operation="saturation",
            operation_len=None,  # use each qubit's own saturation-pulse length
            operation_amp=1.0,
            num_shots=self.params.num_averages,
            z_source_qubit=z_source,
            xy_source_qubit=None,  # None = every measured qubit drives its own xy line
            flux_point=GOVERNED_FLUX_POINT,
        )
        # Canonical names in RAW nesting order (detuning outer, flux inner — see
        # module docstring); the DataArray values (incl. units attrs) are reused.
        sweep_axes = {
            "qubit": axes["qubit"],
            "detuning_hz": axes["detuning"],
            "flux_bias_v": axes["flux_bias"],
        }
        return prog, sweep_axes
