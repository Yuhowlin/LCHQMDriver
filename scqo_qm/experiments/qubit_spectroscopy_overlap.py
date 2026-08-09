"""QM concurrent-tone qubit spectroscopy for scqo - supplies only ``probe()``.

Parameters, peak fitting and the drive_freq_hz writeback are inherited from
``scqo.experiments.QubitSpectroscopyOverlap``; the sequence is
``scqo_qm/probes/qubit_spectroscopy_overlap.py`` (read its docstring - it
carries the FEM-core caveat that decides whether the tones really overlap).
scqo sweeps ``detuning_hz``; the probe builds the same sweep on coord
``detuning``, which the backend's ``_to_canonical`` renames back.

The timing is NOT computed here. ``scqo.experiments._overlap.overlap_windows``
resolves the tone length, the ADC lead and the drive length from the readout
channel's own knobs and refuses off-grid values, so this shell and the Qblox
probe cannot drift apart on what the same Parameters mean.

Drive power contract: unchanged from the sequential probe - the core ``run()``
already solved the drive chain for ``drive_power_dbm`` (recorded set -> acquire
-> revert), parking the exact amplitude on the saturation op, so the probe plays
it at ``amplitude_scale=1.0`` (exact in QUA fixed point).
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitSpectroscopyOverlap
from scqo.experiments._overlap import overlap_windows


@register
class QMQubitSpectroscopyOverlap(QubitSpectroscopyOverlap):
    """Build a multiplexed concurrent-tone spectroscopy QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_spectroscopy_overlap as spec_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        targets = list(self.params.targets)
        qubits = select_qubits(machine, targets, multiplexed=True)

        # One multiplexed program plays ONE set of times, so every target has to
        # agree on them. They can only differ through the per-target readout
        # knobs, and silently taking the first target's would put the others'
        # ADC in the wrong place — visible as a weaker, shifted peak on exactly
        # the qubits nobody was looking at.
        windows = {q: overlap_windows(self, q) for q in targets}
        distinct = {(w.tone_len_ns, w.acq_start_ns, w.drive_len_ns) for w in windows.values()}
        if len(distinct) > 1:
            detail = ", ".join(
                f"{q}: tone={w.tone_len_ns:g} ns, acq_start={w.acq_start_ns:g} ns, "
                f"drive={w.drive_len_ns:g} ns"
                for q, w in windows.items()
            )
            raise ValueError(
                f"qubit_spectroscopy_overlap: the targets resolve to different "
                f"concurrent-tone windows, which one multiplexed program cannot "
                f"play ({detail}). They differ through readout_duration_s / "
                f"readout_integration_s — equalize those, or run the targets in "
                f"separate runs."
            )
        window = windows[targets[0]]

        return spec_probe.build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            operation="saturation",
            drive_len_ns=window.drive_len_ns,
            operation_amp=1.0,  # run() parked the exact amplitude on the saturation op
            acq_lead_ns=window.acq_start_ns,
            ro_operation="readout",
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
        )
