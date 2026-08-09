"""Concurrent-tone qubit-spectroscopy acquisition probe: vendor code only
(qm/quam) - no qualibrate, no scqo, no scqat.

Two-tone spectroscopy with the saturation drive and the readout tone playing
TOGETHER, and the acquisition opening only after both have been on a while:

    t = 0                                     acq_lead_ns
     |                                             |
     v                                             v
     ############## readout tone ###########################
     [========= saturation drive =========]
                                           [==== ADC ====]

WHAT MAKES THE TONES OVERLAP: there is NO `align()` between the qubit drive and
the measurement. Both element timelines start from the one shared `align()`
above them, and neither is given a `wait`, so the drive and the readout tone go
up on the same edge. (`qubit_spectroscopy.py`, the sequential sibling, has an
`align()` right there, which is the entire difference between the two probes.)

WHAT MAKES THE ADC OPEN LATE: QUAM's `measure()` takes no acquisition-delay
argument - the ADC follows its own pulse by `resonator.time_of_flight` and
nothing else - so the lead is a PRE-TONE instead. The same readout operation is
played for `acq_lead_ns` immediately before `measure()`, contiguous on the
resonator's timeline, so the resonator sees one seamless tone of
`acq_lead_ns + readout_len` and the ADC integrates only its tail. `time_of_flight`
is untouched: it is the cable, not the lead.

NOTE (a SILENT failure): the qubit drive (`xy`) and its `resonator` must be on
DIFFERENT QM cores/threads for the tones to overlap at all. Same-core elements
are SERIALIZED - the drive plays to completion first, you get the sequential
experiment back, and the fit still converges and looks clean. Check the analog
traces in the OPX simulator before trusting a result from a new wiring, and if
they are serialized, repartition the FEM rather than working around it here.

Times are given in ns and must be multiples of 4 (the QUA clock cycle); scqo's
`experiments/_overlap.py` is what guarantees that on the scqo path.

QM concurrent-tone qubit spectroscopy for scqo - supplies only ``probe()``.

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

from typing import Callable, Optional

import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def _cycles(name: str, value_ns) -> int:
    """ns -> QUA clock cycles, refusing anything off the 4 ns grid.

    Refused rather than rounded: this probe is called by two shells, and a
    silent round here would make the same requested timing mean different
    things depending on who asked.
    """
    value_ns = int(round(float(value_ns)))
    if value_ns % 4:
        raise ValueError(
            f"{name}={value_ns} ns is not a multiple of the 4 ns QUA clock cycle"
        )
    return value_ns // 4


def build_program(
    machine,
    qubits,
    *,
    dfs,
    operation: str,
    drive_len_ns,
    operation_amp: float,
    acq_lead_ns,
    ro_operation: str,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the concurrent-tone qubit-spectroscopy QUA program.

    Returns ``(program, sweep_axes)``.

    `dfs` is the drive-detuning sweep in Hz; `qubits` is a BatchableList (see
    `_lib.select_qubits`). `drive_len_ns` is the saturation length measured from
    the shared tone onset and `acq_lead_ns` is how long the readout tone runs
    before the ADC opens (0 = the ADC opens with the tone, which is the
    sequential probe's behaviour). Both are ns, both multiples of 4.
    """
    num_qubits = len(qubits)
    drive_cycles = _cycles("drive_len_ns", drive_len_ns)
    lead_cycles = _cycles("acq_lead_ns", acq_lead_ns)
    if drive_cycles < 1:
        raise ValueError(f"drive_len_ns={drive_len_ns} ns is shorter than one clock cycle")

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit drive detuning", "units": "Hz"}),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        # Update the qubit frequency while the drive is still off
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        # Wait for the qubit to decay to the ground state. Unlike the
                        # sequential probe this really IS a state reset: the drive
                        # comes up later, with the tone, so nothing is driving here.
                        qubit.reset(reset_type, simulate, log_callable=log)

                    # THE shared t = 0. Every element below starts from this edge
                    # and none of them waits, which is what makes the two tones
                    # concurrent - do not add an align() inside the block.
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        if lead_cycles:
                            # the ADC lead: the same readout operation, played
                            # back-to-back into measure() so the resonator sees
                            # one seamless tone and only its tail is integrated
                            qubit.resonator.play(ro_operation, duration=lead_cycles)
                        # the saturation drive, on its own element's timeline
                        qubit.xy.play(
                            operation,
                            amplitude_scale=operation_amp,
                            duration=drive_cycles,
                        )
                        # readout the resonator (the ADC opens here, acq_lead_ns
                        # after the tone onset, plus the element's time_of_flight)
                        qubit.resonator.measure(ro_operation, qua_vars=(I[i], Q[i]))
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)


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

        return build_program(
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
