"""Resonator-spectroscopy-vs-power acquisition probe: vendor code only (qm/quam) -
no qualibrate, no scqo, no scqat.

2-D resonator spectroscopy vs readout power: per readout-amplitude point, repeat a
fast sweep of the readout intermediate frequency around each resonator's current IF.
The |IQ| dip locates the resonance at every power. No qubit reset - the resonator is
measured directly.

Loop order (2026-07-14, user-decided; both scqo backends match): amplitude (outer)
-> averages (middle) -> frequency (INNER = fastest) — each power point repeats the
frequency sweep ``num_shots`` times, so the resonator only jumps power between the
slow outer steps. The acquired axis order is therefore (power, detuning). The caller
must have set each resonator's base output power to the sweep's max power *before*
generating the config (the qualibrate shell does this via ``tracked_updates``); this
probe only builds the amplitude/frequency sweep, keeping it framework-free.

``depletion_time_ns`` optionally overrides every resonator's configured
``depletion_time`` for the between-readout ring-down wait (None keeps the per-qubit
QUAM values) — it also covers the wrap-around high->low amplitude jump between
repetitions of the inner sweep.
"""

from typing import Callable, Optional

import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.units import unit

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    dfs,
    amps,
    power_dbm,
    num_shots: int,
    depletion_time_ns: Optional[float] = None,
):
    """Build the resonator-spectroscopy-vs-power QUA program. Returns (program, sweep_axes).

    `dfs` is the readout-frequency detuning sweep in Hz (relative to each
    resonator's current IF); `amps` is the readout-amplitude pre-factor sweep
    (dimensionless, within [-2, 2)); `power_dbm` is the matching readout-power axis
    in dB (same length as `amps`); `qubits` is a BatchableList (see
    `_lib.select_qubits`); `depletion_time_ns` overrides the resonators' configured
    depletion wait (None = per-qubit QUAM `depletion_time`).
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)
    n_avg = num_shots

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "power": xr.DataArray(power_dbm, attrs={"long_name": "readout power", "units": "dBm"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
        df = declare(int)  # QUA variable for the readout frequency detuning
        idx = declare(int)  # progress index over the outer amplitude loop

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            assign(idx, 0)
            with for_each_(a, amps):  # amplitude (outer, slow)
                # Save the amplitude-point counter for the progress bar
                save(idx, n_st)
                assign(idx, idx + 1)
                with for_(n, 0, n < n_avg, n + 1):  # averages (middle)
                    with for_(*from_array(df, dfs)):  # frequency (INNER = fastest sweep)
                        for i, qubit in multiplexed_qubits.items():
                            rr = qubit.resonator
                            # set the readout IF for this detuning point
                            rr.update_frequency(df + rr.intermediate_frequency)
                            # readout the resonator at the swept amplitude
                            rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                            # wait for the resonator to deplete (ring-down)
                            wait_ns = (
                                int(depletion_time_ns)
                                if depletion_time_ns is not None
                                else rr.depletion_time
                            )
                            rr.wait(wait_ns * u.ns)
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Saves arrive frequency-fastest: group the inner freq sweep, stack the
                # n_avg repeats, average over that (middle) axis, then stack powers ->
                # final shape (power, detuning).
                I_st[i].buffer(len(dfs)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(
                    len(amps)
                ).save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(n_avg).map(FUNCTIONS.average(0)).buffer(
                    len(amps)
                ).save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_detuning_points: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    The progress counter tracks the outer amplitude loop (``num_detuning_points``
    is kept as the arg name for the qualibrate shell's call site; it is only the
    progress-bar total).
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_detuning_points, timeout=timeout, log=log)
