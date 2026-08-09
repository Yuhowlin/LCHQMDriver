"""Time-dependent readout-resonator photon acquisition probe: vendor code only
(qm/quam/qualang_tools) - no qualibrate, no scqo, no scqat.

AC-Stark / photon-number-vs-time measurement. A readout-resonator test pulse
populates the resonator; the qubit is then probed by a saturation/x180 pulse at a
swept delay after the test pulse, while the drive frequency is swept across the
qubit line. Locating the AC-Stark-shifted qubit line per delay traces the
resonator photon number filling up and ringing down in time.

The sweep is `detuning` x `delay_time`; the delay is stepped in clock cycles
(4 ns granularity). `delay_time` is the SIGNED probe delay relative to the
cavity-drive ONSET: positive = the qubit probe fires after the drive starts
(cavity filling / ring-down), negative = the probe fires *before* the drive. The
resonator drive is anchored at a fixed reference and only the probe moves; a common
non-negative `offset` lead is prepended to both elements so the most-negative delay
still yields a legal (>=4-cycle) wait, and because both elements share that lead the
probe lands *exactly* `delay_time` ns from the drive (the lead cancels between them).
Both qubits are read out as raw I/Q (state discrimination saves the discriminated
state instead). Downstream this is fit by scqat's `ReadoutPulsePhotonEstimator`,
which expects the `detuning`/`delay_time` coords (negative delays are fine).

Note: the qubit drive (`xy`) and its `resonator` must be on DIFFERENT QM cores/threads
for the probe to overlap the drive; same-core elements are serialized (the drive plays
to completion first).
"""

from typing import Callable, Optional

import numpy as np
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
    delay_time_array,
    num_shots: int,
    reset_type: str,
    xy_operation: str,
    xy_operation_amplitude_factor: float,
    ro_operation: str,
    test_operation: str,
    rr_depletion_time: Optional[int],
    use_state_discrimination: bool,
    multiplexed: bool,
    simulate: bool = False,
):
    """Build the time-dependent readout-resonator photon QUA program.

    Returns ``(program, sweep_axes)``.

    `dfs` is the qubit detuning sweep in Hz; `delay_time_array` is the SIGNED XY-probe
    delay sweep in ns (multiples of 4 ns, relative to the resonator drive onset; may be
    negative); `qubits` is a BatchableList (see `_lib.select_qubits`). The delay is
    looped in clock cycles derived internally as ``delay_time_array // 4``.
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)

    delay_tick_array = (np.asarray(delay_time_array) // 4).astype(int)
    # Common non-negative lead so the most-negative delay still gives a >=4-cycle wait.
    # Both the drive and the probe share this lead, so it cancels and the probe lands
    # exactly delay_time relative to the drive onset.
    offset_tick = max(0, -int(delay_tick_array.min()))

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "delay_time": xr.DataArray(delay_time_array, attrs={"long_name": "delay time", "units": "ns"}),
    }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the qubit frequency
        delay_tick = declare(int)  # QUA variable for the XY-probe delay (in clock cycles)
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_(*from_array(delay_tick, delay_tick_array)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            # Update the qubit frequency
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                            # Wait for the qubits to decay to the ground state
                            qubit.reset(reset_type, simulate)

                        align()

                        # Qubit manipulation: anchor the cavity drive at a fixed reference and probe
                        # the qubit at a SIGNED delay relative to the drive onset (QUA elements run in
                        # parallel from the shared align() t=0). Both waits share the offset_tick lead,
                        # so xy_start - resonator_start = delay_tick (negative => probe before drive).
                        for i, qubit in multiplexed_qubits.items():
                            # Cavity drive at a fixed reference: t = (offset_tick + 4) cycles.
                            qubit.resonator.wait(offset_tick + 4)
                            qubit.resonator.play(test_operation)
                            # Probe delay_tick cycles relative to the drive onset. The min xy wait is
                            # min(delay_tick) + offset_tick + 4 = 4 cycles, satisfying the QUA floor.
                            qubit.xy.wait(delay_tick + offset_tick + 4)
                            qubit.xy.play(xy_operation, amplitude_scale=xy_operation_amplitude_factor)

                            # Let the test-pulse photons decay before the discriminating readout.
                            if rr_depletion_time is not None:
                                qubit.resonator.wait(rr_depletion_time * u.ns // 4)
                            else:
                                qubit.resonator.wait(qubit.resonator.depletion_time * u.ns // 4)
                        align()

                        # Measure the state of the resonators
                        for i, qubit in multiplexed_qubits.items():
                            if use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure(ro_operation, qua_vars=(I[i], Q[i]))
                                # save data
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

            # Measure sequentially
            if not multiplexed:
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if use_state_discrimination:
                    state_st[i].buffer(len(delay_time_array)).buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(delay_time_array)).buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(delay_time_array)).buffer(len(dfs)).average().save(f"Q{i + 1}")

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
