"""Ramsey acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Virtual-detuning Ramsey: y90 -> idle -> x90 with the drive virtually detuned by
`detuning_hz` via a frame rotation proportional to the idle time.
"""

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    idle_times_cycles,
    detuning_hz: int,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Ramsey QUA program. Returns (program, sweep_axes).

    `idle_times_cycles` is the idle-time sweep in clock cycles (4 ns);
    `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    # The QM clock resolves idle times to 4 ns. A linear sweep finer than that
    # collapses adjacent points onto the same clock cycle, leaving duplicate
    # idle times (the log sweep already de-duplicates via np.unique). Collapse
    # them here so the idle-time axis is strictly increasing and the QUA buffer
    # length, the stream processing and the dataset coordinate stay consistent --
    # a zero-width first step otherwise breaks the downstream FFT-based fit.
    idle_times_cycles = np.unique(np.asarray(idle_times_cycles))

    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(4 * idle_times_cycles, attrs={"long_name": "idle times", "units": "ns"}),
    }
    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        idle_time = declare(int)
        virtual_detuning_phases = [declare(fixed) for _ in range(num_qubits)]

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

                with for_each_(idle_time, idle_times_cycles):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log,
                                    max_attempts=reset_max_attempts)
                    align()
                    # Qubit manipulation
                    for i, qubit in multiplexed_qubits.items():
                        # The ramp is NEGATED on purpose -- this is not a typo. The
                        # second pi/2's phase must run BACKWARD relative to the free
                        # precession of a qubit sitting above its drive, so that the
                        # observed fringe is (applied + err) -- the quantity scqo's
                        # shared estimate() subtracts `applied` from. Get this wrong and
                        # every accepted update DOUBLES the residual detuning instead of
                        # cancelling it (the fit still looks clean).
                        #
                        # This is the SAME sign the Qblox backend needs, for the same
                        # reason. An earlier version of this comment claimed a frame
                        # rotation is "the opposite handedness to a pulse-axis phase" --
                        # that is FALSE (frame_rotation_2pi documents its argument as
                        # "the angle to add to the current phase", and applies
                        # envelope x e^{+i phi}, exactly like a pulse phase). The sign
                        # was right, the stated reason was not; do not propagate it.
                        assign(
                            virtual_detuning_phases[i],
                            Cast.mul_fixed_by_int(-detuning_hz * 1e-9, 4 * idle_time),
                        )

                        qubit.xy.play("y90")
                        qubit.xy.frame_rotation_2pi(virtual_detuning_phases[i])
                        qubit.wait(idle_time)
                        qubit.xy.play("x90")

                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if use_state_discrimination:
                            qubit.readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if use_state_discrimination:
                    state_st[i].buffer(len(idle_times_cycles)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(idle_times_cycles)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(idle_times_cycles)).average().save(f"Q{i + 1}")

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
