"""T2 echo (Hahn) acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Echo sequence x90 - wait(t) - x180 - wait(t) - x90 on each qubit: the central pi
pulse refocuses quasi-static dephasing, so the envelope decays with T2_echo.
Sequence mirrors the vendored official ``calibrations/06b_echo.py`` core (which
ends in -x90; the final +x90 here refocuses to |1> instead of |0>, matching the
scqo physics half "X90 - tau/2 - X - tau/2 - X90" - the downstream exponential
fit is sign-aware either way).
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
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the T2-echo QUA program. Returns (program, sweep_axes).

    `idle_times_cycles` is the PER-ARM idle-time sweep in clock cycles (4 ns); the
    echo has two idle arms, so the reported idle-time axis is 8 * idle_times_cycles
    ns (total idle time tau, as in the official 06b_echo node's 2 * 4 * t).
    `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    # Collapse sub-clock duplicates so the buffer length matches the axis (see ramsey).
    idle_times_cycles = np.unique(np.asarray(idle_times_cycles))

    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(
            8 * idle_times_cycles, attrs={"long_name": "total echo idle time", "units": "ns"}
        ),
    }
    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        t = declare(int)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                with for_each_(t, idle_times_cycles):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log,
                                    max_attempts=reset_max_attempts)
                    align()
                    # Echo: x90 - tau/2 - x180 - tau/2 - x90 (official 06b_echo core)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()
                        qubit.xy.play("x90")
                        qubit.xy.wait(t)
                        qubit.xy.play("x180")
                        qubit.xy.wait(t)
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
