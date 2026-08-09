"""Readout-frequency acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Single-shot readout vs readout-frequency detuning: for each detuning the qubit is
prepared in |0> and |1> and measured, giving the per-frequency IQ blobs that the
fidelity estimator scans for the optimal readout frequency.
"""

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    dfs,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
):
    """Build the readout-frequency QUA program. Returns (program, sweep_axes).

    `dfs` is the readout-frequency detuning sweep in Hz (relative to the current
    resonator IF); `qubits` is a BatchableList (see `_lib.select_qubits`).
    Always acquires I/Q single shots (no state-discrimination branch).
    """
    num_qubits = len(qubits)
    n_runs = num_shots

    prepared_states = [0, 1]
    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shot_idx": xr.DataArray(np.arange(1, n_runs + 1), attrs={"long_name": "number of shots"}),
        "frequency": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
        "prepared_state": xr.DataArray(prepared_states, attrs={"long_name": "prepared qubit state", "units": ""}),
    }
    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)
        ps = declare(int)
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                # ground iq blobs for all qubits
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_each_(ps, prepared_states):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate)
                        align()

                        # Change qubit state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.align()

                            with switch_(ps):
                                with case_(0):
                                    pass
                                with case_(1):
                                    qubit.xy.play("x180")

                            qubit.align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            qubit.align()
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(prepared_states)).buffer(len(dfs)).buffer(n_runs).save(f"I{i + 1}")
                Q_st[i].buffer(len(prepared_states)).buffer(len(dfs)).buffer(n_runs).save(f"Q{i + 1}")

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
