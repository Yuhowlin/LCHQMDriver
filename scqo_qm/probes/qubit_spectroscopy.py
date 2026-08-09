"""Qubit-spectroscopy acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Sweep the qubit drive detuning while playing a (typically weak, long) saturation pulse and
reading out the resonator; the qubit line is fitted downstream.
"""

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    dfs,
    operation: str,
    operation_len,
    operation_amp: float,
    num_shots: int,
    reset_type: str,
    drive_qubit: Optional[str] = None,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the qubit-spectroscopy QUA program. Returns (program, sweep_axes).

    `dfs` is the drive-detuning sweep in Hz; `qubits` is a BatchableList (see
    `_lib.select_qubits`). When `drive_qubit` is None every qubit is driven;
    otherwise only that qubit plays the drive. `operation_len` (ns) overrides the
    operation's configured length when not None.
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
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
                        qubit.reset(reset_type, simulate, log_callable=log)

                    for i, qubit in multiplexed_qubits.items():
                        if drive_qubit is None or qubit.name == drive_qubit:
                            # Get the duration of the operation from the node parameters or the state
                            duration = operation_len if operation_len is not None else qubit.xy.operations[operation].length
                            # Update the qubit frequency
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                            # Play the saturation pulse
                            qubit.xy.play(
                                operation,
                                amplitude_scale=operation_amp,
                                duration=duration // 4,
                            )
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        # readout the resonator
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
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
