"""Resonator-spectroscopy acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

1D resonator spectroscopy: sweep the readout intermediate frequency around each
resonator's current IF, measure the demodulated I/Q, and let the |IQ| dip locate
the resonance. No qubit reset - the resonator is measured directly.
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
    num_shots: int,
):
    """Build the resonator-spectroscopy QUA program. Returns (program, sweep_axes).

    `dfs` is the readout-frequency detuning sweep in Hz (relative to each
    resonator's current IF); `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)  # QUA variable for the readout frequency

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        rr = qubit.resonator
                        # Update the resonator frequencies for all resonators
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # Measure the resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait for the resonator to deplete
                        rr.wait(rr.depletion_time * u.ns)
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
