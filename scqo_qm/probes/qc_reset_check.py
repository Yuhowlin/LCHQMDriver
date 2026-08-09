"""Reset-check acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Power-Rabi-shaped reset diagnostic. Sweep the qubit drive amplitude (as a pre-factor of the
current `x180` amplitude) to prepare a *continuum* of excited populations, then - after the
amplitude-modified drive - play the reset macro under test and read out. A working reset
flattens the post-reset population to ~0 (ground) across all amplitudes; residual population
exposes reset failure.

To make "success" self-evident in one run, each amplitude point is measured twice along a
2-value `reset` axis:
  - reset="off": init -> x180(a) -> readout            (baseline: full Rabi oscillation)
  - reset="on" : init -> x180(a) -> reset macro -> readout  (should be flat ~0)
so the node can overlay the two curves. There is no fit and no state writeback downstream.

The reset macro is invoked **bare** (`qubit.macros[reset_operation].apply()` with no extra
args), so the chosen macro must be callable that way - its amplitudes/pulses baked into the
QUAM macro definition (e.g. `ParametricReset`). If `drive_qubit` is set, only that qubit is
driven (and reset); all qubits are still read out.
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
    amps,
    operation: str,
    reset_operation: str,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    drive_qubit: Optional[str] = None,
    simulate: bool = False,
):
    """Build the reset-check QUA program. Returns (program, sweep_axes).

    `amps` is the amplitude pre-factor sweep (must be within [-2; 2)); each point is measured
    with the reset macro off then on (the innermost `reset` axis). `reset_operation` is the
    macro key applied on the driven qubit(s) in the reset="on" branch
    (`qubit.macros[reset_operation].apply()`). `qubits` is a BatchableList (see
    `_lib.select_qubits`).
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
        "reset": xr.DataArray(["off", "on"], attrs={"long_name": "reset macro"}),
    }

    def _readout_and_save(multiplexed_qubits, I, I_st, Q, Q_st, state, state_st):
        for i, qubit in multiplexed_qubits.items():
            if use_state_discrimination:
                qubit.readout_state(state[i])
                save(state[i], state_st[i])
            else:
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        else:
            state = state_st = None
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(a, amps)):
                    # ---- reset OFF (baseline): init -> x180(a) -> readout ----
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if drive_qubit is None or qubit.name == drive_qubit:
                            qubit.xy.play(operation, amplitude_scale=a)
                    align()
                    _readout_and_save(multiplexed_qubits, I, I_st, Q, Q_st, state, state_st)
                    align()

                    # ---- reset ON (macro under test): init -> x180(a) -> reset macro -> readout ----
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if drive_qubit is None or qubit.name == drive_qubit:
                            qubit.xy.play(operation, amplitude_scale=a)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        if drive_qubit is None or qubit.name == drive_qubit:
                            qubit.macros[reset_operation].apply()
                    align()
                    _readout_and_save(multiplexed_qubits, I, I_st, Q, Q_st, state, state_st)
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Each qubit saves OFF then ON per amplitude, so `reset` (len 2) is the
                # innermost buffered axis, then `amp_prefactor`.
                if use_state_discrimination:
                    state_st[i].buffer(2).buffer(len(amps)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(2).buffer(len(amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(2).buffer(len(amps)).average().save(f"Q{i + 1}")

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
