"""ZZ-interaction acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Hahn-echo sequence on a detector qubit while the source qubit is echoed in parallel; the ZZ
crosstalk shows up as an idle-time-dependent phase, fitted downstream (T2_echo).
"""

from typing import Callable, Optional

import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    idle_times_cycles,
    source_qubit: str,
    detector_qubit: str,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    simulate: bool = False,
):
    """Build the ZZ-interaction QUA program. Returns (program, sweep_axes).

    `idle_times_cycles` is the per-arm idle-time sweep in clock cycles (4 ns); the echo
    has two idle arms, so the reported idle-time axis is 8 * idle_times_cycles ns.
    `source_qubit` / `detector_qubit` are looked up on `machine`; `qubits` is a
    BatchableList (see `_lib.select_qubits`).
    """
    qubit_source = machine.qubits[source_qubit]
    qubit_detector = machine.qubits[detector_qubit]
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(8 * idle_times_cycles, attrs={"long_name": "idle time", "units": "ns"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        shot = declare(int)
        t = declare(int)

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            for i, qubit in multiplexed_qubits.items():
                with for_(shot, 0, shot < num_shots, shot + 1):
                    save(shot, n_st)

                    with for_each_(t, idle_times_cycles):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            reset_frame(qubit.xy.name)
                            qubit.reset(reset_type, simulate)
                        align()

                        # Qubit manipulation
                        qubit_detector.xy.play("x90")
                        # qubit_source.xy.play("x90", amplitude_scale=0)
                        align()
                        qubit_detector.xy.wait(t)
                        qubit_source.xy.wait(t)
                        # align()
                        qubit_detector.xy.play("x180")
                        qubit_source.xy.play("x180")
                        # align()
                        qubit_detector.xy.wait(t)
                        qubit_source.xy.wait(t)

                        qubit_detector.xy.play("-x90")
                        align()

                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            # Measure the state of the resonators
                            if use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                # save data
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

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
