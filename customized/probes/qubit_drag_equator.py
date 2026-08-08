"""DRAG Equator (3-Line) calibration acquisition probe: vendor code only (qm/quam)."""

from typing import Callable, Optional, List
import numpy as np
import xarray as xr
from qm.qua import *

from customized.probes._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    num_shots: int,
    beta_array: List[float],
    pulse_repetitions: int,
    reset_type: str,
    use_state_discrimination: bool,
    target_gate: str = "x180",
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the DRAG equator QUA program. Returns (program, sweep_axes)."""
    num_qubits = len(qubits)
    from customized import quam_fields

    alpha_array = np.asarray(beta_array, dtype=float)

    # same normalization as scqo's _gate_target (probes must not import scqo)
    op_name = "x90" if str(target_gate).strip().lower() == "x90" else "x180"

    ref_alpha = float(np.max(np.abs(alpha_array)))
    if ref_alpha < 1e-6:
        ref_alpha = 1.0
    scale_array = alpha_array / ref_alpha  # values in [-1, 1] ✓

    # Save originals and install ref_alpha into DragCosine operation
    orig_alphas: dict = {}
    for q_name in qubits.get_names():
        q_obj = machine.qubits[q_name]
        orig_alphas[q_name] = quam_fields.get_drag_beta(q_obj, operation=op_name)
        quam_fields.set_drag_beta(q_obj, ref_alpha, operation=op_name)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "seq_idx": xr.DataArray([0, 1], attrs={"long_name": "sequence index"}),
        "beta": xr.DataArray(alpha_array, attrs={"long_name": "DRAG alpha coefficient", "units": ""}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # DRAG alpha scale factor
        seq = declare(int)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(seq, [0, 1]):
                    with for_each_(a, scale_array):
                        # Qubit initialization
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                        align()

                        # Play sequence
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.align()
                            # seq 0: Rx(pi) - Ry(pi/2)  (or two x90 - Ry(pi/2))
                            # seq 1: Ry(pi) - Rx(pi/2)  (or two y90 - Rx(pi/2))
                            with if_(seq == 0):
                                if op_name == "x90":
                                    play("x90" * amp(1, 0, 0, a), qubit.xy.name)
                                    play("x90" * amp(1, 0, 0, a), qubit.xy.name)
                                    play("y90" * amp(a, 0, 0, 1), qubit.xy.name)
                                else:
                                    play("x180" * amp(1, 0, 0, a), qubit.xy.name)
                                    play("y90" * amp(a, 0, 0, 1), qubit.xy.name)
                            with else_():
                                if op_name == "x90":
                                    play("y90" * amp(a, 0, 0, 1), qubit.xy.name)
                                    play("y90" * amp(a, 0, 0, 1), qubit.xy.name)
                                    play("x90" * amp(1, 0, 0, a), qubit.xy.name)
                                else:
                                    play("y180" * amp(a, 0, 0, 1), qubit.xy.name)
                                    play("x90" * amp(1, 0, 0, a), qubit.xy.name)

                            qubit.align()

                        # Measurement
                        for i_q, qubit in multiplexed_qubits.items():
                            if use_state_discrimination:
                                qubit.readout_state(state[i_q])
                                save(state[i_q], state_st[i_q])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i_q], Q[i_q]))
                                save(I[i_q], I_st[i_q])
                                save(Q[i_q], Q_st[i_q])
                        align()

        with stream_processing():
            n_st.save("n")
            for i_q in range(num_qubits):
                if use_state_discrimination:
                    # state is int (0/1); save into I slot; Q slot is unused but must exist
                    state_st[i_q].buffer(len(beta_array)).buffer(2).average().save(f"I{i_q + 1}")
                else:
                    I_st[i_q].buffer(len(beta_array)).buffer(2).average().save(f"I{i_q + 1}")
                    Q_st[i_q].buffer(len(beta_array)).buffer(2).average().save(f"Q{i_q + 1}")

    # Generate the QM config while ref_alpha is still active in QUAM
    config = machine.generate_config()

    # Restore original QUAM alpha values
    for q_name, orig_alpha in orig_alphas.items():
        quam_fields.set_drag_beta(machine.qubits[q_name], orig_alpha, operation=op_name)

    return prog, sweep_axes, config



def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config=None,
) -> xr.Dataset:
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)
