"""Single Qubit Randomized Benchmarking (SQRB) acquisition probe (QM/QUAM).

Ported from 11a_single_qubit_randomized_benchmarking.py.
Uses FPGA-side Random sequence generation with Cayley lookup table and 
in-place recovery gate substitution for truncated depths.
"""

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire

# Recovery gates lookup: inv_gates[i] is the Clifford gate index that resets state i to 0
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


def play_clifford(sequence_list, depth_var, qubit):
    """Play sequence of Clifford gates from index 0 up to depth_var (inclusive)."""
    i = declare(int)
    x180_len = qubit.xy.operations["x180"].length
    idle_len = max(x180_len // 4, 1)

    with for_(i, 0, i <= depth_var, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(idle_len)
            with case_(1):
                qubit.xy.play("x180")
            with case_(2):
                qubit.xy.play("y180")
            with case_(3):
                qubit.xy.play("y180")
                qubit.xy.play("x180")
            with case_(4):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
            with case_(5):
                qubit.xy.play("x90")
                qubit.xy.play("-y90")
            with case_(6):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
            with case_(7):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
            with case_(8):
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(9):
                qubit.xy.play("y90")
                qubit.xy.play("-x90")
            with case_(10):
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(11):
                qubit.xy.play("-y90")
                qubit.xy.play("-x90")
            with case_(12):
                qubit.xy.play("x90")
            with case_(13):
                qubit.xy.play("-x90")
            with case_(14):
                qubit.xy.play("y90")
            with case_(15):
                qubit.xy.play("-y90")
            with case_(16):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(17):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(18):
                qubit.xy.play("x180")
                qubit.xy.play("y90")
            with case_(19):
                qubit.xy.play("x180")
                qubit.xy.play("-y90")
            with case_(20):
                qubit.xy.play("y180")
                qubit.xy.play("x90")
            with case_(21):
                qubit.xy.play("y180")
                qubit.xy.play("-x90")
            with case_(22):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(23):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("-x90")


def build_program(
    machine,
    qubits,
    *,
    depths: list[int],
    num_sequences: int,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool = False,
    seed: int | None = None,
    simulate: bool = False,
    strict_timing: bool = False,
):
    """Build the Single Qubit Randomized Benchmarking QUA program."""
    num_qubits = len(qubits)
    depth_arr = np.asarray(depths, dtype=int)
    max_circuit_depth = int(depth_arr.max())
    seq_arr = np.arange(num_sequences, dtype=int)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "sequence_idx": xr.DataArray(seq_arr, attrs={"long_name": "sequence index"}),
        "depth": xr.DataArray(depth_arr, attrs={"long_name": "Clifford depth"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        depth_var = declare(int)
        saved_gate = declare(int)
        m = declare(int)
        m_st = declare_stream()

        # FPGA random generator & lookup tables
        cayley = declare(int, value=c1_table.flatten().tolist())
        inv_list = declare(int, value=inv_gates)
        current_state = declare(int)
        step = declare(int)
        sequence_list = declare(int, size=max_circuit_depth + 1)
        inv_gate_list = declare(int, size=max_circuit_depth + 1)
        i_gen = declare(int)
        rand = Random(seed=seed if seed is not None else 42)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            # Outer loop over random sequences
            with for_(m, 0, m < num_sequences, m + 1):
                save(m, m_st)

                # Generate random Clifford sequence on FPGA
                assign(current_state, 0)
                with for_(i_gen, 0, i_gen < max_circuit_depth, i_gen + 1):
                    assign(step, rand.rand_int(24))
                    assign(current_state, cayley[current_state * 24 + step])
                    assign(sequence_list[i_gen], step)
                    assign(inv_gate_list[i_gen], inv_list[current_state])

                # Sweep depth
                with for_(*from_array(depth_var, depth_arr)):
                    # Replace last gate with recovery gate
                    assign(saved_gate, sequence_list[depth_var])
                    assign(sequence_list[depth_var], inv_gate_list[depth_var - 1])

                    # Averaging shots loop
                    with for_(n, 0, n < num_shots, n + 1):
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate)
                        align()

                        for i_q, qubit in multiplexed_qubits.items():
                            if strict_timing:
                                with strict_timing_():
                                    play_clifford(sequence_list, depth_var, qubit)
                            else:
                                play_clifford(sequence_list, depth_var, qubit)
                        align()

                        for i_q, qubit in multiplexed_qubits.items():
                            if use_state_discrimination:
                                qubit.readout_state(state[i_q])
                                save(state[i_q], state_st[i_q])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i_q], Q[i_q]))
                                save(I[i_q], I_st[i_q])
                                save(Q[i_q], Q_st[i_q])
                        align()

                    # Restore original gate
                    assign(sequence_list[depth_var], saved_gate)

        with stream_processing():
            m_st.save("n")
            for i_q in range(num_qubits):
                if use_state_discrimination:
                    state_st[i_q].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(depths)).buffer(num_sequences).save(f"state{i_q + 1}")
                else:
                    I_st[i_q].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(depths)).buffer(num_sequences).save(f"I{i_q + 1}")
                    Q_st[i_q].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(depths)).buffer(num_sequences).save(f"Q{i_q + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log=None,
    config=None,
) -> xr.Dataset:
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)
