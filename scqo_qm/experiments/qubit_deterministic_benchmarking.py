"""Deterministic Benchmarking probe (QM/QUAM).

Repeats a specified target gate (x180, y180, x90, y90, -x90, -y90) N times
across an array of repetition counts N to observe gate rotation error accumulation.

QM Deterministic Benchmarking for scqo - supplies only ``probe()``.
"""

from __future__ import annotations

from typing import Callable, Optional, List
import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    num_shots: int,
    repetitions: List[int],
    amp_prefactors: Optional[List[float]] = None,
    target_gate: str = "x180",
    reset_type: str,
    use_state_discrimination: bool = True,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Deterministic Benchmarking QUA program."""
    num_qubits = len(qubits)
    rep_arr = np.asarray(repetitions, dtype=int)
    amp_arr = np.asarray(amp_prefactors if amp_prefactors is not None else [1.0], dtype=float)
    check_amp_scale_window(amp_arr, name=", ".join(qubits.get_names()))
    gate_name = str(target_gate).strip().lower()

    # Map friendly gate names to QAM operation names
    if gate_name in ("x180", "x", "pi"):
        op = "x180"
    elif gate_name in ("y180", "y"):
        op = "y180"
    elif gate_name in ("x90", "pi_half"):
        op = "x90"
    elif gate_name in ("y90",):
        op = "y90"
    elif gate_name in ("-x90", "minus_x90"):
        op = "-x90"
    elif gate_name in ("-y90", "minus_y90"):
        op = "-y90"
    else:
        op = gate_name

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amp_arr, attrs={"long_name": "amplitude scaling factor"}),
        "repetitions": xr.DataArray(rep_arr, attrs={"long_name": "gate repetitions N"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        rep_var = declare(int)
        a = declare(fixed)
        k = declare(int)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(a, amp_arr):
                    with for_each_(rep_var, rep_arr):
                        # Qubit reset
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                        align()

                        # Repeat target gate N times with amplitude scaling factor a
                        for i_q, qubit in multiplexed_qubits.items():
                            with for_(k, 0, k < rep_var, k + 1):
                                play(op * amp(a), qubit.xy.name)
                        align()

                        # Readout
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
                    state_st[i_q].buffer(len(rep_arr)).buffer(len(amp_arr)).average().save(f"state{i_q + 1}")
                else:
                    I_st[i_q].buffer(len(rep_arr)).buffer(len(amp_arr)).average().save(f"I{i_q + 1}")
                    Q_st[i_q].buffer(len(rep_arr)).buffer(len(amp_arr)).average().save(f"Q{i_q + 1}")



    return prog, sweep_axes


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


from typing import Any

from scqo import register
from scqo.experiments import QubitDeterministicBenchmarking


@register
class QMQubitDeterministicBenchmarking(QubitDeterministicBenchmarking):
    """Build a multiplexed Deterministic Benchmarking QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)
        repetitions = self.params.get_repetitions()

        return build_program(
            machine,
            qubits,
            num_shots=int(self.params.num_averages),
            repetitions=repetitions,
            # from sweep_axes, not the Parameters: run() has already resolved the
            # window (explicit list / single-point mode / linspace) into the axis
            amp_prefactors=list(self.sweep_axes["amp_prefactor"]),
            target_gate=self.params.target_gate,
            reset_type=check_reset_method(self),
            use_state_discrimination=self.params.use_state_discrimination,
        )


