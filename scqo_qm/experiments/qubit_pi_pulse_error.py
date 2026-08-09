"""Pi-pulse amplitude error amplification probe (QM/QUAM).

Sweeps drive amplitude factor across repeated odd gate counts (X^1, X^3, X^5...)
to precisely measure and calibrate pi-pulse rotation amplitude errors.

QM Pi-pulse error amplification for scqo - supplies only ``probe()``.
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
    amp_prefactors: List[float],
    gate_counts: List[int],
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Pi-pulse error amplification QUA program. Returns (program, sweep_axes)."""
    check_amp_scale_window(amp_prefactors, name=", ".join(qubits.get_names()))
    num_qubits = len(qubits)
    amp_arr = np.asarray(amp_prefactors, dtype=float)
    gc_arr = np.asarray(gate_counts, dtype=int)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "gate_count": xr.DataArray(gc_arr, attrs={"long_name": "gate count (repetitions)"}),
        "amp_prefactor": xr.DataArray(amp_arr, attrs={"long_name": "amplitude scaling factor"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)
        gc = declare(int)
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
                with for_each_(gc, gc_arr):
                    with for_each_(a, amp_arr):
                        # Qubit reset
                        for i_q, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                        align()

                        # Repeat X180 gate gc times
                        for i_q, qubit in multiplexed_qubits.items():
                            with for_(k, 0, k < gc, k + 1):
                                play("x180" * amp(a), qubit.xy.name)
                        align()

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
                    state_st[i_q].buffer(len(amp_arr)).buffer(len(gc_arr)).average().save(f"I{i_q + 1}")
                else:
                    I_st[i_q].buffer(len(amp_arr)).buffer(len(gc_arr)).average().save(f"I{i_q + 1}")
                    Q_st[i_q].buffer(len(amp_arr)).buffer(len(gc_arr)).average().save(f"Q{i_q + 1}")

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
from scqo.experiments import QubitPiPulseError


@register
class QMQubitPiPulseError(QubitPiPulseError):
    """Build a multiplexed Pi-pulse error amplification QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # self.sweep_axes, not a second define_sweep() call: run() has already
        # built the axes, and rebuilding them here is a way for the emitted sweep
        # and the declared one to drift apart
        return build_program(
            machine,
            qubits,
            amp_prefactors=list(self.sweep_axes["amp_prefactor"]),
            gate_counts=list(self.sweep_axes["gate_count"]),
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=False,
        )
