"""T1 relaxation acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Excite with x180, wait a swept delay, measure. Sequence mirrors the vendored official
``calibrations/05_T1.py`` core (reset -> x180 -> resonator.wait -> measure).

QM qubit relaxation (T1) for scqo - supplies only ``probe()``.

Parameters, exponential fit and reporting are inherited from
``scqo.experiments.QubitRelaxation``. scqo sweeps ``wait_time_ns``; the QM builder
builds the same sweep on coord ``idle_time``, which the backend's ``_to_canonical``
renames back.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    wait_times_cycles,
    num_shots: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the T1 QUA program. Returns (program, sweep_axes).

    `wait_times_cycles` is the post-pi-pulse delay sweep in clock cycles (4 ns);
    `qubits` is a BatchableList (see `_lib.select_qubits`).
    """
    # Collapse sub-clock duplicates so the buffer length matches the axis (see ramsey).
    wait_times_cycles = np.unique(np.asarray(wait_times_cycles))

    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "idle_time": xr.DataArray(4 * wait_times_cycles, attrs={"long_name": "wait after pi pulse", "units": "ns"}),
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

                with for_each_(t, wait_times_cycles):
                    # Qubit initialization
                    for i, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log,
                                    max_attempts=reset_max_attempts)
                    align()
                    # pi pulse, then let the qubit decay for t (official 05_T1 core)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.align()
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.wait(t)
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
                    state_st[i].buffer(len(wait_times_cycles)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(wait_times_cycles)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(wait_times_cycles)).average().save(f"Q{i + 1}")

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


from typing import Any, ClassVar

import numpy as np

from scqo import register
from scqo.experiments import QubitRelaxation


@register
class QMQubitRelaxation(QubitRelaxation):
    """Build a multiplexed T1 QUA program on the QM OPX."""

    #: Readout is held at the calibrated point for the whole run and the reset is
    #: a genuine state reset, so reset_method='active' is valid here (_reset.py).
    supports_active_reset: ClassVar[bool] = True

    def probe(self) -> Any:
        from ._reset import check_reset_method, reset_max_attempts
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        wait_ns = self.sweep_axes["wait_time_ns"]
        wait_cycles = np.maximum(1, np.round(wait_ns / 4)).astype(int)

        return build_program(
            machine,
            qubits,
            wait_times_cycles=wait_cycles,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
