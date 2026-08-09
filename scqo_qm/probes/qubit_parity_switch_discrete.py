"""Discrete charge-parity monitor acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Per cycle: measure M1 - depletion wait - x90 - fixed idle - y90 - measure M2 -
pad wait, repeated as `num_shots` back-to-back cycles. M1 PROJECTS the qubit
(measurement-based initialization — the reason this probe carries no
`qubit.reset(...)` of any kind), the 90-degree-shifted pulse pair maps the
parity onto the pole, and M2 reads it: the parity of each cycle is the
WITHIN-CYCLE difference m1 XOR m2, so — unlike the continuous sibling — decay
during the pad wait corrupts no parity sample and the cycle may be padded
arbitrarily long (`tau_wait_cycles`, from scqo's `cycle_period_ns`).

BOTH measurements are recorded, into the SAME streams: two saves per cycle,
`buffer(2).buffer(num_shots)` and NO `.average()`, so each stream comes back
shaped (shot_idx, meas_idx) with meas_idx 0 = M1. Handle base names stay
digit-free (`state1`/`I1`/`Q1`), which is what lets the fetcher group them per
qubit.

Like the continuous probe, this does NOT call `qubit.readout_state()` in
discriminated mode: that helper ends with its own depletion wait, which would
double-count the governed wait and break the scheduled cycle period. The
threshold assign is inlined instead (the threshold still comes from the QUAM
readout operation, i.e. scqo's accepted `readout_threshold` knob).

`idle_cycles`, `depletion_cycles` and `tau_wait_cycles` are per-qubit-name
dicts of 4 ns clock cycles; the caller (the scqo shell) owns the ns->cycles
conversion, the cycle-period padding math and its too-short refusal.
"""

from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    idle_cycles: Dict[str, int],
    depletion_cycles: Dict[str, int],
    tau_wait_cycles: Dict[str, int],
    num_shots: int,
    use_state_discrimination: bool = False,
    simulate: bool = False,
):
    """Build the discrete parity-monitor QUA program. Returns (program, sweep_axes).

    `qubits` is a BatchableList (see `_lib.select_qubits`); the scqo shell
    selects it with `multiplexed=False` so each qubit gets its own batch and
    therefore its own uncoupled cycle period — the quantity the rate is
    divided by.
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shot_idx": xr.DataArray(np.arange(1, num_shots + 1),
                                 attrs={"long_name": "cycle index"}),
        "meas_idx": xr.DataArray(np.arange(2),
                                 attrs={"long_name": "measurement index (0 = M1)"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        def _measure_and_save(multiplexed_qubits):
            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                if use_state_discrimination:
                    # readout_state()'s body WITHOUT its trailing depletion
                    # wait — see the module docstring.
                    assign(state[i],
                           Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])
                else:
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                # M1: the projective initialization — deliberately no
                # qubit.reset() anywhere in this loop.
                _measure_and_save(multiplexed_qubits)
                align()

                # wait out M1's readout photons before the coherent block
                for i, qubit in multiplexed_qubits.items():
                    cycles = depletion_cycles[qubit.name]
                    if cycles:
                        qubit.resonator.wait(cycles)
                align()

                # x90 - fixed idle - y90 (the picture's pulse order; equivalent
                # to the continuous y90-first order up to a telegraph sign flip)
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("x90")
                    qubit.xy.wait(idle_cycles[qubit.name])
                    qubit.xy.play("y90")
                align()

                # M2: the parity readout, into the SAME streams as M1
                _measure_and_save(multiplexed_qubits)
                align()

                # pad the cycle to the requested period (0 = minimal cycle)
                for i, qubit in multiplexed_qubits.items():
                    cycles = tau_wait_cycles[qubit.name]
                    if cycles:
                        qubit.resonator.wait(cycles)
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # two saves per cycle -> meas_idx (len 2) is the innermost
                # buffered axis, then shot_idx; NO .average() — every
                # measurement is its own sample.
                if use_state_discrimination:
                    state_st[i].buffer(2).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(2).buffer(num_shots).save(f"I{i + 1}")
                    Q_st[i].buffer(2).buffer(num_shots).save(f"Q{i + 1}")

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
