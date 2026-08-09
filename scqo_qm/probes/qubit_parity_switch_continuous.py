"""Continuous charge-parity monitor acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

y90 - fixed idle - x90 - measure, repeated as `num_shots` back-to-back single
shots. (The two-measurement-per-cycle sibling is
`scqo_qm/probes/qubit_parity_switch_discrete.py`.) The 90-degree-SHIFTED second pulse measures sin of the parity-dependent
phase (+/- pi/2 at the intended idle), which is ODD in parity, so the two charge
parities land on opposite readout poles; a same-axis pair would measure the even
cos and see no parity at all.

Two deliberate absences, both load-bearing:

* **No `qubit.reset(...)`.** Between shots only the RESONATOR is waited out
  (its depletion time), and the absence of a qubit reset is REQUIRED rather
  than an optimization: the sequence inverts with the pole the previous shot
  left behind, so the readout is the running XOR of the parity and the
  consecutive-pair difference is the parity telegraph the rate is fitted from.
  Resetting to |0> each shot would sever that chain. The shot cadence is also
  the telegraph timebase — a thermal wait would stretch it by orders of
  magnitude and scale every reported rate with it.
* **No `.average()` in the stream processing.** Every shot is its own sample of
  the telegraph; averaging them is exactly the information being measured.
  (The legacy `calibrations/exclude/LCH_parity_switch_ramsey.py` node carried a
  `.buffer(n).average()` that was a no-op only because a single buffer was ever
  produced.)

`idle_cycles` and `depletion_cycles` are per-qubit-name dicts of 4 ns clock
cycles; the caller (the scqo shell) owns the ns->cycles conversion and the
precedence between a per-run override and the standing QUAM value.

WHY THIS DOES NOT CALL `qubit.readout_state()` in discriminated mode, unlike
every other per-shot probe here: that helper ENDS with its own
`wait(resonator.depletion_time // 4)`. On top of this program's own
between-shot wait that is two depletion waits per shot — the cadence, and
therefore every reported rate, off by the difference — and it would also make
the per-run `readout_depletion_ns` override apply to only one of them. The
threshold assign is three lines of the same vendor API, so the wait keeps
exactly one owner. (The threshold itself still comes from the QUAM readout
operation, i.e. scqo's accepted `readout_threshold` knob.)
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
    num_shots: int,
    use_state_discrimination: bool = False,
    simulate: bool = False,
):
    """Build the parity-monitor QUA program. Returns (program, sweep_axes).

    `qubits` is a BatchableList (see `_lib.select_qubits`); the scqo shell
    selects it with `multiplexed=False` so each qubit gets its own batch and
    therefore its own uncoupled shot cadence — a multiplexed batch's aligns
    would tie every qubit's period to the slowest member, and the period is the
    quantity the rate is divided by.
    """
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "shot_idx": xr.DataArray(np.arange(1, num_shots + 1),
                                 attrs={"long_name": "shot index"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)

                # Resonator-only reset: wait out the readout photons of the
                # PREVIOUS shot. There is deliberately no qubit.reset() here.
                for i, qubit in multiplexed_qubits.items():
                    cycles = depletion_cycles[qubit.name]
                    if cycles:
                        qubit.resonator.wait(cycles)
                align()

                # y90 - fixed idle - x90
                for i, qubit in multiplexed_qubits.items():
                    qubit.xy.play("y90")
                    qubit.xy.wait(idle_cycles[qubit.name])
                    qubit.xy.play("x90")
                align()

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
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # buffer(num_shots) and NO .average(): every shot is a sample of
                # the telegraph (see the module docstring).
                if use_state_discrimination:
                    state_st[i].buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(num_shots).save(f"I{i + 1}")
                    Q_st[i].buffer(num_shots).save(f"Q{i + 1}")

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
