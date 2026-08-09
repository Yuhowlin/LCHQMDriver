"""Swap-based-reset circuit acquisition probe: vendor code only (qm/quam/qualang_tools) -
no qualibrate, no scqo, no scqat.

Exercises a swap-based reset scheme. Each shot prepares an excitation on a chosen
`excite_qubit` (`x180`), then applies the circuit body R times: a SWAP on the
qubit pair followed by a RESET on an ancilla qubit. The number of rounds R is the swept
axis, so reading out a chosen set of qubits versus R shows how the population is removed
round by round (R=0 is the no-round baseline: just the x180 prep). One curve per measured
qubit.

The excited qubit is chosen independently of the swap pair: `excite_qubit` need not be the
swap pair's control qubit (it can be the target or any other qubit).

Circuit per shot (for a swept round count R):
  1. Initialize every involved qubit with `q.reset(reset_type, simulate)`
     (involved = measured qubits + `excite_qubit` + the swap pair's control/target + the
     reset qubit).
  2. State prep: `excite_qubit.xy.play("x180")`.
  3. Repeat R times: `swap_pair.macros[swap_operation].apply()` then
     `reset_qubit.macros[reset_operation].apply()`.
  4. Read out every measured qubit (state discrimination -> discriminated state, else raw I/Q).

The swap/reset macros are invoked **bare** (`.apply()` with no extra args), so the chosen
macros must be callable that way - their amplitudes/pulses baked into the QUAM macro
definition (e.g. like `CZImplementation`, whose `apply()` has all-default args). A macro
whose `apply()` has required positional args (e.g. the current `ISwapImplementation`,
which needs `ctrl_amp`/`cplr_amp`) needs a default-carrying variant before it can be used
here as `swap_operation`/`reset_operation`.

There is no fit/state-writeback downstream. With state discrimination the probe saves the
**per-shot** discriminated states (var `state`, dims `(qubit, shot, round)`) so the node can
render either the joint multi-qubit populations (000, 001, ... when `multiplexed`) or the
per-qubit marginals; without it the shot-averaged raw I/Q is saved instead.
"""

from typing import Callable, List, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def _dedup_involved(measure_qubits, swap_pair, reset_qubit, excite_qubit) -> List:
    """Return the unique (by `.name`) set of qubit elements that must be initialized.

    The measured qubits need not include `excite_qubit`, the swap pair's control/target, or
    the reset qubit, but all of them must be flux-initialized and reset at the start of each
    shot.
    """
    involved = []
    seen = set()
    for q in list(measure_qubits) + [excite_qubit, swap_pair.qubit_control, swap_pair.qubit_target, reset_qubit]:
        if q.name not in seen:
            seen.add(q.name)
            involved.append(q)
    return involved


def build_program(
    machine,
    measure_qubits,
    swap_pair,
    reset_qubit,
    *,
    excite_qubit,
    swap_operation: str,
    reset_operation: str,
    rounds_array,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    operation_gap_ns: int = 0,
    simulate: bool = False,
):
    """Build the swap-based-reset QUA program.

    Returns ``(program, sweep_axes)``.

    `measure_qubits` is a plain list of qubit objects read out at the end of the circuit;
    `excite_qubit` is the qubit object excited with `x180` for state prep (chosen
    independently of the swap pair);
    `swap_pair` is a qubit-pair object whose `macros[swap_operation]` is applied each round;
    `reset_qubit` is a qubit object whose `macros[reset_operation]` is applied each round.
    `rounds_array` is the integer sweep over the number of swap+reset rounds (R=0 allowed,
    giving just the x180 prep). All measured qubits are read out within the same shot
    (joint / multiplexed readout), since they share one circuit.

    `operation_gap_ns` (multiple of 4, default 0) idles the swap pair's flux lines between
    the circuit's gate operations (before/after each swap), so a preceding reset's flux
    pulse can settle before the swap fires.
    (For pure swap rounds with no reset, use the `qc_N_swap` node instead.)
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)

    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(f"operation_gap_ns must be a non-negative multiple of 4 ns, got {operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4

    involved = _dedup_involved(measure_qubits, swap_pair, reset_qubit, excite_qubit)

    # With state discrimination we save the per-shot discriminated states (so the joint
    # multi-qubit populations can be reconstructed downstream), hence the extra `shot` axis.
    # Without it we keep the shot-averaged raw I/Q schema (no `shot` axis).
    if use_state_discrimination:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            "shot": xr.DataArray(np.arange(num_shots)),
            "round": xr.DataArray(
                rounds_array, attrs={"long_name": "number of swap+reset rounds"}
            ),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            "round": xr.DataArray(
                rounds_array, attrs={"long_name": "number of swap+reset rounds"}
            ),
        }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the measured qubits.
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        r = declare(int)  # swept round count (current value from rounds_array)
        rr = declare(int)  # inner round counter
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        # Initialize the QPU in terms of flux points for every involved element.
        for q in involved:
            machine.initialize_qpu(target=q)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(r, rounds_array)):
                # Initialization: thermalize / actively reset every involved qubit.
                for q in involved:
                    q.reset(reset_type, simulate)
                align()

                # State prep: excite the chosen qubit to |1>.
                excite_qubit.xy.play("x180")
                align()

                # Circuit body: R rounds of (swap on the pair, reset on the ancilla).
                # Dynamic loop bound on r -> R=0 skips the body entirely (baseline).
                # `gap_cycles` idles the pair's flux lines between gate operations so a
                # preceding reset's flux pulse can settle before the swap fires.
                reset_qubit.macros[reset_operation].apply()
                align()
                with for_(rr, 0, rr < r, rr + 1):
                    if gap_cycles > 0:
                        swap_pair.wait(gap_cycles)
                    align()
                    swap_pair.macros[swap_operation].apply()
                    if gap_cycles > 0:
                        swap_pair.wait(gap_cycles)
                    align()
                    reset_qubit.macros[reset_operation].apply()
                    align()

                # Joint (multiplexed) readout of all measured qubits.
                for i, qubit in enumerate(measure_qubits):
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
                    # Keep every shot (no average) so the joint populations stay reconstructable:
                    # inner `round` buffer first, then group `num_shots` of them -> (shot, round).
                    state_st[i].buffer(len(rounds_array)).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(rounds_array)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(rounds_array)).average().save(f"Q{i + 1}")

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
