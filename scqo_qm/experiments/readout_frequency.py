"""Readout-frequency acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Single-shot readout vs readout-frequency detuning: for each detuning the qubit is
prepared in |0> and |1> and measured, giving the per-frequency IQ blobs that the
fidelity estimator scans for the optimal readout frequency.

QM readout-frequency (fidelity vs readout detuning) for scqo - supplies only ``probe()``.

Parameters, the analysis and the ``readout_freq_hz`` writeback are inherited from
``scqo.experiments.ReadoutFrequency``. TWO READOUT MODES, one program shape,
differing only in the stream terminal:

* ``readout_mode="shot"`` — every readout shot's I/Q point is recorded
  individually: ``buffer(2).buffer(len(dfs)).buffer(num_shots)``, NO
  ``.average()``, and a ``shot_idx`` sweep axis.
* ``readout_mode="average"`` — the same shots are averaged on the FPGA:
  ``buffer(2).buffer(len(dfs)).average()``, and ``shot_idx`` leaves the sweep
  axes entirely (scqo's contract accepts that form as the alt set). The shot
  loop still runs ``num_shots`` times; only its terminal changes.

scqo sweeps ``detuning_hz``; the QM builder applies each value
relative to the resonator's current IF (``df + intermediate_frequency`` — same
convention as the resonator_spectroscopy wrapper) but names its axis
``frequency``, which we re-key below.

AXIS-ORDER NOTE (do not "fix" the order below): the probe's QUA loops nest shot
(outer) over detuning over prepared-state (inner), so the raw per-qubit array is
shaped (shot_idx, frequency, prepared_state) — a permutation of scqo's declared
sweep order (detuning_hz, prepared_state, shot_idx). We re-key the probe's
sweep_axes with the canonical scqo names IN RAW NESTING ORDER (only ``frequency``
-> ``detuning_hz`` actually changes; the values already ARE the detunings), so
the backend's ``_to_canonical`` takes its name-based path (per-name size checks,
no positional rename — which would scramble the axes) and ``estimate()``
transposes by name.

SHOT-INDEX VALUES NOTE: the probe's ``shot_idx`` coord is ``arange(1, n+1)``
while scqo declares ``arange(n)``. This offset is acceptable: the name-based
``_to_canonical`` path asserts SIZES, not values, and ``estimate()`` uses the
coord only for transposing/slicing, never its numeric values (same situation as
the single_shot_readout / readout_fidelity pair).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    dfs,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
    average_shots: bool = False,
):
    """Build the readout-frequency QUA program. Returns (program, sweep_axes).

    `dfs` is the readout-frequency detuning sweep in Hz (relative to the current
    resonator IF); `qubits` is a BatchableList (see `_lib.select_qubits`).
    Always acquires analog I/Q (no state-discrimination branch). With
    `average_shots` the FPGA averages the `num_shots` repetitions instead of
    streaming each one, and the returned sweep_axes carry no `shot_idx`.
    """
    num_qubits = len(qubits)
    n_runs = num_shots

    prepared_states = [0, 1]
    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "frequency": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
        "prepared_state": xr.DataArray(prepared_states, attrs={"long_name": "prepared qubit state", "units": ""}),
    }
    if not average_shots:  # shot-major nesting: the outer loop is the shot loop
        sweep_axes = {"qubit": sweep_axes["qubit"],
                      "shot_idx": xr.DataArray(np.arange(1, n_runs + 1),
                                               attrs={"long_name": "number of shots"}),
                      **{k: v for k, v in sweep_axes.items() if k != "qubit"}}
    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)
        ps = declare(int)
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                # ground iq blobs for all qubits
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    with for_each_(ps, prepared_states):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate)
                        align()

                        # Change qubit state
                        for i, qubit in multiplexed_qubits.items():
                            qubit.align()

                            with switch_(ps):
                                with case_(0):
                                    pass
                                with case_(1):
                                    qubit.xy.play("x180")

                            qubit.align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            update_frequency(qubit.resonator.name, df + qubit.resonator.intermediate_frequency)
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            qubit.align()
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # the ONE difference between the modes: what closes the shot loop
                i_st = I_st[i].buffer(len(prepared_states)).buffer(len(dfs))
                q_st = Q_st[i].buffer(len(prepared_states)).buffer(len(dfs))
                i_st = i_st.average() if average_shots else i_st.buffer(n_runs)
                q_st = q_st.average() if average_shots else q_st.buffer(n_runs)
                i_st.save(f"I{i + 1}")
                q_st.save(f"Q{i + 1}")

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


from typing import Any

from scqo import register
from scqo.experiments import ReadoutFrequency


@register
class QMReadoutFrequency(ReadoutFrequency):
    """Build a multiplexed readout-frequency-scan QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, axes = build_program(
            machine,
            qubits,
            dfs=self.sweep_axes["detuning_hz"],
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
            average_shots=self.params.readout_mode == "average",
        )
        # Canonical names in RAW nesting order (shot outer where it exists, then
        # detuning, prepared state inner — see module docstring); the DataArray
        # values are reused (the probe's "frequency" values already are the
        # detunings in Hz).
        sweep_axes = {"qubit": axes["qubit"]}
        if "shot_idx" in axes:
            sweep_axes["shot_idx"] = axes["shot_idx"]
        sweep_axes["detuning_hz"] = axes["frequency"]
        sweep_axes["prepared_state"] = axes["prepared_state"]
        return prog, sweep_axes
