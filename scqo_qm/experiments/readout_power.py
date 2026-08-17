"""Readout-power acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Single-shot readout of each qubit prepared in |0> and |1> while sweeping the readout amplitude;
the amplitude maximising single-shot fidelity is found downstream.

QM readout-power (fidelity vs readout amplitude) for scqo - supplies only ``probe()``.

Parameters, the analysis and the ``readout_amp`` writeback are inherited from
``scqo.experiments.ReadoutPower``. TWO READOUT MODES, one program shape,
differing only in the stream terminal:

* ``readout_mode="shot"`` — every readout shot's I/Q point is recorded
  individually: ``buffer(2).buffer(len(amps)).buffer(num_shots)``, NO
  ``.average()``, and a ``shot_idx`` sweep axis.
* ``readout_mode="average"`` — the same shots are averaged on the FPGA:
  ``buffer(2).buffer(len(amps)).average()``, and ``shot_idx`` leaves the sweep
  axes entirely (scqo's contract accepts that form as the alt set). The shot
  loop still runs ``num_shots`` times; only its terminal changes.

The swept ``amp_prefactor`` values are applied as QUA ``amplitude_scale`` on the
readout pulse (prefactor x current readout_amp).

AXIS-ORDER NOTE: the probe's QUA loops nest shot (outer) over amplitude over
prepared-state (inner), so the raw per-qubit array is shaped
(shot_idx, amp_prefactor, prepared_state) — a permutation of scqo's declared
sweep order (amp_prefactor, prepared_state, shot_idx). The probe's sweep_axes
already carry exactly the canonical names ``shot_idx``/``amp_prefactor``/
``prepared_state`` in that raw nesting order, so the backend's ``_to_canonical``
takes its name-based path (no positional rename — which would scramble the axes)
and ``estimate()`` transposes by name.

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

from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    qubits,
    *,
    amps,
    num_shots: int,
    reset_type: str,
    simulate: bool = False,
    average_shots: bool = False,
):
    """Build the readout-power QUA program. Returns (program, sweep_axes).

    `amps` is the readout-amplitude prefactor sweep; `num_shots` measurements are taken per
    (amplitude, prepared state) point (0 = ground, 1 = x180-excited). `qubits` is a
    BatchableList (see `_lib.select_qubits`). With `average_shots` the FPGA
    averages those repetitions instead of streaming each one, and the returned
    sweep_axes carry no `shot_idx`.
    """
    check_amp_scale_window(amps, name=", ".join(qubits.get_names()))
    num_qubits = len(qubits)
    prepared_states = [0, 1]

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "readout amplitude", "units": ""}),
        "prepared_state": xr.DataArray(prepared_states, attrs={"long_name": "prepared qubit state", "units": ""}),
    }
    if not average_shots:  # shot-major nesting: the outer loop is the shot loop
        sweep_axes = {"qubit": sweep_axes["qubit"],
                      "shot_idx": xr.DataArray(np.arange(1, num_shots + 1),
                                               attrs={"long_name": "number of shots"}),
                      **{k: v for k, v in sweep_axes.items() if k != "qubit"}}
    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)
        ps = declare(int)
        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                # ground iq blobs for all qubits
                save(n, n_st)
                with for_(*from_array(a, amps)):
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
                            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                            qubit.align()
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # the ONE difference between the modes: what closes the shot loop
                i_st = I_st[i].buffer(len(prepared_states)).buffer(len(amps))
                q_st = Q_st[i].buffer(len(prepared_states)).buffer(len(amps))
                i_st = i_st.average() if average_shots else i_st.buffer(num_shots)
                q_st = q_st.average() if average_shots else q_st.buffer(num_shots)
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
from scqo.experiments import ReadoutPower


@register
class QMReadoutPower(ReadoutPower):
    """Build a multiplexed readout-amplitude-scan QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        return build_program(
            machine,
            qubits,
            amps=self.sweep_axes["amp_prefactor"],
            num_shots=self.params.num_shots,
            reset_type=check_reset_method(self),
            average_shots=self.params.readout_mode == "average",
        )
