"""N-swap (swap-chain) x AC-Stark-amplitude acquisition probe: vendor code only
(qm/quam/qualang_tools) - no qualibrate, no scqo, no scqat.

The AC-Stark sibling of the qc_n_swap_amp probe: instead of sweeping the swap's
own flux amplitude, the swap runs at its FIXED baked amplitude (bare
``macros[swap_operation].apply()``) and after every swap a separate off-resonant
RF (XY) tone -- a new named ``stark_operation`` played on the control qubit's xy
line -- is played at the swept amplitude factor (outer axis). The tone is made
off-resonant by ``update_frequency`` (a `SquarePulse` carries no per-pulse
detuning), so it AC-Stark-shifts the qubit rather than rotating it; the frequency
is restored before readout.

Circuit per shot (for a swept stark amplitude factor a and swap count N):
  1. Initialize every involved qubit with ``q.reset(reset_type, simulate)``
     (involved = measured qubits + the swap pair's control/target).
  2. State prep: ``swap_pair.qubit_control.xy.play("x180")`` (at the RESONANT IF).
  3. Detune the control xy by ``stark_detuning_hz`` (``update_frequency``).
  4. Repeat N times: bare ``macros[swap_operation].apply()`` then
     ``qubit_control.xy.play(stark_operation, amplitude_scale=a)``, then idle the
     pair's flux lines for ``operation_gap_ns`` (if nonzero).
  5. Restore the control xy IF, then read out every measured qubit.

Reading the measured qubits versus (stark amplitude, N) gives a 2D population map
per joint state (a Stark-amplitude fine-tuning map by error amplification: more
swaps amplify a small residual detuning the Stark tone compensates). With state
discrimination the probe saves the **per-shot** discriminated states (var
``state``, dims ``(qubit, shot, stark_amplitude, round)``) so the node can render
the joint multi-qubit populations; without it the shot-averaged raw I/Q is saved
instead. There is no fit and no state writeback.

QM N-swap x AC-Stark-amplitude error-amplification map for scqo -- supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are inherited
from ``scqo.experiments.QcNStarkAmp``. This adapter:

* refuses unless ``drive_side``/``flux_side`` resolve to the vendor CONTROL
  member (the probe excites the control, plays the Stark tone on its xy line, and
  rides the swap on its flux line; a silent role mismatch would mislabel the
  prepared/transfer panels);
* orders the per-shot states into the readout schema's (high, low) ``member``
  axis and, in ``readout_mode="average"``, reduces them to ``joint_population``
  via scqo's shared ``states_to_joint_population`` -- ``"shot"`` returns the
  per-member ``state`` form as-is (the full-information trade).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._amp_limits import check_amp_scale_window


def _dedup_involved(measure_qubits, swap_pair) -> List:
    """Return the unique (by `.name`) set of qubit elements that must be initialized.

    The measured qubits need not include the swap pair's control/target, but all of them
    must be flux-initialized and reset at the start of each shot.
    """
    involved = []
    seen = set()
    for q in list(measure_qubits) + [swap_pair.qubit_control, swap_pair.qubit_target]:
        if q.name not in seen:
            seen.add(q.name)
            involved.append(q)
    return involved


def build_program(
    machine,
    measure_qubits,
    swap_pair,
    *,
    swap_operation: str,
    stark_operation: str,
    stark_detuning_hz: float,
    rounds_array,
    stark_amps,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    operation_gap_ns: int = 0,
    simulate: bool = False,
):
    """Build the N-swap x AC-Stark-amplitude QUA program.

    Returns ``(program, sweep_axes)``.

    `measure_qubits` is a plain list of qubit objects read out at the end of the circuit;
    `swap_pair` is a qubit-pair object whose `macros[swap_operation]` is applied (bare, at
    its baked amplitude) each swap. `rounds_array` is the integer sweep over the number of
    swaps (N=0 allowed, giving just the x180 prep, inner axis); `stark_amps` is the swept
    stark amplitude FACTOR (dimensionless amplitude_scale, outer axis) applied to the
    control qubit's `stark_operation` xy tone. `stark_detuning_hz` is the FIXED off-resonant
    detuning of that tone. All measured qubits are read out within the same shot (joint /
    multiplexed readout), since they share one circuit.

    `operation_gap_ns` (multiple of 4, default 0) idles the swap pair's flux lines after
    each swap+stark, so the pulses can settle before the next swap fires (the gap also
    separates the last swap from readout).
    """
    measure_qubits = list(measure_qubits)
    num_qubits = len(measure_qubits)
    rounds_array = np.asarray(rounds_array).astype(int)
    stark_amps = np.asarray(stark_amps, dtype=float)

    if operation_gap_ns < 0 or operation_gap_ns % 4 != 0:
        raise ValueError(f"operation_gap_ns must be a non-negative multiple of 4 ns, got {operation_gap_ns}.")
    gap_cycles = operation_gap_ns // 4

    involved = _dedup_involved(measure_qubits, swap_pair)
    ctrl = swap_pair.qubit_control

    # Validate the swap macro's flux path (the swap plays BARE at its baked amplitude, so
    # no rail-sweep check — only that the macro and its z flux pulse exist and are playable).
    if swap_operation not in swap_pair.macros:
        raise ValueError(f"Pair {swap_pair.name} has no macro {swap_operation!r}; available: {list(swap_pair.macros)}.")
    flux_pulse_name = getattr(swap_pair.macros[swap_operation], "flux_pulse", None)
    if not isinstance(flux_pulse_name, str) or flux_pulse_name not in ctrl.z.operations:
        raise ValueError(
            f"Macro {swap_operation!r} on {swap_pair.name} has no z flux_pulse playable on "
            f"{ctrl.name}.z (flux_pulse={flux_pulse_name!r})."
        )
    # Validate the swept AC-Stark tone: the operation must exist on the control's xy line,
    # and the amplitude-factor window must be expressible by QUA's dynamic amplitude_scale.
    if stark_operation not in ctrl.xy.operations:
        raise ValueError(
            f"Qubit {ctrl.name} has no xy operation {stark_operation!r}; available: "
            f"{list(ctrl.xy.operations)}. Register it first (quam_config/register_stark.py)."
        )
    check_amp_scale_window(stark_amps, name=f"{ctrl.name}.xy {stark_operation!r}",
                           knob="max_stark_amp")

    base_if = ctrl.xy.intermediate_frequency          # resonant IF (x180 + readout)
    stark_if = int(stark_detuning_hz) + base_if        # off-resonant IF for the stark tone

    # With state discrimination we save the per-shot discriminated states (so the joint
    # multi-qubit populations can be reconstructed downstream), hence the extra `shot` axis.
    # Without it we keep the shot-averaged raw I/Q schema (no `shot` axis).
    if use_state_discrimination:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            "shot": xr.DataArray(np.arange(num_shots)),
            # Outer loop -> y axis.
            "stark_amplitude": xr.DataArray(
                stark_amps, attrs={"long_name": "AC-Stark amplitude factor", "units": ""}
            ),
            # Inner loop -> x axis.
            "round": xr.DataArray(rounds_array, attrs={"long_name": "number of swaps"}),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in measure_qubits]),
            # Outer loop -> y axis.
            "stark_amplitude": xr.DataArray(
                stark_amps, attrs={"long_name": "AC-Stark amplitude factor", "units": ""}
            ),
            # Inner loop -> x axis.
            "round": xr.DataArray(rounds_array, attrs={"long_name": "number of swaps"}),
        }

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for the measured qubits.
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        q_a = declare(fixed)  # swept stark amplitude factor (amplitude_scale)
        r = declare(int)  # swept swap count (current value from rounds_array)
        rr = declare(int)  # inner swap counter
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        # Initialize the QPU in terms of flux points for every involved element.
        for q in involved:
            machine.initialize_qpu(target=q)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            # Stark-amplitude loop (outer -> y axis)
            with for_(*from_array(q_a, stark_amps)):
                # Swap-count loop (inner -> x axis)
                with for_(*from_array(r, rounds_array)):
                    # Initialization: thermalize / actively reset every involved qubit.
                    for q in involved:
                        q.reset(reset_type, simulate)
                    align()

                    # State prep: excite the swap pair's control qubit to |1> (resonant IF).
                    ctrl.xy.play("x180")
                    align()

                    # Detune the control's xy so the stark tone is off-resonant (a Stark
                    # SHIFT, not a rotation). Hoisted outside the swap loop: the control's
                    # xy frequency only matters for the stark play, and x180 (above) and
                    # readout (below) run at the resonant IF.
                    ctrl.xy.update_frequency(stark_if)

                    # Circuit body: N swaps on the pair, each bare (fixed baked amplitude),
                    # each followed SEQUENTIALLY by the swept off-resonant stark tone on the
                    # control xy. The swap and the stark do NOT overlap: apply() ends with the
                    # pair's align() (FluxTunableTransmonPair.align aligns every channel of both
                    # qubits, INCLUDING control.xy), so the control's xy timeline is synced to the
                    # end of the swap before the stark plays.
                    # Dynamic loop bound on r -> N=0 skips the body entirely (baseline).
                    # `gap_cycles` idles the pair BETWEEN the swap and the stark tone, so the
                    # swap's flux pulse settles before the stark tone plays.
                    with for_(rr, 0, rr < r, rr + 1):
                        swap_pair.macros[swap_operation].apply()
                        if gap_cycles > 0:
                            swap_pair.wait(gap_cycles)
                        ctrl.xy.play(stark_operation, amplitude_scale=q_a)
                        align()

                    # Restore the resonant IF before readout.
                    ctrl.xy.update_frequency(base_if)
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
                # Inner buffer = swap count (x), next buffer = stark amplitude (y).
                if use_state_discrimination:
                    # Keep every shot (no average) so the joint populations stay reconstructable:
                    # round buffer, then stark_amplitude, then group num_shots -> (shot, stark_amplitude, round).
                    state_st[i].buffer(len(rounds_array)).buffer(len(stark_amps)).buffer(num_shots).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(rounds_array)).buffer(len(stark_amps)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(rounds_array)).buffer(len(stark_amps)).average().save(f"Q{i + 1}")

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

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import QcNStarkAmp, states_to_joint_population


def _member_states(state: xr.DataArray, high_side: str) -> xr.DataArray:
    """The probe's per-shot states, reordered onto the schema's axes.

    ``state`` arrives ``(qubit, shot, stark_amplitude, round)`` with the qubit
    axis in READOUT order — ``[control, target]``, fixed by the adapter's
    measure list. The member axis is ROLE-ordered (high, low), so the vendor
    order flips when the roster's high member is the vendor target."""
    order = [0, 1] if high_side == "control" else [1, 0]
    da = state.isel(qubit=order).rename({
        "qubit": "member", "shot": "shot_idx",
        "stark_amplitude": "stark_amp", "round": "swap_count",
    })
    return da.assign_coords(member=["high", "low"])


@register
class QMQcNStarkAmp(QcNStarkAmp):
    """Build, run and fetch the N-swap AC-Stark amplitude map on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it executes one program per swap pair in a "
                           "Python loop inside probe()")

    def _build_pair_program(self, pair):
        """Build ONE target pair's QUA program — the build half of ``probe()``,
        no acquire. Shared by ``probe()`` (looped over targets + acquired) and
        ``preview_program()`` (a single pair, dumped for ``--preview``).

        The role/reset validation reads ``self.params.targets`` (it refuses a
        role that maps onto different vendor sides across the run), so it is the
        same whichever pair is passed — cheap to re-run per call for a handful
        of targets, and it keeps ``preview_program`` self-contained."""
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair

        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time.
        drive = role_side(self, self.params.drive_side, field="drive_side")
        flux = role_side(self, self.params.flux_side, field="flux_side",
                         needs_flux=True)
        if drive != "control" or flux != "control":
            raise ValueError(
                f"qc_n_stark_amp on QM excites the vendor pair's CONTROL member, "
                f"plays the AC-Stark tone on its xy line, and rides the swap on its "
                f"flux line, but drive_side={self.params.drive_side!r} / flux_side="
                f"{self.params.flux_side!r} resolve to (drive={drive}, flux={flux}). "
                f"Select the control-side role for both — or run pair_swap_chevron, "
                f"which honors either side.")

        # One door for the reset method: refuses what QM cannot honour (this
        # shell does not opt into active reset — default DENY — so 'active'
        # refuses by name until it has hardware evidence for this sequence).
        reset = check_reset_method(self)
        qp = vendor_pair(self, pair)
        return build_program(
            self.backend.machine, [qp.qubit_control, qp.qubit_target], qp,
            swap_operation=self.params.swap_operation,
            stark_operation=self.params.stark_operation,
            stark_detuning_hz=self.params.stark_detuning_hz,
            rounds_array=np.asarray(self.sweep_axes["swap_count"]).astype(int),
            stark_amps=np.asarray(self.sweep_axes["stark_amp"], dtype=float),
            num_shots=int(self.params.num_averages),
            reset_type=reset,
            use_state_discrimination=True,
            operation_gap_ns=int(self.params.operation_gap_ns),
        )

    def preview_program(self) -> Any:
        """The single-target ``--preview`` build (QMBackend's single-pair preview
        path). The backend gates on exactly one target before calling this, so a
        self-acquiring shell can still be inspected without touching the QPU."""
        prog, _sweep_axes = self._build_pair_program(self.params.targets[0])
        return prog

    def probe(self) -> Any:
        from ._vendor import role_side

        machine = self.backend.machine  # type: ignore[attr-defined]
        # `high` fixes the member ordering for the readout-schema reshape;
        # resolved once (the per-pair build re-checks drive/flux and reset).
        high_side = role_side(self, "high", field="targets")
        shots = int(self.params.num_averages)

        # One program per pair (the probe takes a single swap pair); the two
        # measured qubits are exactly the pair's members, control first.
        per_pair = []
        for pair in self.params.targets:
            prog, sweep_axes = self._build_pair_program(pair)
            raw = acquire(machine, prog, sweep_axes,
                          num_shots=shots,
                          timeout=self.backend._timeout)
            per_pair.append(_member_states(raw["state"], high_side))

        state = xr.concat(per_pair, dim="qubit_pair").assign_coords(
            qubit_pair=list(self.params.targets))
        if self.params.readout_mode == "shot":
            return state.to_dataset(name="state")
        jp = states_to_joint_population(state, member_dim="member",
                                        shot_dim="shot_idx")
        return jp.to_dataset()
