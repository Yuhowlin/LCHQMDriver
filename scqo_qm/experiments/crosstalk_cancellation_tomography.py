"""Crosstalk Cancellation Tomography probe (QM/QUAM).

Supplies QUA program generation and data acquisition for CrosstalkCancellationTomography.
Measures quantum state trajectories on the Bloch sphere during microwave crosstalk
cancellation across three conditions:
1. 'noise_only': Probe state evolution driven purely by aggressor microwave leakage
   with drive_qubit phase rotating by -sub_p per pulse to freeze the crosstalk drive axis.
2. 'cancel_only': Probe state evolution driven purely by cancellation pulses on cancel_elem
   with constant init_phase.
3. 'compensated': Probe state evolution under simultaneous drive and cancellation.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, List
import numpy as np
import xarray as xr
from qm.qua import *

from scqo import register
from scqo.experiments.crosstalk_cancellation_tomography import (
    CrosstalkCancellationTomography as _CoreTomography,
    compute_theoretical_phase_rate,
)
from scqo_qm.experiments._lib import acquire as _acquire


def build_program(
    machine,
    probe_qubit,
    drive_qubit,
    *,
    target_gate: str = "x180",
    cancel_amp: float = 0.0,
    init_phase: float = 0.0,
    sub_phase_rate_turns: float = 0.0,
    repetitions: list[int],
    conditions: list[str],
    bases: list[str],
    num_shots: int,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    simulate: bool = False,
):
    """Build the QUA program for Crosstalk Cancellation Tomography."""
    cancel_elem = f"{probe_qubit.xy.name}_cancel"

    try:
        op_len = probe_qubit.xy.operations[target_gate].length
        pulse_cycles = max(op_len // 4, 1)
    except Exception:
        pulse_cycles = 4

    sub_p = float(sub_phase_rate_turns)

    sweep_axes = {
        "qubit": xr.DataArray([probe_qubit.name]),
        "condition": xr.DataArray(np.asarray(conditions, dtype=str)),
        "repetitions": xr.DataArray(np.asarray(repetitions, dtype=int)),
        "basis": xr.DataArray(np.asarray(bases, dtype=str)),
    }


    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()

        if use_state_discrimination:
            state = [declare(int)]
            state_st = [declare_stream()]

        amp_fixed = declare(fixed)
        init_phase_fixed = declare(fixed)
        assign(amp_fixed, float(cancel_amp))
        assign(init_phase_fixed, float(init_phase / (2.0 * np.pi)))

        machine.initialize_qpu(target=probe_qubit)
        machine.initialize_qpu(target=drive_qubit)
        align()

        for cond in conditions:
            for r_val in repetitions:
                r_int = int(r_val)
                for b_str in bases:
                    with for_(n, 0, n < num_shots, n + 1):
                        probe_qubit.reset(reset_type, simulate)
                        drive_qubit.reset(reset_type, simulate)

                        with strict_timing_():
                            align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                            reset_frame(probe_qubit.xy.name)
                            reset_frame(drive_qubit.xy.name)
                            reset_frame(cancel_elem)
                            reset_if_phase(probe_qubit.xy.name)
                            reset_if_phase(drive_qubit.xy.name)
                            reset_if_phase(cancel_elem)
                            reset_global_phase()

                            # In cancel_only and compensated conditions, set initial phase on cancel_elem
                            if cond in ("cancel_only", "compensated"):
                                frame_rotation_2pi(init_phase_fixed, cancel_elem)

                            # Play N repetition pulses
                            if r_int > 0:
                                if cond == "noise_only":
                                    for _ in range(r_int):
                                        drive_qubit.xy.play(target_gate)
                                        wait(pulse_cycles, cancel_elem)
                                        if sub_p != 0.0:
                                            frame_rotation_2pi(-sub_p, drive_qubit.xy.name)
                                elif cond == "cancel_only":
                                    for _ in range(r_int):
                                        wait(pulse_cycles, drive_qubit.xy.name)
                                        play(target_gate * amp(amp_fixed), cancel_elem)
                                elif cond == "compensated":
                                    for _ in range(r_int):
                                        drive_qubit.xy.play(target_gate)
                                        play(target_gate * amp(amp_fixed), cancel_elem)
                                        if sub_p != 0.0:
                                            frame_rotation_2pi(-sub_p, drive_qubit.xy.name)
                            else:
                                wait(4, probe_qubit.xy.name)

                        # Strictly align probe with drive and cancel element before tomography basis pre-rotation
                        align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)

                        # Tomography basis pre-rotation on probe_qubit
                        b_lower = str(b_str).lower()
                        if b_lower == "z":
                            pass
                        elif b_lower == "x":
                            probe_qubit.xy.play("-y90")
                        elif b_lower == "y":
                            probe_qubit.xy.play("x90")

                        align()

                        if use_state_discrimination:
                            probe_qubit.readout_state(state[0])
                            save(state[0], state_st[0])
                        else:
                            probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                            save(I[0], I_st[0])
                            save(Q[0], Q_st[0])
                        align()

        with stream_processing():
            if use_state_discrimination:
                state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(bases)).buffer(len(repetitions)).buffer(len(conditions)).save("state1")
            else:
                I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(bases)).buffer(len(repetitions)).buffer(len(conditions)).save("I1")
                Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(bases)).buffer(len(repetitions)).buffer(len(conditions)).save("Q1")

    return prog, sweep_axes



from .crosstalk_compensated_benchmark import _patch_cancel_element
from scqo_qm.experiments._lib import select_qubits


@register
class CrosstalkCancellationTomography(_CoreTomography):
    """Concrete Crosstalk Cancellation Tomography running on QM OPX/QUAM hardware."""

    def patch_preview_config(self, config: dict) -> dict:
        import copy
        config = copy.deepcopy(config)
        machine = self.backend.machine
        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit
        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        probe_qubit = q_dict.get(probe_name, qubits[0])
        drive_qubit = q_dict.get(drive_name, qubits[1])
        return _patch_cancel_element(config, probe_qubit, drive_qubit, self.params.target_gate)

    def preview_program(self):
        """Render a single representative program for preview without memory bloat or simulator timeouts."""
        machine = self.backend.machine
        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit

        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        probe_qubit = q_dict.get(probe_name, qubits[0])
        drive_qubit = q_dict.get(drive_name, qubits[1])

        try:
            op_len = probe_qubit.xy.operations[self.params.target_gate].length
            duration_ns = float(op_len)
        except Exception:
            duration_ns = float(self.params.pulse_duration_ns)

        f_drive = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None)
        f_probe = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None)
        if f_drive is not None and f_probe is not None:
            phase_rate_rad = compute_theoretical_phase_rate(float(f_drive), float(f_probe), duration_ns)
        else:
            phase_rate_rad = 0.0

        sub_phase_rate_turns = float((phase_rate_rad / (2.0 * np.pi)) % 1.0)
        if sub_phase_rate_turns > 0.5:
            sub_phase_rate_turns -= 1.0

        preview_reps = [0, 2]
        preview_conds = ["compensated"]
        preview_bases = ["z"]
        reset_type = getattr(self.params, "reset_method", None) or getattr(self.params, "reset_type", "thermal")

        prog, _ = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            target_gate=self.params.target_gate,
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            sub_phase_rate_turns=sub_phase_rate_turns,
            repetitions=preview_reps,
            conditions=preview_conds,
            bases=preview_bases,
            num_shots=min(int(self.params.num_averages), 10),
            reset_type=reset_type,
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
        )
        return prog

    def probe(
        self,
        timeout: int = 180,
        log: Optional[Callable] = None,
        config: Optional[dict] = None,
    ) -> xr.Dataset:
        machine = self.backend.machine
        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit

        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        probe_qubit = q_dict.get(probe_name, qubits[0])
        drive_qubit = q_dict.get(drive_name, qubits[1])

        # Calculate phase rate
        if self.params.phase_rate is not None:
            phase_rate_rad = float(self.params.phase_rate)
        else:
            try:
                op_len = probe_qubit.xy.operations[self.params.target_gate].length
                duration_ns = float(op_len)
            except Exception:
                duration_ns = float(self.params.pulse_duration_ns)

            f_drive = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None)
            f_probe = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None)
            if f_drive is not None and f_probe is not None:
                phase_rate_rad = compute_theoretical_phase_rate(float(f_drive), float(f_probe), duration_ns)
            else:
                phase_rate_rad = 0.0

        sub_phase_rate_turns = float((phase_rate_rad / (2.0 * np.pi)) % 1.0)
        if sub_phase_rate_turns > 0.5:
            sub_phase_rate_turns -= 1.0

        if config is None:
            config = machine.generate_config()
        config = _patch_cancel_element(config, probe_qubit, drive_qubit, self.params.target_gate)

        reps = self.params.get_repetitions().tolist()
        conditions = self.params.get_conditions()
        bases = self.params.get_bases()
        num_shots = int(self.params.num_averages)

        reset_type = getattr(self.params, "reset_method", None) or getattr(self.params, "reset_type", "thermal")
        use_sd = bool(self.params.use_state_discrimination)
        batch_size = max(1, getattr(self.params, "batch_size", 4))

        if len(reps) <= batch_size:
            prog, sweep_axes = build_program(
                machine,
                probe_qubit,
                drive_qubit,
                target_gate=self.params.target_gate,
                cancel_amp=float(self.params.cancel_amp),
                init_phase=float(self.params.init_phase),
                sub_phase_rate_turns=sub_phase_rate_turns,
                repetitions=reps,
                conditions=conditions,
                bases=bases,
                num_shots=num_shots,
                reset_type=reset_type,
                use_state_discrimination=use_sd,
                simulate=getattr(self.backend, "simulate", False),
            )

            return _acquire(
                machine,
                prog,
                sweep_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )

        batches = [reps[i : i + batch_size] for i in range(0, len(reps), batch_size)]
        num_batches = len(batches)
        batch_datasets: list[xr.Dataset] = []

        print(
            f"# [Tomography] Executing {len(reps)} repetition points in {num_batches} "
            f"batches (batch_size={batch_size})..."
        )

        for b_idx, batch in enumerate(batches):
            prog_b, sweep_axes_b = build_program(
                machine,
                probe_qubit,
                drive_qubit,
                target_gate=self.params.target_gate,
                cancel_amp=float(self.params.cancel_amp),
                init_phase=float(self.params.init_phase),
                sub_phase_rate_turns=sub_phase_rate_turns,
                repetitions=batch,
                conditions=conditions,
                bases=bases,
                num_shots=num_shots,
                reset_type=reset_type,
                use_state_discrimination=use_sd,
                simulate=getattr(self.backend, "simulate", False),
            )
            ds_b = _acquire(
                machine,
                prog_b,
                sweep_axes_b,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )
            batch_datasets.append(ds_b)
            print(f"# [Tomography Batch {b_idx + 1}/{num_batches}] Repetitions {batch} completed.")

        total_ds = xr.concat(batch_datasets, dim="repetitions")
        return total_ds

