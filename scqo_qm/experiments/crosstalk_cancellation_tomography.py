"""Crosstalk Cancellation Tomography probe (QM/QUAM).

Supplies QUA program generation and data acquisition for CrosstalkCancellationTomography: the
probe qubit's Bloch trajectory under N repetitions of target_gate in three conditions.

1. 'noise_only':  only the drive qubit plays - the probe is driven by the leaked field.
2. 'cancel_only': only the cancel element (on the probe's line, at the drive's frequency) plays
                  a cancel_amp-scaled copy of the drive's pulse, phase init_phase.
3. 'compensated': both together.

The leaked field is at the drive's frequency, so in the probe's frame its phase advances by
sub_p = (f_drive - f_probe) * pulse length every pulse. The drive and the cancel element both
turn their frame by -sub_p per pulse, which holds the leak's (and the cancel's) rotation axis
still so each trajectory is a fixed rotation per pulse.

The cancel elements are built exactly as crosstalk_compensated_sqrb builds them (same tone,
same core order), so the fitted cancel_amp / init_phase carry over to the benchmark when the
context matches (bidirectional + read_drive for a bidirectional benchmark).
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo import register
from scqo.experiments.crosstalk_cancellation_tomography import (
    CrosstalkCancellationTomography as _CoreTomography,
    compute_theoretical_phase_rate,
)
from scqo_qm.experiments._lib import acquire as _acquire, select_qubits
from .crosstalk_compensated_sqrb import _patch_cancel_element


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
    bidirectional: bool = False,
    read_drive: bool = False,
):
    """Build the QUA program for Crosstalk Cancellation Tomography.

    bidirectional: the drive-line cancel element is in the program too, idle (as in the
    bidirectional benchmark while only the drive's leak is measured). read_drive: the drive
    qubit is read in the same shot as the probe (its data is not returned).
    """
    cancel_elem = f"{probe_qubit.xy.name}_cancel"
    reverse_elem = f"{drive_qubit.xy.name}_cancel" if bidirectional else None
    elems = [probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem] + ([reverse_elem] if reverse_elem else [])

    try:
        pulse_cycles = max(probe_qubit.xy.operations[target_gate].length // 4, 1)
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
        if read_drive:
            drive_state = declare(int)
            drive_I, drive_Q = declare(fixed), declare(fixed)
            drive_st = declare_stream()

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
                            align(*elems)
                            for el in elems:
                                reset_frame(el)
                            for el in elems:
                                reset_if_phase(el)
                            reset_global_phase()
                            if cond in ("cancel_only", "compensated"):
                                frame_rotation_2pi(init_phase_fixed, cancel_elem)

                            if r_int > 0:
                                for _ in range(r_int):
                                    if cond in ("noise_only", "compensated"):
                                        drive_qubit.xy.play(target_gate)
                                    else:
                                        wait(pulse_cycles, drive_qubit.xy.name)
                                    if cond in ("cancel_only", "compensated"):
                                        play(target_gate * amp(amp_fixed), cancel_elem)
                                    else:
                                        wait(pulse_cycles, cancel_elem)
                                    if reverse_elem is not None:
                                        wait(pulse_cycles, reverse_elem)
                                    if sub_p != 0.0:
                                        # hold the leak's (and the copy's) axis still in the probe frame
                                        frame_rotation_2pi(-sub_p, drive_qubit.xy.name)
                                        frame_rotation_2pi(-sub_p, cancel_elem)
                            else:
                                wait(4, probe_qubit.xy.name)

                        align(*elems)

                        # Tomography basis pre-rotation on probe_qubit
                        b_lower = str(b_str).lower()
                        if b_lower == "x":
                            probe_qubit.xy.play("-y90")
                        elif b_lower == "y":
                            probe_qubit.xy.play("x90")

                        align()
                        if use_state_discrimination:
                            probe_qubit.readout_state(state[0])
                            save(state[0], state_st[0])
                            if read_drive:
                                drive_qubit.readout_state(drive_state)
                                save(drive_state, drive_st)
                        else:
                            probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                            if read_drive:
                                drive_qubit.resonator.measure("readout", qua_vars=(drive_I, drive_Q))
                                save(drive_I, drive_st)
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


@register
class CrosstalkCancellationTomography(_CoreTomography):
    """Concrete Crosstalk Cancellation Tomography running on QM OPX/QUAM hardware."""

    def _select_pair(self):
        machine = self.backend.machine
        probe_name, drive_name = self.params.probe_qubit, self.params.drive_qubit
        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        return q_dict.get(probe_name, qubits[0]), q_dict.get(drive_name, qubits[1])

    def _sub_phase_rate_turns(self, probe_qubit, drive_qubit) -> float:
        """Leak phase advance per pulse in the probe frame, in turns wrapped to (-0.5, 0.5]."""
        if self.params.phase_rate is not None:
            phase_rate_rad = float(self.params.phase_rate)
        else:
            try:
                duration_ns = float(probe_qubit.xy.operations[self.params.target_gate].length)
            except Exception:
                duration_ns = float(self.params.pulse_duration_ns)
            f_drive = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None)
            f_probe = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None)
            phase_rate_rad = (
                compute_theoretical_phase_rate(float(f_drive), float(f_probe), duration_ns)
                if f_drive is not None and f_probe is not None
                else 0.0
            )
        turns = float((phase_rate_rad / (2.0 * np.pi)) % 1.0)
        return turns - 1.0 if turns > 0.5 else turns

    def _patch(self, config: dict, probe_qubit, drive_qubit) -> dict:
        """The cancel element(s), built and ordered as crosstalk_compensated_sqrb builds them."""
        pairs = [(probe_qubit, drive_qubit)] + ([(drive_qubit, probe_qubit)] if self.params.bidirectional else [])
        for victim, aggressor in sorted(pairs, key=lambda p: p[0].xy.name):
            config = _patch_cancel_element(config, victim, aggressor)
        return config

    def patch_preview_config(self, config: dict) -> dict:
        import copy

        probe_qubit, drive_qubit = self._select_pair()
        return self._patch(copy.deepcopy(config), probe_qubit, drive_qubit)

    def _program_kwargs(self, probe_qubit, drive_qubit) -> dict:
        reset_type = getattr(self.params, "reset_method", None) or getattr(self.params, "reset_type", "thermal")
        return dict(
            target_gate=self.params.target_gate,
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            sub_phase_rate_turns=self._sub_phase_rate_turns(probe_qubit, drive_qubit),
            reset_type=reset_type,
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
            bidirectional=bool(self.params.bidirectional),
            read_drive=bool(self.params.read_drive),
        )

    def preview_program(self):
        """One small representative program (compensated, N = 0 and 2, z basis)."""
        machine = self.backend.machine
        probe_qubit, drive_qubit = self._select_pair()
        prog, _ = build_program(
            machine, probe_qubit, drive_qubit, repetitions=[0, 2], conditions=["compensated"],
            bases=["z"], num_shots=min(int(self.params.num_averages), 10),
            **self._program_kwargs(probe_qubit, drive_qubit),
        )
        return prog

    def probe(
        self,
        timeout: int = 180,
        log: Optional[Callable] = None,
        config: Optional[dict] = None,
    ) -> xr.Dataset:
        machine = self.backend.machine
        probe_qubit, drive_qubit = self._select_pair()
        if config is None:
            config = machine.generate_config()
        config = self._patch(config, probe_qubit, drive_qubit)

        reps = self.params.get_repetitions().tolist()
        conditions = self.params.get_conditions()
        bases = self.params.get_bases()
        num_shots = int(self.params.num_averages)
        batch_size = max(1, getattr(self.params, "batch_size", 4))
        batches = [reps[i : i + batch_size] for i in range(0, len(reps), batch_size)]
        if len(batches) > 1:
            print(f"# [Tomography] Executing {len(reps)} repetition points in {len(batches)} batches (batch_size={batch_size})...")

        datasets: list[xr.Dataset] = []
        for b_idx, batch in enumerate(batches):
            prog, sweep_axes = build_program(
                machine, probe_qubit, drive_qubit, repetitions=batch, conditions=conditions,
                bases=bases, num_shots=num_shots, **self._program_kwargs(probe_qubit, drive_qubit),
            )
            datasets.append(_acquire(
                machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config
            ))
            if len(batches) > 1:
                print(f"# [Tomography Batch {b_idx + 1}/{len(batches)}] Repetitions {batch} completed.")
        return datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="repetitions")
