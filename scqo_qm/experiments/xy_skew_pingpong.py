"""xy_skew_pingpong probe (QM/QUAM): the QUA half of SCQO's XySkewPingpong (contrib, being hardware-tested).

The physics (pulse definitions, modes, checks), the waveform math and the simulator live in
``scqo.experiments.xy_skew_pingpong``; the analysis and figures in scqat's XySkewPingpongEstimator.
This module builds the elements and ops into the generated config and the QUA programs.

Elements: L1 is the sensor's own xy element; L2 is ``<source xy>_via``, built by the crosstalk
cancel-element builder (victim = source, aggressor = sensor: a copy of the source's element with its IF
moved to the sensor's frequency, on the next free core of the FEM) and renamed. Every op of every mode is
always added, so all modes run on identical elements and cores; every shot starts with
crosstalk_compensated_sqrb.phase_reset and reads only the sensor (the phase reference is per core, and
reading a qubit slips its core's phase, chipQM 2026-09-30).

Every shot aligns both elements AFTER phase_reset: without it L2 started 4 cycles (16 ns) ahead of
L1 on hardware (debug_timestamps, 2026-10-19), a skew outside any sensible scan.

Waveforms are pre-computed ops (one per tau_sw, kflip, sign), so the programs carry no amplitude
variables and no real-time branches (2026-10-05: a play in a real-time branch hung the job).

debug_timestamps (scan): ONE small program (first tau_sw, <= 10 shots) whose plays carry
timestamp_stream ts_y90 / ts_l1 / ts_l2 (4 ns cycles); the probe fetches them itself and stores them
next to the population, to check that the two lines play in lockstep.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from scqo import register
from scqo.experiments.xy_skew_pingpong import (
    BASES,
    CAL_CONDITIONS,
    FLIP_CONDITIONS,
    MAX_VIA_AMPLITUDE,
    NULL_CONDITION,
    XySkewPingpong as _CoreXySkewPingpong,
    axis_factor,
    null_waveforms,
    scan_waveforms,
    scan_width,
)
from scqo_qm.experiments._lib import acquire as _acquire, select_qubits, shared_qm
from .crosstalk_compensated_sqrb import _op_iq, _patch_cancel_element, phase_reset

#: timestamp streams of debug_timestamps
TS_STREAMS = ("ts_y90", "ts_l1", "ts_l2")
DEBUG_MAX_SHOTS = 10


def _add_op(config: dict, element: str, op: str, z: np.ndarray) -> None:
    name = f"{element}.{op}"
    config["waveforms"][f"{name}.wf.I"] = {"type": "arbitrary", "samples": [float(v) for v in np.real(z)]}
    config["waveforms"][f"{name}.wf.Q"] = {"type": "arbitrary", "samples": [float(v) for v in np.imag(z)]}
    config["pulses"][f"{name}.pulse"] = {
        "operation": "control", "length": int(len(z)), "waveforms": {"I": f"{name}.wf.I", "Q": f"{name}.wf.Q"},
    }
    config["elements"][element]["operations"][op] = f"{name}.pulse"


def _kname(kflip: int) -> str:
    return {1: "p", -1: "m", 0: "0"}[int(kflip)]


def scan_op(i_tau: int, kflip: int, sign: int) -> str:
    return f"skew_t{i_tau}_k{_kname(kflip)}_s{'p' if sign > 0 else 'm'}"


def flip_op(sign: int) -> str:
    """L1's Y part alone (the l1_flip condition), + or - waveform."""
    return f"skewflip_s{'p' if sign > 0 else 'm'}"


def via_element_name(source_qubit) -> str:
    return f"{source_qubit.xy.name}_via"


def patch_config(
    config: dict, sensor_qubit, source_qubit, *, k: float, c_amp: float, via_phase: float, tau_sw_ns: list[float],
    pad_ns: int, cal_via_scale: float, cal_null_k: float = 0.03,
) -> dict:
    """Add `<source xy>_via` (L2's port, the sensor's frequency, next free core) and every op any mode plays."""
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    if via not in config["elements"]:
        probe = copy.deepcopy(config)
        probe = _patch_cancel_element(probe, source_qubit, sensor_qubit, tone="aggressor")
        elem = probe["elements"].pop(f"{source_qubit.xy.name}_cancel")
        elem["operations"] = {}
        config["elements"][via] = elem

    x180 = _op_iq(config, sensor_xy, "x180")
    y180 = _op_iq(config, sensor_xy, "y180")
    if len(x180) != len(y180):
        raise ValueError(f"{sensor_xy}: x180 and y180 differ in length")
    ox = np.real(x180)
    peak_via = max(k / c_amp, cal_via_scale, cal_null_k / c_amp) * float(np.max(np.abs(ox)))
    if peak_via > MAX_VIA_AMPLITUDE:
        raise ValueError(
            f"L2 would need amplitude {peak_via:.2f} (k {k} / c_amp {c_amp} x envelope peak "
            f"{np.max(np.abs(ox)):.3f}); keep it below {MAX_VIA_AMPLITUDE}: lower k"
        )
    for i, tau in enumerate(tau_sw_ns):
        for kflip in (1, -1, 0):
            for sign in (1, -1):
                l1, l2 = scan_waveforms(
                    x180, y180, k=k, c_amp=c_amp, via_phase=via_phase, tau_ns=tau, pad=pad_ns, kflip=kflip, sign=sign
                )
                _add_op(config, sensor_xy, scan_op(i, kflip, sign), l1)
                if kflip != 0:
                    _add_op(config, via, scan_op(i, kflip, sign), l2)
    _add_op(config, via, "skewcal_0", cal_via_scale * ox.astype(complex))
    _add_op(config, via, "skewcal_90", cal_via_scale * 1j * ox.astype(complex))
    n1, n2 = null_waveforms(x180, y180, k=cal_null_k, c_amp=c_amp, via_phase=via_phase)
    _add_op(config, sensor_xy, "skewnull", n1)
    _add_op(config, via, "skewnull", n2)
    y_part = -k * axis_factor(x180, y180) * ox.astype(complex)
    for sign in (1, -1):
        _add_op(config, sensor_xy, flip_op(sign), sign * y_part)
    return config


def scan_pulse_cycles(n_samples: int, pad_ns: int) -> int:
    return scan_width(n_samples, pad_ns) // 4


# ----------------------------------------------------------------------------- QUA programs
def _readout(qubit, use_state_discrimination, state, state_st, I, I_st, Q, Q_st):
    if use_state_discrimination:
        qubit.readout_state(state)
        save(state, state_st)
    else:
        qubit.resonator.measure("readout", qua_vars=(I, Q))
        save(I, I_st)
        save(Q, Q_st)


def build_scan_program(
    machine, sensor_qubit, source_qubit, *, tau_indices: list[int], tau_values: list[float], repetitions: list[int],
    kflips: list[int], pulse_cycles: int, num_shots: int, reset_type: str = "thermal", reset_max_attempts: int = 1,
    use_state_discrimination: bool = False, simulate: bool = False, debug_timestamps: bool = False,
):
    """y90 - [C(+), C(-)] x N - read, for every (tau_sw, N, kflip), interleaved inside one shot loop.

    The two lines play inside strict_timing_ from one align, each pulse back to back on both elements
    (L2 waits where it is silent). debug_timestamps stamps every play (TS_STREAMS).
    """
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    elems = [sensor_xy, via]
    try:
        y90_cycles = max(sensor_qubit.xy.operations["y90"].length // 4, 1)
    except Exception:
        y90_cycles = 4
    ts = (lambda name: {"timestamp_stream": name}) if debug_timestamps else (lambda name: {})

    sweep_axes = {
        "qubit": xr.DataArray([sensor_qubit.name]),
        "tau_sw_ns": xr.DataArray(np.asarray(tau_values, dtype=float)),
        "repetitions": xr.DataArray(np.asarray(repetitions, dtype=int)),
        "kflip": xr.DataArray(np.asarray(kflips, dtype=int)),
    }
    shape = (len(tau_indices), len(repetitions), len(kflips))

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        state = declare(int)
        state_st = declare_stream()
        machine.initialize_qpu(target=sensor_qubit)
        machine.initialize_qpu(target=source_qubit)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            for i_tau in tau_indices:
                for reps in repetitions:
                    for kflip in kflips:
                        sensor_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
                        source_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
                        phase_reset(elems)
                        # the resets leave the two elements' timelines apart (hardware 2026-10-19: L2 ran
                        # 4 cycles ahead of L1 without this); the SQRB aligns here too
                        align(*elems)
                        with strict_timing_():
                            play("y90", sensor_xy, **ts("ts_y90"))
                            wait(y90_cycles, via)
                            for _ in range(int(reps)):
                                for sign in (1, -1):
                                    play(scan_op(i_tau, kflip, sign), sensor_xy, **ts("ts_l1"))
                                    if kflip != 0:
                                        play(scan_op(i_tau, kflip, sign), via, **ts("ts_l2"))
                                    else:
                                        wait(pulse_cycles, via)
                        align()
                        _readout(sensor_qubit, use_state_discrimination, state, state_st, I[0], I_st[0], Q[0], Q_st[0])
                        align()
            save(n, n_st)

        with stream_processing():
            n_st.save("n")
            if use_state_discrimination:
                state_st.buffer(*shape).average().save("state1")
            else:
                I_st[0].buffer(*shape).average().save("I1")
                Q_st[0].buffer(*shape).average().save("Q1")
    return prog, sweep_axes


def build_calibrate_program(
    machine, sensor_qubit, source_qubit, *, repetitions: list[int], cal_own_scale: float, num_shots: int,
    conditions: tuple[str, ...] = CAL_CONDITIONS, bases: tuple[str, ...] = BASES, flip_tau_index: int = 0,
    scan_cycles: int = 6, reset_type: str = "thermal", reset_max_attempts: int = 1,
    use_state_discrimination: bool = False, simulate: bool = False,
):
    """Sensor tomography after N pulses of one condition, from |0>.

    own_x / own_y / via_0 / via_90 / null play one op N times; l1_flip / l2_flip alternate the + and -
    waveform of one line (L2: the scan op at flip_tau_index, scan_cycles long) while the other line waits.
    Bases are a QUA loop with the pre-rotations amplitude-switched 1/0 (a play in a real-time branch hung
    the job, 2026-10-05); conditions and N are unrolled with fixed plays.
    """
    unknown = [c for c in conditions if c not in (*CAL_CONDITIONS, NULL_CONDITION, *FLIP_CONDITIONS)]
    if unknown:
        raise ValueError(f"unknown conditions {unknown}")
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    elems = [sensor_xy, via]
    try:
        cycles = max(sensor_qubit.xy.operations["x180"].length // 4, 1)
    except Exception:
        cycles = 4

    sweep_axes = {
        "qubit": xr.DataArray([sensor_qubit.name]),
        "condition": xr.DataArray(np.asarray(conditions, dtype=str)),
        "repetitions": xr.DataArray(np.asarray(repetitions, dtype=int)),
        "basis": xr.DataArray(np.asarray(bases, dtype=str)),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        state = declare(int)
        state_st = declare_stream()
        pre_x_amps = declare(fixed, value=[1.0 if b == "x" else 0.0 for b in bases])
        pre_y_amps = declare(fixed, value=[1.0 if b == "y" else 0.0 for b in bases])
        b_idx = declare(int)
        pre_x, pre_y = declare(fixed), declare(fixed)
        machine.initialize_qpu(target=sensor_qubit)
        machine.initialize_qpu(target=source_qubit)
        align()

        for cond in conditions:
            for reps in repetitions:
                with for_(b_idx, 0, b_idx < len(bases), b_idx + 1):
                    assign(pre_x, pre_x_amps[b_idx])
                    assign(pre_y, pre_y_amps[b_idx])
                    with for_(n, 0, n < num_shots, n + 1):
                        sensor_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
                        source_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
                        phase_reset(elems)
                        # the resets leave the two elements' timelines apart (hardware 2026-10-19: L2 ran
                        # 4 cycles ahead of L1 without this); the SQRB aligns here too
                        align(*elems)
                        with strict_timing_():
                            if int(reps) == 0:
                                wait(4, sensor_xy)
                            for j in range(int(reps)):
                                sign = 1 if j % 2 == 0 else -1
                                if cond == "own_x":
                                    play("x180" * amp(float(cal_own_scale)), sensor_xy)
                                    wait(cycles, via)
                                elif cond == "own_y":
                                    play("y180" * amp(float(cal_own_scale)), sensor_xy)
                                    wait(cycles, via)
                                elif cond == NULL_CONDITION:
                                    play("skewnull", sensor_xy)
                                    play("skewnull", via)
                                elif cond == "l1_flip":
                                    play(flip_op(sign), sensor_xy)
                                    wait(cycles, via)
                                elif cond == "l2_flip":
                                    play(scan_op(flip_tau_index, 1, sign), via)
                                    wait(scan_cycles, sensor_xy)
                                else:
                                    play("skewcal_0" if cond == "via_0" else "skewcal_90", via)
                                    wait(cycles, sensor_xy)
                        align(*elems)
                        sensor_qubit.xy.play("-y90", amplitude_scale=pre_x)
                        sensor_qubit.xy.play("x90", amplitude_scale=pre_y)
                        align()
                        _readout(sensor_qubit, use_state_discrimination, state, state_st, I[0], I_st[0], Q[0], Q_st[0])
                        align()

        with stream_processing():
            if use_state_discrimination:
                state_st.buffer(num_shots).map(FUNCTIONS.average()).buffer(len(bases)).buffer(len(repetitions)).buffer(len(conditions)).save("state1")
            else:
                I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(bases)).buffer(len(repetitions)).buffer(len(conditions)).save("I1")
                Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(bases)).buffer(len(repetitions)).buffer(len(conditions)).save("Q1")
    return prog, sweep_axes


def _fetch_values(handles, name: str) -> np.ndarray:
    h = handles.get(name)
    if h is None:
        return np.zeros(0, dtype=np.int64)
    arr = np.asarray(h.fetch_all())
    if arr.dtype.names:
        arr = arr["value"]
    return np.asarray(arr).ravel()


def acquire_with_timestamps(qm, prog, sweep_axes, *, use_state_discrimination: bool) -> xr.Dataset:
    """Run a debug_timestamps scan program and return its averaged data plus the raw timestamp streams."""
    job = qm.execute(prog)
    handles = job.result_handles
    handles.wait_for_all_values()
    dims = ("qubit", "tau_sw_ns", "repetitions", "kflip")
    shape = tuple(len(sweep_axes[d]) for d in dims)
    coords = {d: sweep_axes[d].values for d in dims}
    names = ("state1",) if use_state_discrimination else ("I1", "Q1")
    data = {}
    for name in names:
        arr = np.asarray(handles.get(name).fetch_all(), dtype=float).reshape(shape)
        data[name[:-1]] = (dims, arr)
    for ts_name in TS_STREAMS:
        values = _fetch_values(handles, ts_name)
        data[ts_name] = (f"{ts_name}_index", values.astype(np.int64))
    return xr.Dataset(data, coords=coords)


@register
class QMXySkewPingpong(_CoreXySkewPingpong):
    """xy_skew_pingpong on QM OPX/QUAM hardware. Thermal reset only (active reset is validated for SQRB only)."""

    def _select_pair(self):
        machine = self.backend.machine
        qubits = select_qubits(machine, [self.params.sensor_qubit, self.params.source_qubit], multiplexed=True)
        q = {x.name: x for x in qubits}
        return q[self.params.sensor_qubit], q[self.params.source_qubit]

    def _patch(self, config: dict, sensor_qubit, source_qubit) -> dict:
        p = self.params
        return patch_config(
            config, sensor_qubit, source_qubit, k=p.k, c_amp=p.c_amp, via_phase=p.via_phase,
            tau_sw_ns=list(p.tau_sw_ns), pad_ns=p.pad_ns, cal_via_scale=p.cal_via_scale, cal_null_k=p.cal_null_k,
        )

    def patch_preview_config(self, config: dict) -> dict:
        sensor_qubit, source_qubit = self._select_pair()
        return self._patch(copy.deepcopy(config), sensor_qubit, source_qubit)

    def _program_kwargs(self, sensor_qubit, source_qubit) -> dict:
        from ._reset import check_reset_method, reset_max_attempts

        reset_type = check_reset_method(self, qubits=[sensor_qubit.name, source_qubit.name])
        return dict(
            reset_type=reset_type,
            reset_max_attempts=reset_max_attempts(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
        )

    def _num_shots(self) -> int:
        n = int(self.params.num_averages)
        return min(n, DEBUG_MAX_SHOTS) if (self.params.mode == "scan" and self.params.debug_timestamps) else n

    def _programs(self, config: dict, sensor_qubit, source_qubit, num_shots: int):
        """[(program, sweep_axes)]: one calibration program, one scan program per tau batch, or the one
        debug_timestamps program."""
        p = self.params
        kw = self._program_kwargs(sensor_qubit, source_qubit)
        n_samples = len(_op_iq(config, sensor_qubit.xy.name, "x180"))
        cycles = scan_pulse_cycles(n_samples, p.pad_ns)
        if p.mode == "calibrate":
            return [build_calibrate_program(
                self.backend.machine, sensor_qubit, source_qubit, repetitions=list(p.repetitions),
                cal_own_scale=p.cal_own_scale, num_shots=num_shots, conditions=tuple(p.cal_conditions()),
                flip_tau_index=p.flip_tau_index(), scan_cycles=cycles, **kw,
            )]
        idx = list(range(len(p.tau_sw_ns)))
        if p.debug_timestamps:
            batches = [idx[:1]]
        else:
            batches = [idx[i:i + p.tau_batch_size] for i in range(0, len(idx), p.tau_batch_size)]
        return [
            build_scan_program(
                self.backend.machine, sensor_qubit, source_qubit, tau_indices=b,
                tau_values=[p.tau_sw_ns[i] for i in b], repetitions=list(p.repetitions), kflips=p.kflips(),
                pulse_cycles=cycles, num_shots=num_shots, debug_timestamps=p.debug_timestamps, **kw,
            )
            for b in batches
        ]

    def preview_program(self):
        """The first program of the run, with at most 10 shots."""
        machine = self.backend.machine
        sensor_qubit, source_qubit = self._select_pair()
        config = self._patch(machine.generate_config(), sensor_qubit, source_qubit)
        prog, _ = self._programs(config, sensor_qubit, source_qubit, min(self._num_shots(), 10))[0]
        return prog

    def probe(self, timeout: int = 180, log: Optional[Callable] = None, config: Optional[dict] = None) -> xr.Dataset:
        machine = self.backend.machine
        sensor_qubit, source_qubit = self._select_pair()
        if config is None:
            config = machine.generate_config()
        config = self._patch(config, sensor_qubit, source_qubit)
        num_shots = self._num_shots()
        datasets = []
        with shared_qm(machine, config, timeout=timeout) as qm:
            for prog, axes in self._programs(config, sensor_qubit, source_qubit, num_shots):
                if self.params.mode == "scan" and self.params.debug_timestamps:
                    ds = acquire_with_timestamps(qm, prog, axes,
                                                 use_state_discrimination=bool(self.params.use_state_discrimination))
                    _print_timestamp_summary(ds)
                    datasets.append(ds)
                else:
                    datasets.append(_acquire(machine, prog, axes, num_shots=num_shots, timeout=timeout, log=log,
                                             config=config, qm=qm))
        return datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="tau_sw_ns")


def _print_timestamp_summary(ds: xr.Dataset) -> None:
    """First shot's y90 / L1 / L2 start cycles, and the L2 - L1 offset of each paired play."""
    t1 = ds["ts_l1"].values
    t2 = ds["ts_l2"].values
    ty = ds["ts_y90"].values
    print(f"# [xy_skew_pingpong] timestamps (4 ns cycles): y90 {ty[:3].tolist()} ... ({ty.size} total)")
    print(f"#   L1 {t1[:8].tolist()} ... ({t1.size} total)")
    print(f"#   L2 {t2[:8].tolist()} ... ({t2.size} total)")
    if t2.size:
        print("#   note: L2 plays only at kflip +-1; pair them with L1's kflip +-1 plays when reading offsets")
