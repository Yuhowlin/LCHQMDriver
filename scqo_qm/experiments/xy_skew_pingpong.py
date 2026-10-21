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

sensor_transition='ef': the sensor's |1>-|2> transition senses instead of |0>-|1> - same qubit, same
place, a frequency lower by the anharmonicity. L1 then plays on ``<sensor xy>_ef`` (a copy of the sensor's
element at f12, next free core) with the sensor's EF_x180 / EF_x90 pulses; L2's via element plays at f12.
Every shot prepares |1> with the sensor's ge x180 and ends with another ge x180 that maps the EF result
onto the ge readout (|1> -> |0>, |2> stays and reads as excited), so P1 read = p2 and S keeps its form.
envelope_length_ns: the scan / calibration pulses use a DragCosine of that length with the 16 ns pulse's
area and DRAG coefficient (both lines share it); the y90 preparation and basis pre-rotations stay as they are.
Both default off: the ge, 16 ns programs are unchanged.

mode='ef_cal': one self-fetched program (spectroscopy with update_frequency on the f12 element, an
amplitude Rabi with amp(), and the |0>/|1>/|2> readout check); the f12 guess, the EF_x180 envelope peak and
f01 ride the dataset attrs for the analysis.

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
    EF_ATTR_CENTER,
    EF_ATTR_F01,
    EF_ATTR_PEAK,
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
from .crosstalk_compensated_sqrb import _assign_free_core, _op_iq, _patch_cancel_element, phase_reset

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


def l1_op(kflip: int, sign: int) -> str:
    """L1's scan pulse: it does not depend on tau_sw (only L2 is shifted), so one op per (kflip, sign)."""
    return f"skewl1_k{_kname(kflip)}_s{'p' if sign > 0 else 'm'}"


def flip_op(sign: int) -> str:
    """L1's Y part alone (the l1_flip condition), + or - waveform."""
    return f"skewflip_s{'p' if sign > 0 else 'm'}"


def via_element_name(source_qubit) -> str:
    return f"{source_qubit.xy.name}_via"


def ef_element_name(sensor_qubit) -> str:
    return f"{sensor_qubit.xy.name}_ef"


def sensor_anharmonicity(sensor_qubit, override_hz=None) -> float:
    """f01 - f12 (Hz, positive): the override, else the QUAM qubit's anharmonicity."""
    value = override_hz if override_hz is not None else getattr(sensor_qubit, "anharmonicity", None)
    if value is None:
        raise ValueError(f"{sensor_qubit.name}: no anharmonicity for the EF sensor (pass ef_anharmonicity_hz)")
    return abs(float(value))


def stretch_drag_cosine(x180: np.ndarray, length: int) -> np.ndarray:
    """A DragCosine of `length` samples with x180's area and DRAG coefficient (I = area/L (1 - cos), Q = beta I')."""
    if length < 16 or length % 4:
        raise ValueError(f"envelope_length_ns {length} must be a multiple of 4, at least 16")
    i0, q0 = np.real(x180), np.imag(x180)
    d0 = np.gradient(i0)
    beta = float(np.dot(q0, d0) / np.dot(d0, d0)) if np.dot(d0, d0) else 0.0
    t = np.arange(length)
    i1 = float(i0.sum()) / length * (1 - np.cos(2 * np.pi * t / length))
    return i1 + 1j * beta * np.gradient(i1)


def sensor_drive(sensor_qubit, *, transition: str = "ge", envelope_length_ns=None, ge_length: int = 16,
                 ef_length: int = 16) -> dict:
    """Which element and ops a program plays on the sensor's side (matches patch_config's naming).

    ge, no envelope_length: exactly the original program (QUAM y90 / x180 / y180 / pre-rotation macros).
    """
    sensor_xy = sensor_qubit.xy.name
    ef = transition == "ef"
    custom = ef or envelope_length_ns is not None
    own_len = int(envelope_length_ns) if envelope_length_ns is not None else (ef_length if ef else ge_length)
    try:
        y90_len = sensor_qubit.xy.operations["EF_x90" if ef else "y90"].length
    except Exception:
        y90_len = 16
    return {
        "play": ef_element_name(sensor_qubit) if ef else sensor_xy,
        "extra": [ef_element_name(sensor_qubit)] if ef else [],
        "ef": ef,
        "y90": "skewy90" if ef else "y90",
        "y90_cycles": max(int(y90_len) // 4, 1),
        "own_x": "skewown_x" if custom else "x180",
        "own_y": "skewown_y" if custom else "y180",
        "own_cycles": max(own_len // 4, 1),
        "pre_my90": "skewpre_my90" if ef else None,
        "pre_x90": "skewpre_x90" if ef else None,
        "pulse_len": own_len,
    }


def patch_config(
    config: dict, sensor_qubit, source_qubit, *, k: float, c_amp: float, via_phase: float, tau_sw_ns: list[float],
    pad_ns: int, cal_via_scale: float, cal_null_k: float = 0.03, transition: str = "ge",
    ef_anharmonicity_hz=None, envelope_length_ns=None, ef_spec_length_ns: int = 400, ef_spec_rotation: float = 1 / 3,
    build_via: bool = True,
) -> dict:
    """Add `<source xy>_via` (L2's port, the sensor's frequency, next free core) and every op any mode plays.

    transition 'ef' also adds `<sensor xy>_ef` (the sensor's element at f12, next free core, created after
    the via element so the via keeps its core) and moves the via element to f12. build_via=False (ef_cal,
    which never plays L2) leaves the source's line without any added element and builds only the sensor's ops.
    """
    if transition not in ("ge", "ef"):
        raise ValueError(f"unknown sensor transition {transition!r}")
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    ef = transition == "ef"
    anh = sensor_anharmonicity(sensor_qubit, ef_anharmonicity_hz) if ef else 0.0
    if build_via and via not in config["elements"]:
        probe = copy.deepcopy(config)
        probe = _patch_cancel_element(probe, source_qubit, sensor_qubit, tone="aggressor")
        elem = probe["elements"].pop(f"{source_qubit.xy.name}_cancel")
        elem["operations"] = {}
        if ef:
            elem["intermediate_frequency"] = float(elem["intermediate_frequency"]) - anh
        config["elements"][via] = elem
    play_elem = sensor_xy
    if ef:
        play_elem = ef_element_name(sensor_qubit)
        if play_elem not in config["elements"]:
            elem = copy.deepcopy(config["elements"][sensor_xy])
            elem["operations"] = {}
            elem["intermediate_frequency"] = float(elem["intermediate_frequency"]) - anh
            _assign_free_core(config, elem)
            config["elements"][play_elem] = elem
        for name in ((via, play_elem) if build_via else (play_elem,)):
            if abs(float(config["elements"][name]["intermediate_frequency"])) > 400e6:
                raise ValueError(f"{name}: IF {config['elements'][name]['intermediate_frequency'] / 1e6:.1f} MHz "
                                 "at f12 is outside the MW-FEM +-400 MHz band")

    ge_x180 = _op_iq(config, sensor_xy, "x180")
    ge_y180 = _op_iq(config, sensor_xy, "y180")
    if len(ge_x180) != len(ge_y180):
        raise ValueError(f"{sensor_xy}: x180 and y180 differ in length")
    yax = axis_factor(ge_x180, ge_y180)
    if ef:
        x180 = _op_iq(config, sensor_xy, "EF_x180")
        ef_x90 = _op_iq(config, sensor_xy, "EF_x90")
        y180 = yax * x180
    else:
        x180, y180 = ge_x180, ge_y180
    if envelope_length_ns is not None:
        x180 = stretch_drag_cosine(x180, int(envelope_length_ns))
        y180 = yax * x180
    if ef or envelope_length_ns is not None:
        _add_op(config, play_elem, "skewown_x", x180)
        _add_op(config, play_elem, "skewown_y", y180)
    if ef:
        # ef_cal: the current EF pi, its envelope with the DRAG off (Rabi), and a weak long line probe
        ef_x180 = _op_iq(config, sensor_xy, "EF_x180")
        _add_op(config, play_elem, "skewefpi", ef_x180)
        _add_op(config, play_elem, "skewefrabi", np.real(ef_x180).astype(complex))
        t = np.arange(int(ef_spec_length_ns))
        area = float(np.real(ef_x180).sum()) * float(ef_spec_rotation)
        _add_op(config, play_elem, "skewefspec",
                (area / len(t) * (1 - np.cos(2 * np.pi * t / len(t)))).astype(complex))
        _add_op(config, play_elem, "skewy90", yax * ef_x90)
        _add_op(config, play_elem, "skewpre_x90", ef_x90)
        _add_op(config, play_elem, "skewpre_my90", -yax * ef_x90)
    ox = np.real(x180)
    peak_via = max(k / c_amp, cal_via_scale, cal_null_k / c_amp) * float(np.max(np.abs(ox)))
    if peak_via > MAX_VIA_AMPLITUDE:
        raise ValueError(
            f"L2 would need amplitude {peak_via:.2f} (k {k} / c_amp {c_amp} x envelope peak "
            f"{np.max(np.abs(ox)):.3f}); keep it below {MAX_VIA_AMPLITUDE}: lower k"
        )
    for kflip in (1, -1, 0):
        for sign in (1, -1):
            l1, _ = scan_waveforms(
                x180, y180, k=k, c_amp=c_amp, via_phase=via_phase, tau_ns=0.0, pad=pad_ns, kflip=kflip, sign=sign
            )
            _add_op(config, play_elem, l1_op(kflip, sign), l1)
    if not build_via:
        n1, _ = null_waveforms(x180, y180, k=cal_null_k, c_amp=c_amp, via_phase=via_phase)
        _add_op(config, play_elem, "skewnull", n1)
        y_part = -k * axis_factor(x180, y180) * ox.astype(complex)
        for sign in (1, -1):
            _add_op(config, play_elem, flip_op(sign), sign * y_part)
        return config
    for i, tau in enumerate(tau_sw_ns):
        for kflip in (1, -1):
            for sign in (1, -1):
                _, l2 = scan_waveforms(
                    x180, y180, k=k, c_amp=c_amp, via_phase=via_phase, tau_ns=tau, pad=pad_ns, kflip=kflip, sign=sign
                )
                _add_op(config, via, scan_op(i, kflip, sign), l2)
    _add_op(config, via, "skewcal_0", cal_via_scale * ox.astype(complex))
    _add_op(config, via, "skewcal_90", cal_via_scale * 1j * ox.astype(complex))
    n1, n2 = null_waveforms(x180, y180, k=cal_null_k, c_amp=c_amp, via_phase=via_phase)
    _add_op(config, play_elem, "skewnull", n1)
    _add_op(config, via, "skewnull", n2)
    y_part = -k * axis_factor(x180, y180) * ox.astype(complex)
    for sign in (1, -1):
        _add_op(config, play_elem, flip_op(sign), sign * y_part)
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
    layout: str = "loops", drive: Optional[dict] = None,
):
    """y90 - [C(+), C(-)] x N - read, for every (tau_sw, N, kflip), interleaved inside one shot loop.

    layout 'loops' (default): tau_sw and kflip unrolled, N a QUA for_each_, and the N pairs one QUA for_
    loop PER ELEMENT on its own loop variable after a common align - the SQRB's gap-free pattern
    (hardware 2026-10-01). The program is ~2N x smaller and N costs no compile time. No strict_timing_
    around the loops: check the lockstep with debug_timestamps.
    layout 'unrolled': every pulse unrolled inside strict_timing_ (the first hardware version, for A/B).
    debug_timestamps stamps every play (TS_STREAMS). drive: sensor_drive() (default: ge, the original program).
    """
    if drive is None:
        drive = sensor_drive(sensor_qubit)
    if layout == "loops":
        return _build_scan_program_loops(
            machine, sensor_qubit, source_qubit, tau_indices=tau_indices, tau_values=tau_values,
            repetitions=repetitions, kflips=kflips, num_shots=num_shots, reset_type=reset_type,
            reset_max_attempts=reset_max_attempts, use_state_discrimination=use_state_discrimination,
            simulate=simulate, debug_timestamps=debug_timestamps, drive=drive,
        )
    if layout != "unrolled":
        raise ValueError(f"unknown program layout {layout!r}")
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    elems = [sensor_xy, *drive["extra"], via]
    l1_elem, y90_cycles = drive["play"], drive["y90_cycles"]
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
                        if drive["ef"]:
                            play("x180", sensor_xy)
                            align(*elems)
                        with strict_timing_():
                            play(drive["y90"], l1_elem, **ts("ts_y90"))
                            wait(y90_cycles, via)
                            for _ in range(int(reps)):
                                for sign in (1, -1):
                                    play(l1_op(kflip, sign), l1_elem, **ts("ts_l1"))
                                    if kflip != 0:
                                        play(scan_op(i_tau, kflip, sign), via, **ts("ts_l2"))
                                    else:
                                        wait(pulse_cycles, via)
                        align()
                        if drive["ef"]:
                            play("x180", sensor_xy)
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


def _build_scan_program_loops(
    machine, sensor_qubit, source_qubit, *, tau_indices, tau_values, repetitions, kflips, num_shots, reset_type,
    reset_max_attempts, use_state_discrimination, simulate, debug_timestamps, drive,
):
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    elems = [sensor_xy, *drive["extra"], via]
    l1_elem, y90_cycles = drive["play"], drive["y90_cycles"]
    ts = (lambda name: {"timestamp_stream": name}) if debug_timestamps else (lambda name: {})

    # saved in (tau, kflip, N) order: the axes follow it, and the backend maps them by NAME
    sweep_axes = {
        "qubit": xr.DataArray([sensor_qubit.name]),
        "tau_sw_ns": xr.DataArray(np.asarray(tau_values, dtype=float)),
        "kflip": xr.DataArray(np.asarray(kflips, dtype=int)),
        "repetitions": xr.DataArray(np.asarray(repetitions, dtype=int)),
    }
    shape = (len(tau_indices), len(kflips), len(repetitions))

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        state = declare(int)
        state_st = declare_stream()
        n_pairs = declare(int)
        j1, j2 = declare(int), declare(int)
        machine.initialize_qpu(target=sensor_qubit)
        machine.initialize_qpu(target=source_qubit)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            for i_tau in tau_indices:
                for kflip in kflips:
                    with for_each_(n_pairs, [int(r) for r in repetitions]):
                        sensor_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
                        source_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
                        phase_reset(elems)
                        align(*elems)
                        if drive["ef"]:
                            play("x180", sensor_xy)
                            align(*elems)
                        play(drive["y90"], l1_elem, **ts("ts_y90"))
                        wait(y90_cycles, via)
                        # one loop per element, each on its own variable: the loops run side by side
                        with for_(j1, 0, j1 < n_pairs, j1 + 1):
                            play(l1_op(kflip, 1), l1_elem, **ts("ts_l1"))
                            play(l1_op(kflip, -1), l1_elem, **ts("ts_l1"))
                        if kflip != 0:
                            with for_(j2, 0, j2 < n_pairs, j2 + 1):
                                play(scan_op(i_tau, kflip, 1), via, **ts("ts_l2"))
                                play(scan_op(i_tau, kflip, -1), via, **ts("ts_l2"))
                        align()
                        if drive["ef"]:
                            play("x180", sensor_xy)
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
    use_state_discrimination: bool = False, simulate: bool = False, drive: Optional[dict] = None,
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
    if drive is None:
        drive = sensor_drive(sensor_qubit)
        try:
            drive["own_cycles"] = max(sensor_qubit.xy.operations["x180"].length // 4, 1)
        except Exception:
            drive["own_cycles"] = 4
    elems = [sensor_xy, *drive["extra"], via]
    l1_elem, cycles = drive["play"], drive["own_cycles"]

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
                        if drive["ef"]:
                            play("x180", sensor_xy)
                            align(*elems)
                        with strict_timing_():
                            if int(reps) == 0:
                                wait(4, l1_elem)
                            for j in range(int(reps)):
                                sign = 1 if j % 2 == 0 else -1
                                if cond == "own_x":
                                    play(drive["own_x"] * amp(float(cal_own_scale)), l1_elem)
                                    wait(cycles, via)
                                elif cond == "own_y":
                                    play(drive["own_y"] * amp(float(cal_own_scale)), l1_elem)
                                    wait(cycles, via)
                                elif cond == NULL_CONDITION:
                                    play("skewnull", l1_elem)
                                    play("skewnull", via)
                                elif cond == "l1_flip":
                                    play(flip_op(sign), l1_elem)
                                    wait(cycles, via)
                                elif cond == "l2_flip":
                                    play(scan_op(flip_tau_index, 1, sign), via)
                                    wait(scan_cycles, l1_elem)
                                else:
                                    play("skewcal_0" if cond == "via_0" else "skewcal_90", via)
                                    wait(cycles, l1_elem)
                        align(*elems)
                        if drive["pre_x90"] is None:
                            sensor_qubit.xy.play("-y90", amplitude_scale=pre_x)
                            sensor_qubit.xy.play("x90", amplitude_scale=pre_y)
                        else:
                            play(drive["pre_my90"] * amp(pre_x), l1_elem)
                            play(drive["pre_x90"] * amp(pre_y), l1_elem)
                        align()
                        if drive["ef"]:
                            play("x180", sensor_xy)
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


def build_ef_cal_program(
    machine, sensor_qubit, source_qubit, *, ef_if_hz: float, detunings_hz: list[int], scales: list[float],
    num_shots: int, reset_type: str = "thermal", reset_max_attempts: int = 1, simulate: bool = False,
):
    """EF spectroscopy, EF amplitude Rabi and the |0>/|1>/|2> readout check, each shot framed by ge x180s.

    Needs state discrimination. Streams: spec / rabi / ro (averaged), n.
    """
    sensor_xy = sensor_qubit.xy.name
    ef = ef_element_name(sensor_qubit)
    if0 = int(round(ef_if_hz))

    def shot_start():
        sensor_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
        source_qubit.reset(reset_type, simulate, max_attempts=reset_max_attempts)
        # a GLOBAL align: without it the sensor started as soon as ITS thermal wait ended, and the readout
        # (after a global align) waited for a longer source wait - the excited sensor decayed in between
        # (2026-10-20: sensor q2/q3 with source q1 read ~0.1 for every state). The other modes sync here
        # through phase_reset's reset_global_phase.
        align()

    def map_and_read(state, stream):
        align(sensor_xy, ef)
        play("x180", sensor_xy)
        align()
        sensor_qubit.readout_state(state)
        save(state, stream)
        align()

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        state = declare(int)
        spec_st, rabi_st, ro_st = declare_stream(), declare_stream(), declare_stream()
        df = declare(int)
        a = declare(fixed)
        machine.initialize_qpu(target=sensor_qubit)
        machine.initialize_qpu(target=source_qubit)
        align()
        with for_(n, 0, n < num_shots, n + 1):
            with for_each_(df, [int(d) for d in detunings_hz]):
                shot_start()
                play("x180", sensor_xy)
                align(sensor_xy, ef)
                update_frequency(ef, df + if0)
                play("skewefspec", ef)
                map_and_read(state, spec_st)
            update_frequency(ef, if0)
            with for_each_(a, [float(x) for x in scales]):
                shot_start()
                play("x180", sensor_xy)
                align(sensor_xy, ef)
                play("skewefrabi" * amp(a), ef)
                map_and_read(state, rabi_st)
            for prep in (0, 1, 2):
                shot_start()
                if prep >= 1:
                    play("x180", sensor_xy)
                    align(sensor_xy, ef)
                if prep == 2:
                    play("skewefpi", ef)
                map_and_read(state, ro_st)
            save(n, n_st)
        with stream_processing():
            n_st.save("n")
            spec_st.buffer(len(detunings_hz)).average().save("spec")
            rabi_st.buffer(len(scales)).average().save("rabi")
            ro_st.buffer(3).average().save("ro")
    return prog


def acquire_ef_cal(qm, prog, *, sensor: str, detunings_hz, scales, attrs: dict) -> xr.Dataset:
    """Run the ef_cal program and return population_spec / _rabi / _ro on a qubit axis, attrs stamped."""
    job = qm.execute(prog)
    handles = job.result_handles
    handles.wait_for_all_values()
    get = lambda name: np.asarray(handles.get(name).fetch_all(), dtype=float).ravel()
    ds = xr.Dataset(
        {
            "population_spec": (("qubit", "ef_detuning_hz"), get("spec")[None, :]),
            "population_rabi": (("qubit", "ef_amp_scale"), get("rabi")[None, :]),
            "population_ro": (("qubit", "prepared_state"), get("ro")[None, :]),
        },
        coords={"qubit": [sensor], "ef_detuning_hz": np.asarray(detunings_hz, float),
                "ef_amp_scale": np.asarray(scales, float), "prepared_state": [0, 1, 2]},
    )
    ds.attrs.update(attrs)
    return ds


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
    dims = tuple(sweep_axes)
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
            tau_sw_ns=list(p.tau_sw_ns), pad_ns=p.pad_ns, cal_via_scale=p.effective_cal_via_scale(),
            cal_null_k=p.cal_null_k, transition="ef" if p.mode == "ef_cal" else p.sensor_transition,
            ef_anharmonicity_hz=p.ef_anharmonicity_hz, envelope_length_ns=p.envelope_length_ns,
            ef_spec_length_ns=p.ef_spec_length_ns, ef_spec_rotation=p.ef_spec_rotation,
            # ef_cal never plays L2, so it adds nothing to the source's line
            build_via=p.mode != "ef_cal",
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
        sensor_xy = sensor_qubit.xy.name
        if p.mode == "ef_cal":
            if not p.use_state_discrimination:
                raise ValueError("ef_cal needs use_state_discrimination=true")
            ef_if = float(config["elements"][ef_element_name(sensor_qubit)]["intermediate_frequency"])
            return [build_ef_cal_program(
                self.backend.machine, sensor_qubit, source_qubit, ef_if_hz=ef_if, detunings_hz=p.ef_detunings_hz(),
                scales=list(p.ef_rabi_scales), num_shots=num_shots, reset_type=kw["reset_type"],
                reset_max_attempts=kw["reset_max_attempts"], simulate=kw["simulate"],
            )]
        drive = sensor_drive(
            sensor_qubit, transition=p.sensor_transition, envelope_length_ns=p.envelope_length_ns,
            ge_length=len(_op_iq(config, sensor_xy, "x180")),
            ef_length=len(_op_iq(config, sensor_xy, "EF_x180")) if p.sensor_transition == "ef" else 16,
        )
        cycles = scan_pulse_cycles(drive["pulse_len"], p.pad_ns)
        if p.mode == "calibrate":
            return [build_calibrate_program(
                self.backend.machine, sensor_qubit, source_qubit, repetitions=list(p.repetitions),
                cal_own_scale=p.cal_own_scale, num_shots=num_shots, conditions=tuple(p.cal_conditions()),
                flip_tau_index=p.flip_tau_index(), scan_cycles=cycles, drive=drive, **kw,
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
                pulse_cycles=cycles, num_shots=num_shots, debug_timestamps=p.debug_timestamps,
                layout=p.program_layout, drive=drive, **kw,
            )
            for b in batches
        ]

    def preview_program(self):
        """The first program of the run, with at most 10 shots."""
        machine = self.backend.machine
        sensor_qubit, source_qubit = self._select_pair()
        config = self._patch(machine.generate_config(), sensor_qubit, source_qubit)
        first = self._programs(config, sensor_qubit, source_qubit, min(self._num_shots(), 10))[0]
        return first if self.params.mode == "ef_cal" else first[0]

    def probe(self, timeout: int = 180, log: Optional[Callable] = None, config: Optional[dict] = None) -> xr.Dataset:
        machine = self.backend.machine
        sensor_qubit, source_qubit = self._select_pair()
        if config is None:
            config = machine.generate_config()
        config = self._patch(config, sensor_qubit, source_qubit)
        num_shots = self._num_shots()
        if self.params.mode == "ef_cal":
            p = self.params
            sensor_xy = sensor_qubit.xy.name
            anh = sensor_anharmonicity(sensor_qubit, p.ef_anharmonicity_hz)
            f01 = float(sensor_qubit.xy.RF_frequency)
            attrs = {EF_ATTR_CENTER: f01 - anh, EF_ATTR_F01: f01,
                     EF_ATTR_PEAK: float(np.max(np.abs(np.real(_op_iq(config, sensor_xy, "EF_x180")))))}
            with shared_qm(machine, config, timeout=timeout) as qm:
                prog = self._programs(config, sensor_qubit, source_qubit, num_shots)[0]
                return acquire_ef_cal(qm, prog, sensor=sensor_qubit.name, detunings_hz=p.ef_detunings_hz(),
                                      scales=list(p.ef_rabi_scales), attrs=attrs)
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
