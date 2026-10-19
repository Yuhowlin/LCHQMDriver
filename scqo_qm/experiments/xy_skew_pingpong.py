"""XY line skew by a DRAG-like ping-pong (QM/QUAM, contrib prototype - not hardware-tested yet).

Measures, with the sensor qubit itself, the arrival-time difference tau at the sensor between its own
xy line (L1) and a second xy line (L2, the source qubit's line) that reaches the sensor by crosstalk.
L2 plays at the SENSOR's frequency, so its leak drives the sensor resonantly.

Per pulse C(s), s = +-1 (the whole waveform sign, both lines together):

    L1:  s * [ x180(t) - kflip * k * Y * Ox(t) ]            x180 = the sensor's own DRAG pulse
    L2:  s * kflip * (k / c_amp) * exp(i via_phase) * Ox(t - tau_sw)

Ox is x180's in-phase envelope, Y the sensor's y180/x180 axis factor (+-i). With c_amp / via_phase
from mode='calibrate', the leak of L2 at the sensor is +kflip*k*Y*Ox(t - tau_sw - tau): the two
Y parts cancel only when tau_sw = -tau. The residual k*Y*[Ox(t - tau_sw - tau) - Ox(t)] is a DRAG-like
derivative term, i.e. a detuning that the sequence

    y90 - [C(+), C(-)] x N - read P1

accumulates over the N pairs while a c_amp / via_phase miscalibration (first order) cancels within each
pair. The signal S = P1(kflip=+1) - P1(kflip=-1) is odd in (tau_sw + tau); its zero crossing tau_sw* is
the alignment point. kflip = 0 (include_background) plays L1's plain x180 pairs only: the sensor's own
DRAG background, which sets the slope's size and SIGN at each N (16 ns pulses turn it fast).

tau_sw is a software shift: each value is a separate pre-computed waveform (band-limited fractional
delay of the 1 GS/s samples), so it cannot change in real time; every (tau_sw, N, kflip) point is unrolled
and interleaved shot by shot inside one shot loop.

mode='calibrate' measures the L2 -> sensor crosstalk at the sensor's frequency by tomography of N
repeated small pulses: own_x / own_y (the sensor's x180 / y180 scaled by cal_own_scale) and via_0 /
via_90 (L2 playing cal_via_scale * Ox at waveform phase 0 / 90 deg). The fitted rotation axes give
via_phase (L2's leak along the sensor's +Y, handedness from via_90) and the rates c_amp.
include_null adds the direct check: 'null' plays the scan's two Y parts alone and together - L1
-cal_null_k * Y * Ox and L2 (cal_null_k / c_amp) e^{i via_phase} Ox, no x180 - with c_amp / via_phase
from a previous calibration. They must cancel: the net rotation over cal_null_k x the sensor's rate
(null_ratio) is ~0 when amplitude and direction are right and ~2 when the direction is flipped. The
residual's size and axis also give refined c_amp / via_phase.

Both modes build the same elements in the same order (the via element takes the next free core on L2's
FEM, as the crosstalk cancel elements do), start every shot with crosstalk_compensated_sqrb.phase_reset and
read only the sensor, so the calibration holds in the scan (the phase reference is per core; reading a
qubit slips its core's phase, chipQM 2026-09-30).

Sign: the reported tau_sw* is the software shift of L2 that aligns it. Do not trust the sign convention
before the hardware check: add +1 ns to L2's port delay and the zero crossing must move by -1 ns.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Callable, ClassVar, Literal, Optional

import numpy as np
import xarray as xr
from pydantic import Field, field_validator, model_validator
from qm.qua import *

from scqo import register
from scqo.contract import ContractError, DatasetContract
from scqo.experiment import Experiment
from scqo.experiments._capabilities.qubit_reset import QubitResetParameters
from scqo.experiments._capabilities.state_readout import POPULATION_ALT, StateReadoutParameters
from scqo.experiments._sim import stable_seed
from scqo.parameters import AveragingParameters, TargetSelection
from scqo.result import Outcome, Result

from scqo_qm.experiments._lib import acquire as _acquire, select_qubits, shared_qm
from .crosstalk_compensated_sqrb import _op_iq, _patch_cancel_element, phase_reset

MODES = ("scan", "calibrate")
CAL_CONDITIONS = ("own_x", "own_y", "via_0", "via_90")
NULL_CONDITION = "null"
BASES = ("z", "x", "y")
#: largest |L2 waveform sample| allowed (MW-FEM waveform range is [-1, 1))
MAX_VIA_AMPLITUDE = 0.95
#: skew the offline simulator puts on the line (ns) and its crosstalk (amp, axis)
SIM_SKEW_NS = 0.35
SIM_C_AMP = 0.09
SIM_C_AXIS = 0.7


def _parse_float_list(val: Any) -> Any:
    """'a:b:step' (inclusive) or a single number -> list of floats."""
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(":")]
        if len(parts) == 3:
            start, stop, step = (float(p) for p in parts)
            n = int(round((stop - start) / step)) + 1
            return [round(start + i * step, 6) for i in range(max(n, 1))]
        return [float(p) for p in val.split(",")]
    if isinstance(val, (int, float)):
        return [float(val)]
    return val


def _parse_int_list(val: Any) -> Any:
    """'a:b:step' (inclusive), 'a,b,c' or a single int -> list of ints."""
    if isinstance(val, str):
        parts = [p.strip() for p in val.split(":")]
        if len(parts) in (2, 3):
            step = int(parts[2]) if len(parts) == 3 else 1
            return list(range(int(parts[0]), int(parts[1]) + 1, step))
        return [int(p) for p in val.split(",")]
    if isinstance(val, (int, float)):
        return [int(val)]
    return val


class XySkewPingpongParameters(TargetSelection, AveragingParameters, StateReadoutParameters, QubitResetParameters):
    """Inputs for xy_skew_pingpong."""

    sensor_qubit: str = Field("q1", description="Qubit that senses the skew (its own line is L1).")
    source_qubit: str = Field("q3", description="Qubit whose xy line is L2; L2 plays at the sensor's frequency.")
    targets: list[str] = Field(default_factory=list, description="[sensor_qubit, source_qubit]; auto-derived.")

    mode: Literal["scan", "calibrate"] = Field(
        "scan",
        description="'calibrate': tomography of the L2 -> sensor crosstalk (gives c_amp, via_phase). "
        "'scan': the ping-pong S(tau_sw) sweep.",
    )

    # ---- scan
    k: float = Field(
        0.1, gt=0.0, le=1.0,
        description="Crosstalk Rabi amplitude at the sensor as a fraction of the sensor's own x180 amplitude.",
    )
    c_amp: float = Field(
        0.09, gt=0.0,
        description="|c|: leak amplitude at the sensor per unit L2 amplitude, relative to L1 (from mode='calibrate').",
    )
    via_phase: float = Field(
        0.0, description="Phase (rad) on L2's waveform that puts its leak along the sensor's +Y (from mode='calibrate')."
    )
    tau_sw_ns: list[float] = Field(
        default_factory=lambda: [round(-3.0 + 0.25 * i, 6) for i in range(25)],
        description="Software shifts of L2 (ns): a list, 'start:stop:step' (inclusive) or 'a,b,c'.",
    )
    repetitions: list[int] = Field(
        default_factory=lambda: [2],
        description="Ping-pong pair counts N (scan) or pulse counts N (calibrate): list, 'a:b:step' or 'a,b'.",
    )
    include_background: bool = Field(
        True, description="Also run kflip = 0 (L1's plain x180 pairs, L2 silent): the sensor's own DRAG background."
    )
    pad_ns: int = Field(
        4, ge=1, le=16,
        description="Zeros (ns) on each side of each scan pulse; |tau_sw| must stay below it.",
    )
    tau_batch_size: int = Field(
        13, gt=0, description="tau_sw points per program (each program compiles and runs on its own)."
    )
    fit_window_ns: float = Field(
        0.5, gt=0.0, description="Half width (ns) around the zero crossing used for the linear fit of S."
    )

    # ---- calibrate
    cal_own_scale: float = Field(
        0.05, gt=0.0, le=1.0,
        description="Scale of the sensor's x180 / y180 in the calibration pulse trains (9 deg per pulse at 0.05); "
        "keep N x rotation below 2 pi for both own and via conditions.",
    )
    cal_via_scale: float = Field(
        0.5, gt=0.0, le=MAX_VIA_AMPLITUDE,
        description="L2 amplitude in the calibration, in units of the sensor's x180 envelope.",
    )
    include_null: bool = Field(
        False,
        description="Calibrate: add the 'null' condition (the scan's L1 -Y part and L2's leak together, no x180) "
        "with the c_amp / via_phase given here, from a previous calibration. Net rotation ~0 confirms them.",
    )
    cal_null_k: float = Field(
        0.03, gt=0.0, le=0.2,
        description="k of the null condition. Small, so a flipped direction (2x the rate) stays below 2 pi at the "
        "largest N: 0.03 is 5.4 deg per pulse for one part alone.",
    )

    def cal_conditions(self) -> list[str]:
        return list(CAL_CONDITIONS) + ([NULL_CONDITION] if self.include_null else [])

    @field_validator("tau_sw_ns", mode="before")
    @classmethod
    def _parse_taus(cls, val: Any) -> Any:
        return _parse_float_list(val)

    @field_validator("repetitions", mode="before")
    @classmethod
    def _parse_reps(cls, val: Any) -> Any:
        return _parse_int_list(val)

    @model_validator(mode="before")
    @classmethod
    def _resolve_targets(cls, data: Any) -> Any:
        if isinstance(data, dict):
            targets = data.get("targets")
            if isinstance(targets, str):
                targets = [t.strip() for t in targets.split(",") if t.strip()]
            if targets and len(targets) >= 2:
                data.setdefault("sensor_qubit", targets[0])
                data.setdefault("source_qubit", targets[1])
            elif targets:
                data.setdefault("sensor_qubit", targets[0])
            data["targets"] = [data.get("sensor_qubit", "q1"), data.get("source_qubit", "q3")]
        return data

    @model_validator(mode="after")
    def _check(self) -> "XySkewPingpongParameters":
        if self.sensor_qubit == self.source_qubit:
            raise ValueError("sensor_qubit and source_qubit must differ")
        if not self.repetitions or min(self.repetitions) < 0:
            raise ValueError("repetitions must be non-negative counts")
        if self.mode == "scan":
            if not self.tau_sw_ns:
                raise ValueError("tau_sw_ns is empty")
            worst = max(abs(t) for t in self.tau_sw_ns)
            if worst > self.pad_ns - 1:
                raise ValueError(f"|tau_sw| up to {worst} ns needs pad_ns >= {math.ceil(worst) + 1}")
        return self

    def kflips(self) -> list[int]:
        return [1, -1, 0] if self.include_background else [1, -1]


class XySkewPingpongContract(DatasetContract):
    """target x (tau_sw_ns, repetitions, kflip) in scan mode, target x (condition, repetitions, basis) in calibrate."""

    def validate(self, ds: xr.Dataset) -> None:
        scan = {"tau_sw_ns", "repetitions", "kflip"}
        cal = {"condition", "repetitions", "basis"}
        dims = set(ds.dims)
        problems = []
        if "target" not in dims:
            problems.append("missing dimension 'target'")
        if not (scan <= dims or cal <= dims):
            problems.append(f"dims {sorted(dims)} carry neither {sorted(scan)} nor {sorted(cal)}")
        if not any(v in ds.data_vars for v in ("population", "I")):
            problems.append("missing data variable ('population' or 'I')")
        if problems:
            raise ContractError("; ".join(problems))


class XySkewPingpongResult(Result):
    """Output of xy_skew_pingpong."""

    fit: dict[str, dict[str, Any]] = Field(default_factory=dict, description="Per-sensor fit (mode-dependent).")


# ----------------------------------------------------------------------------- waveforms
def fractional_shift(x: np.ndarray, tau: float) -> np.ndarray:
    """x delayed by tau samples (band-limited: the shift of what the DAC reconstructs from x).

    Zero-padded FFT phase ramp; an integer tau is an exact sample shift. x should carry enough zeros
    at both ends for the shifted pulse (the band-limited tails beyond the window are cut).
    """
    x = np.asarray(x, dtype=complex)
    n = len(x)
    size = 1 << int(math.ceil(math.log2(8 * n)))
    f = np.fft.fftfreq(size)
    y = np.fft.ifft(np.fft.fft(x, size) * np.exp(-2j * np.pi * f * float(tau)))[:n]
    return y


def axis_factor(x180: np.ndarray, y180: np.ndarray) -> complex:
    """Y = y180 / x180 as one unit complex number (+-i for a standard QUAM pair)."""
    z = complex(np.vdot(x180, y180))
    if abs(z) == 0:
        raise ValueError("x180 and y180 are orthogonal or empty")
    return z / abs(z)


def scan_waveforms(
    x180: np.ndarray, y180: np.ndarray, *, k: float, c_amp: float, via_phase: float, tau_ns: float, pad: int,
    kflip: int, sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    """(L1, L2) complex samples of C(sign) for one tau_sw and kflip (L2 is all zeros at kflip = 0)."""
    n = len(x180)
    width = max(16, int(math.ceil((n + 2 * pad) / 4.0)) * 4)
    zx = np.zeros(width, complex)
    zx[pad:pad + n] = x180
    ox = np.zeros(width, complex)
    ox[pad:pad + n] = np.real(x180)
    yax = axis_factor(x180, y180)
    l1 = sign * (zx - kflip * k * yax * ox)
    l2 = sign * kflip * (k / c_amp) * np.exp(1j * via_phase) * fractional_shift(ox, tau_ns)
    return l1, l2


def null_waveforms(
    x180: np.ndarray, y180: np.ndarray, *, k: float, c_amp: float, via_phase: float
) -> tuple[np.ndarray, np.ndarray]:
    """(L1, L2) of the null check: the scan's two Y parts at tau_sw = 0 without x180 (unpadded).

    L1 = -k Y Ox and L2 = (k / c_amp) e^{i via_phase} Ox; with the right c_amp / via_phase L2's leak at the
    sensor is +k Y Ox and the sum is zero.
    """
    ox = np.real(x180).astype(complex)
    return -k * axis_factor(x180, y180) * ox, (k / c_amp) * np.exp(1j * via_phase) * ox


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


def via_element_name(source_qubit) -> str:
    return f"{source_qubit.xy.name}_via"


def patch_config(
    config: dict, sensor_qubit, source_qubit, *, k: float, c_amp: float, via_phase: float, tau_sw_ns: list[float],
    pad_ns: int, cal_via_scale: float, cal_null_k: float = 0.03,
) -> dict:
    """Add `<source xy>_via` (L2's port, the sensor's frequency, next free core) and every op either mode plays.

    The via element is built by the crosstalk cancel-element builder (victim = source, aggressor = sensor:
    a copy of the source's element with its IF moved to the sensor's frequency), then renamed. All ops
    of both modes are always added, so both modes run on identical elements and cores.
    """
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
    return config


def scan_pulse_cycles(n_samples: int, pad_ns: int) -> int:
    return max(16, int(math.ceil((n_samples + 2 * pad_ns) / 4.0)) * 4) // 4


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
    use_state_discrimination: bool = False, simulate: bool = False,
):
    """y90 - [C(+), C(-)] x N - read, for every (tau_sw, N, kflip), interleaved inside one shot loop.

    The two lines play inside strict_timing_ from one align, each pulse back to back on both elements
    (L2 waits where it is silent). No amplitude variables and no real-time branches: every waveform is
    its own op.
    """
    sensor_xy = sensor_qubit.xy.name
    via = via_element_name(source_qubit)
    elems = [sensor_xy, via]
    try:
        y90_cycles = max(sensor_qubit.xy.operations["y90"].length // 4, 1)
    except Exception:
        y90_cycles = 4
    n_points = len(tau_indices) * len(repetitions) * len(kflips)

    sweep_axes = {
        "qubit": xr.DataArray([sensor_qubit.name]),
        "tau_sw_ns": xr.DataArray(np.asarray(tau_values, dtype=float)),
        "repetitions": xr.DataArray(np.asarray(repetitions, dtype=int)),
        "kflip": xr.DataArray(np.asarray(kflips, dtype=int)),
    }

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
                        with strict_timing_():
                            sensor_qubit.xy.play("y90")
                            wait(y90_cycles, via)
                            for _ in range(int(reps)):
                                for sign in (1, -1):
                                    play(scan_op(i_tau, kflip, sign), sensor_xy)
                                    if kflip != 0:
                                        play(scan_op(i_tau, kflip, sign), via)
                                    else:
                                        wait(pulse_cycles, via)
                        align()
                        _readout(sensor_qubit, use_state_discrimination, state, state_st, I[0], I_st[0], Q[0], Q_st[0])
                        align()
            save(n, n_st)

        with stream_processing():
            n_st.save("n")
            shape = (len(tau_indices), len(repetitions), len(kflips))
            if use_state_discrimination:
                state_st.buffer(*shape).average().save("state1")
            else:
                I_st[0].buffer(*shape).average().save("I1")
                Q_st[0].buffer(*shape).average().save("Q1")
    assert n_points == int(np.prod(shape))
    return prog, sweep_axes


def build_calibrate_program(
    machine, sensor_qubit, source_qubit, *, repetitions: list[int], cal_own_scale: float, num_shots: int,
    conditions: tuple[str, ...] = CAL_CONDITIONS, bases: tuple[str, ...] = BASES, reset_type: str = "thermal",
    reset_max_attempts: int = 1, use_state_discrimination: bool = False, simulate: bool = False,
):
    """Sensor tomography after N pulses of one condition (own_x, own_y, via_0, via_90, null), from |0>.

    Bases are a QUA loop with the pre-rotations amplitude-switched 1/0 (a play in a real-time branch
    hung the job, 2026-10-05); conditions and N are unrolled with fixed plays.
    """
    unknown = [c for c in conditions if c not in (*CAL_CONDITIONS, NULL_CONDITION)]
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
                        with strict_timing_():
                            if int(reps) == 0:
                                wait(4, sensor_xy)
                            for _ in range(int(reps)):
                                if cond == "own_x":
                                    play("x180" * amp(float(cal_own_scale)), sensor_xy)
                                    wait(cycles, via)
                                elif cond == "own_y":
                                    play("y180" * amp(float(cal_own_scale)), sensor_xy)
                                    wait(cycles, via)
                                elif cond == NULL_CONDITION:
                                    play("skewnull", sensor_xy)
                                    play("skewnull", via)
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


# ----------------------------------------------------------------------------- analysis
def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def fit_rotation(reps: np.ndarray, p1: dict[str, np.ndarray]) -> dict[str, float]:
    """Rotation per pulse (rad) and its axis angle (rad) from |0> tomography after N pulses.

    Bloch r_b = 1 - 2 P1_b. A rotation by N*theta about (cos phi, sin phi, 0) takes z to
    (sin(N theta) sin phi, -sin(N theta) cos phi, cos(N theta)), so i(rx + i ry) = sin(N theta) e^{i phi}
    and the polar angle is N theta. The measured polar angle folds back at pi; it is unfolded once
    (a decrease after passing pi/2), so keep N theta below 2 pi. Axis angles share the tomography's
    convention, so only differences between conditions are meaningful.
    """
    reps = np.asarray(reps, dtype=float)
    order = np.argsort(reps)
    reps = reps[order]
    rx, ry, rz = ((1.0 - 2.0 * np.asarray(p1[b], dtype=float))[order] for b in ("x", "y", "z"))
    use = reps > 0
    if not use.any():
        return {"theta": float("nan"), "axis": float("nan"), "contrast": float("nan")}
    folded_polar = np.arctan2(np.hypot(rx, ry), rz)
    polar = folded_polar.copy()
    folded, prev = False, 0.0
    for i, a in enumerate(folded_polar):
        if not folded and prev > np.pi / 2 and a < prev:
            folded = True
        polar[i] = 2 * np.pi - a if folded else a
        prev = a
    theta = float(np.sum(reps[use] * polar[use]) / np.sum(reps[use] ** 2))
    # past pi the transverse part points the other way: sin(N theta) < 0
    phasor = np.sum((np.sign(np.sin(polar)) * 1j * (rx + 1j * ry))[use])
    contrast = float(np.mean(np.sqrt(rx**2 + ry**2 + rz**2)))
    return {"theta": theta, "axis": float(np.angle(phasor)), "contrast": contrast}


def analyze_null(fit_null: dict[str, float], *, own_rate: float, own_y_axis: float, handedness: float,
                 cal_null_k: float, c_amp: float, via_phase: float) -> dict[str, Any]:
    """The null check: residual of the two Y parts, and the c_amp / via_phase that would zero it.

    In the tomography frame L1's part is -cal_null_k * own_rate along own_y and L2's leak is
    cal_null_k * own_rate * G along own_y (G = 1 when c_amp / via_phase are right), so the net drive
    theta_null e^{i axis_null} = cal_null_k * own_rate * (G - 1) e^{i own_y}. null_ratio = |G - 1|:
    ~0 when right, ~2 when the direction is flipped. Correcting: c_amp x |G|, and via_phase minus arg(G)
    turned back into a waveform phase by the via handedness. (A mirrored tomography frame flips both
    arg(G) and the handedness, so the correction holds either way.)
    """
    ref = cal_null_k * own_rate
    if not (np.isfinite(ref) and ref > 0 and np.isfinite(fit_null["theta"])):
        return {"null_ratio": float("nan")}
    ratio = float(fit_null["theta"] / ref)
    g = 1.0 + ratio * np.exp(1j * (fit_null["axis"] - own_y_axis))
    return {
        "null_ratio": ratio,
        "null_rad_per_pulse": float(fit_null["theta"]),
        "null_gain": abs(g),
        "null_angle_deg": float(np.rad2deg(np.angle(g))),
        "c_amp_refined": float(c_amp * abs(g)),
        "via_phase_refined": _wrap(via_phase - handedness * float(np.angle(g))),
        "null_checked_c_amp": float(c_amp),
        "null_checked_via_phase": float(via_phase),
    }


def analyze_calibration(ds: xr.Dataset, *, cal_own_scale: float, cal_via_scale: float,
                        cal_null_k: float = 0.03, c_amp_played: float | None = None,
                        via_phase_played: float | None = None) -> dict[str, Any]:
    """c_amp and via_phase from the four calibration conditions (population dataset of one sensor).

    With a 'null' condition in the data, c_amp_played / via_phase_played are the values that run played
    (the run's parameters); the null check is added to the result.
    """
    reps = ds["repetitions"].values
    fits = {}
    for cond in CAL_CONDITIONS:
        sub = ds.sel(condition=cond)
        fits[cond] = fit_rotation(reps, {b: sub.sel(basis=b).values for b in BASES})
    xy_angle = _wrap(fits["own_y"]["axis"] - fits["own_x"]["axis"])
    via_turn = _wrap(fits["via_90"]["axis"] - fits["via_0"]["axis"])
    handedness = 1.0 if via_turn >= 0 else -1.0
    via_phase = _wrap(handedness * _wrap(fits["own_y"]["axis"] - fits["via_0"]["axis"]))
    own_rate = fits["own_x"]["theta"] / cal_own_scale
    via_rate = 0.5 * (fits["via_0"]["theta"] + fits["via_90"]["theta"]) / cal_via_scale
    c_amp = float(via_rate / own_rate) if own_rate else float("nan")
    ok = (
        all(np.isfinite([c_amp, via_phase]))
        and c_amp > 0
        and abs(abs(xy_angle) - np.pi / 2) < np.deg2rad(20)
        and abs(abs(via_turn) - np.pi / 2) < np.deg2rad(20)
    )
    out = {
        "success": bool(ok),
        "c_amp": c_amp,
        "via_phase": via_phase,
        "via_phase_deg": float(np.rad2deg(via_phase)),
        "handedness": handedness,
        "own_xy_angle_deg": float(np.rad2deg(xy_angle)),
        "via_turn_deg": float(np.rad2deg(via_turn)),
        "own_x_rad_per_pulse": fits["own_x"]["theta"],
        "via_rad_per_pulse": 0.5 * (fits["via_0"]["theta"] + fits["via_90"]["theta"]),
        "fits": fits,
    }
    if NULL_CONDITION in ds["condition"].values:
        if c_amp_played is None or via_phase_played is None:
            raise ValueError("a null condition needs the c_amp / via_phase it was played with")
        sub = ds.sel(condition=NULL_CONDITION)
        fits[NULL_CONDITION] = fit_rotation(reps, {b: sub.sel(basis=b).values for b in BASES})
        out.update(analyze_null(
            fits[NULL_CONDITION], own_rate=own_rate, own_y_axis=fits["own_y"]["axis"], handedness=handedness,
            cal_null_k=cal_null_k, c_amp=c_amp_played, via_phase=via_phase_played,
        ))
    return out


def zero_crossing(tau: np.ndarray, s: np.ndarray, *, window: float, center: float | None = None) -> dict[str, float]:
    """Zero crossing of S(tau) nearest `center` (default: the scan's middle), refined by a line fit.

    Uses the points within +-window of the bracketing crossing (at least the 4 nearest). Returns
    tau0, its standard error from the fit residuals, and the slope.
    """
    order = np.argsort(tau)
    tau, s = np.asarray(tau, float)[order], np.asarray(s, float)[order]
    center = float(np.mean([tau[0], tau[-1]])) if center is None else float(center)
    crossings = []
    for i in range(len(tau) - 1):
        if s[i] == 0:
            crossings.append(tau[i])
        elif s[i] * s[i + 1] < 0:
            crossings.append(tau[i] - s[i] * (tau[i + 1] - tau[i]) / (s[i + 1] - s[i]))
    if len(tau) and s[-1] == 0:
        crossings.append(tau[-1])
    if not crossings:
        return {"tau0": float("nan"), "tau0_err": float("nan"), "slope": float("nan"), "n_points": 0}
    tc = min(crossings, key=lambda t: abs(t - center))
    near = np.abs(tau - tc) <= window
    if near.sum() < 4:
        near = np.zeros_like(near)
        near[np.argsort(np.abs(tau - tc))[: min(4, len(tau))]] = True
    x, y = tau[near], s[near]
    if len(x) < 2 or np.ptp(x) == 0:
        return {"tau0": float(tc), "tau0_err": float("nan"), "slope": float("nan"), "n_points": int(len(x))}
    slope, icpt = np.polyfit(x, y, 1)
    tau0 = -icpt / slope if slope else float("nan")
    resid = y - (slope * x + icpt)
    dof = max(len(x) - 2, 1)
    sig = float(np.sqrt(np.sum(resid**2) / dof))
    sxx = float(np.sum((x - x.mean()) ** 2))
    err = sig / abs(slope) * math.sqrt(1.0 / len(x) + (x.mean() - tau0) ** 2 / sxx) if slope and sxx else float("nan")
    return {"tau0": float(tau0), "tau0_err": float(err), "slope": float(slope), "n_points": int(len(x))}


def analyze_scan(ds: xr.Dataset, *, window: float) -> dict[str, Any]:
    """Per N: S = P1(k+) - P1(k-), its zero crossing tau_sw*, slope, and the kflip = 0 background."""
    tau = ds["tau_sw_ns"].values.astype(float)
    per_n = {}
    for reps in ds["repetitions"].values:
        sub = ds.sel(repetitions=reps)
        s = sub.sel(kflip=1).values - sub.sel(kflip=-1).values
        zc = zero_crossing(tau, s, window=window)
        entry = {**zc, "S": [float(v) for v in s]}
        if 0 in ds["kflip"].values:
            entry["background_p1"] = float(np.mean(sub.sel(kflip=0).values))
        per_n[int(reps)] = entry
    good = {n: e for n, e in per_n.items() if np.isfinite(e["tau0"]) and np.isfinite(e["tau0_err"])}
    best = min(good.items(), key=lambda kv: kv[1]["tau0_err"]) if good else None
    return {
        "success": best is not None,
        "tau_sw_star_ns": best[1]["tau0"] if best else float("nan"),
        "tau_sw_star_err_ns": best[1]["tau0_err"] if best else float("nan"),
        "best_repetitions": best[0] if best else None,
        "per_repetitions": per_n,
        "tau_sw_ns": [float(t) for t in tau],
    }


# ----------------------------------------------------------------------------- offline model
def _sim_pulses(width_pad: int = 4, length: int = 16, alpha_ghz: float = -0.205):
    """DragCosine x180 / y180 at 1 GS/s for the offline model (first-order DRAG)."""
    t = np.arange(length)
    env = 0.5 * (1 - np.cos(2 * np.pi * t / length))
    denv = np.pi / length * np.sin(2 * np.pi * t / length)
    a0 = np.pi / env.sum()
    beta = -1 / (2 * np.pi * alpha_ghz)
    x180 = a0 * (env + 1j * beta * denv)
    return x180, 1j * x180


def _sim_evolve(segments: list[np.ndarray], alpha_ghz: float = -0.205, psi0=None, sub: int = 4) -> np.ndarray:
    """3-level transmon in the sensor frame driven by 1 GS/s envelopes (zero-order hold, `sub` steps/ns)."""
    from scipy.linalg import expm

    alpha = 2 * np.pi * alpha_ghz
    a = np.diag([1.0, np.sqrt(2.0)], 1)
    h0 = np.diag([0.0, 0.0, alpha])
    dt = 1.0 / sub
    psi = np.array([1, 0, 0], complex) if psi0 is None else psi0
    cache: dict = {}
    for seg in segments:
        key = seg.tobytes()
        if key not in cache:
            u = np.eye(3, dtype=complex)
            for o in seg:
                step = expm(-1j * (h0 + 0.5 * (o * a.T + np.conj(o) * a)) * dt)
                u = np.linalg.matrix_power(step, sub) @ u
            cache[key] = u
        psi = cache[key] @ psi
    return psi


# ----------------------------------------------------------------------------- experiment
@register
class XySkewPingpong(Experiment):
    """Sub-ns XY line skew at a sensor qubit from a DRAG-like ping-pong (QM/QUAM)."""

    name: ClassVar[str] = "xy_skew_pingpong"
    description: ClassVar[str] = (
        "Arrival-time skew between two xy lines at a sensor qubit: the source line plays at the sensor's "
        "frequency through crosstalk against an equal opposite Y part on the sensor's line; a ping-pong "
        "accumulates the skew's DRAG-like residual. mode='calibrate' measures the crosstalk first."
    )
    maturity: ClassVar[str] = "contrib"
    Parameters: ClassVar[type] = XySkewPingpongParameters
    Result: ClassVar[type] = XySkewPingpongResult
    Contract: ClassVar[DatasetContract] = XySkewPingpongContract(
        sweeps=("tau_sw_ns", "repetitions", "kflip"),
        sweep_units=("ns", "", ""),
        variables=("I", "Q"),
        alt_variables=(*POPULATION_ALT, ("I",)),
    )
    # thermal reset only: active reset is hardware-validated for the SQRB, not for this probe

    params: XySkewPingpongParameters

    # ---- physics half
    def define_sweep(self) -> dict[str, np.ndarray]:
        p = self.params
        if p.mode == "calibrate":
            return {
                "condition": np.asarray(p.cal_conditions(), dtype=str),
                "repetitions": np.asarray(p.repetitions, dtype=int),
                "basis": np.asarray(BASES, dtype=str),
            }
        return {
            "tau_sw_ns": np.asarray(p.tau_sw_ns, dtype=float),
            "repetitions": np.asarray(p.repetitions, dtype=int),
            "kflip": np.asarray(p.kflips(), dtype=int),
        }

    def simulate(self, coords: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        p = self.params
        rng = np.random.default_rng(stable_seed("xy_skew_pingpong", p.sensor_qubit, p.mode))
        x180, y180 = _sim_pulses()
        ox = np.real(x180).astype(complex)
        if p.mode == "calibrate":
            c_true = SIM_C_AMP * np.exp(1j * SIM_C_AXIS)
            seg = {
                "own_x": p.cal_own_scale * x180,
                "own_y": p.cal_own_scale * y180,
                "via_0": c_true * p.cal_via_scale * ox,
                "via_90": c_true * p.cal_via_scale * 1j * ox,
            }
            n1, n2 = null_waveforms(x180, y180, k=p.cal_null_k, c_amp=p.c_amp, via_phase=p.via_phase)
            seg[NULL_CONDITION] = n1 + c_true * n2
            pre = {"z": np.zeros(16, complex), "x": -0.5 * y180, "y": 0.5 * x180}
            out = np.zeros((len(coords["condition"]), len(coords["repetitions"]), len(coords["basis"])))
            for ic, cond in enumerate(coords["condition"]):
                for ir, reps in enumerate(coords["repetitions"]):
                    for ib, b in enumerate(coords["basis"]):
                        psi = _sim_evolve([seg[str(cond)]] * int(reps) + [pre[str(b)]])
                        out[ic, ir, ib] = abs(psi[1]) ** 2
        else:
            # truth: L2's leak arrives SIM_SKEW_NS late, and c is known to 5% / 3 deg
            c_err = 1.05 * np.exp(1j * np.deg2rad(3.0))
            pad = p.pad_ns
            y90 = 0.5 * y180
            out = np.zeros((len(coords["tau_sw_ns"]), len(coords["repetitions"]), len(coords["kflip"])))
            for it, tau in enumerate(coords["tau_sw_ns"]):
                for ik, kflip in enumerate(coords["kflip"]):
                    segs = {}
                    for sign in (1, -1):
                        l1, _ = scan_waveforms(x180, y180, k=p.k, c_amp=1.0, via_phase=0.0, tau_ns=0.0, pad=pad,
                                               kflip=int(kflip), sign=sign)
                        _, l2 = scan_waveforms(x180, y180, k=p.k, c_amp=1.0, via_phase=0.0,
                                               tau_ns=float(tau) + SIM_SKEW_NS, pad=pad, kflip=int(kflip), sign=sign)
                        segs[sign] = l1 + c_err * 1j * l2
                    for ir, reps in enumerate(coords["repetitions"]):
                        psi = _sim_evolve([y90] + [segs[1], segs[-1]] * int(reps))
                        out[it, ir, ik] = abs(psi[1]) ** 2
        noisy = np.clip(out + rng.normal(0, 0.005, out.shape), 0.0, 1.0)
        dims = ("target", *coords.keys())
        n_targets = max(1, len(p.targets))
        data = np.repeat(noisy[None, ...], n_targets, axis=0)
        if p.use_state_discrimination:
            return {"population": (dims, data)}
        return {"I": (dims, data), "Q": (dims, np.zeros_like(data))}

    def estimate(self) -> XySkewPingpongResult:
        assert self.dataset is not None, "run() populates self.dataset before estimate()"
        p = self.params
        result = XySkewPingpongResult()
        ds = self.dataset
        sensor = p.sensor_qubit
        if "target" in ds.dims and sensor in ds["target"].values:
            ds = ds.sel(target=sensor)
        elif "target" in ds.dims:
            ds = ds.isel(target=0)
        if "population" not in ds.data_vars:
            result.fit[sensor] = {"success": False, "reason": "needs use_state_discrimination (population)"}
            result.outcomes[sensor] = Outcome.FAILED
            return result
        pop = ds["population"]
        if p.mode == "calibrate":
            fit = analyze_calibration(pop, cal_own_scale=p.cal_own_scale, cal_via_scale=p.cal_via_scale,
                                      cal_null_k=p.cal_null_k, c_amp_played=p.c_amp, via_phase_played=p.via_phase)
        else:
            fit = analyze_scan(pop, window=p.fit_window_ns)
        fit["mode"] = p.mode
        result.fit[sensor] = fit
        result.outcomes[sensor] = Outcome.SUCCESSFUL if fit.get("success") else Outcome.FAILED
        return result

    # ---- QM half
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

    def _programs(self, config: dict, sensor_qubit, source_qubit, num_shots: int):
        """[(program, sweep_axes)] for this run: one calibration program, or one scan program per tau batch."""
        p = self.params
        kw = self._program_kwargs(sensor_qubit, source_qubit)
        if p.mode == "calibrate":
            return [build_calibrate_program(
                self.backend.machine, sensor_qubit, source_qubit, repetitions=list(p.repetitions),
                cal_own_scale=p.cal_own_scale, num_shots=num_shots, conditions=tuple(p.cal_conditions()), **kw,
            )]
        n_samples = len(_op_iq(config, sensor_qubit.xy.name, "x180"))
        cycles = scan_pulse_cycles(n_samples, p.pad_ns)
        idx = list(range(len(p.tau_sw_ns)))
        batches = [idx[i:i + p.tau_batch_size] for i in range(0, len(idx), p.tau_batch_size)]
        return [
            build_scan_program(
                self.backend.machine, sensor_qubit, source_qubit, tau_indices=b,
                tau_values=[p.tau_sw_ns[i] for i in b], repetitions=list(p.repetitions), kflips=p.kflips(),
                pulse_cycles=cycles, num_shots=num_shots, **kw,
            )
            for b in batches
        ]

    def preview_program(self):
        """The first program of the run, with at most 10 shots."""
        machine = self.backend.machine
        sensor_qubit, source_qubit = self._select_pair()
        config = self._patch(machine.generate_config(), sensor_qubit, source_qubit)
        prog, _ = self._programs(config, sensor_qubit, source_qubit, min(int(self.params.num_averages), 10))[0]
        return prog

    def probe(self, timeout: int = 180, log: Optional[Callable] = None, config: Optional[dict] = None) -> xr.Dataset:
        machine = self.backend.machine
        sensor_qubit, source_qubit = self._select_pair()
        if config is None:
            config = machine.generate_config()
        config = self._patch(config, sensor_qubit, source_qubit)
        num_shots = int(self.params.num_averages)
        datasets = []
        with shared_qm(machine, config, timeout=timeout) as qm:
            for prog, axes in self._programs(config, sensor_qubit, source_qubit, num_shots):
                datasets.append(_acquire(machine, prog, axes, num_shots=num_shots, timeout=timeout, log=log,
                                         config=config, qm=qm))
        return datasets[0] if len(datasets) == 1 else xr.concat(datasets, dim="tau_sw_ns")
