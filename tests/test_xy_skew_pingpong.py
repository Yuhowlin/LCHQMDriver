"""Tests for xy_skew_pingpong: waveforms, config patch, QUA structure and the offline model + analysis."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from qm.qua import amp, declare, declare_stream, fixed, play

from scqo_qm.experiments.xy_skew_pingpong import (
    BASES,
    CAL_CONDITIONS,
    XySkewPingpong,
    XySkewPingpongParameters,
    _sim_pulses,
    analyze_calibration,
    analyze_scan,
    axis_factor,
    build_calibrate_program,
    build_scan_program,
    fractional_shift,
    patch_config,
    scan_op,
    scan_pulse_cycles,
    scan_waveforms,
    zero_crossing,
)


def _mock_machine():
    def _decl():
        return [declare(fixed)], [declare_stream()], [declare(fixed)], [declare_stream()], declare(int), declare_stream()

    def qubit(name, rf):
        xy = SimpleNamespace(
            name=f"{name}.xy",
            RF_frequency=rf,
            operations={op: SimpleNamespace(length=16) for op in ("x180", "y180", "x90", "y90", "-y90")},
            play=lambda op, amplitude_scale=None, _e=f"{name}.xy", **kw: play(
                op * amp(amplitude_scale) if amplitude_scale is not None else op, _e
            ),
        )
        res = SimpleNamespace(name=f"{name}.res", measure=lambda *a, **kw: None)
        return SimpleNamespace(name=name, xy=xy, resonator=res, reset=lambda *a, **kw: None, readout_state=lambda *a: None)

    q1, q3 = qubit("q1", 5.1446e9), qubit("q3", 5.1920e9)
    mach = SimpleNamespace(declare_qua_variables=_decl, initialize_qpu=lambda **kw: None, qubits={"q1": q1, "q3": q3})
    return mach, q1, q3


def _config(x180, y180):
    def elem(port, if_hz, core):
        return {"MWInput": {"port": ("con1", 6, port), "upconverter": 1}, "intermediate_frequency": if_hz,
                "core": core, "operations": {}}

    cfg = {"elements": {"q1.xy": elem(2, 244.6e6, "a"), "q3.xy": elem(4, 192.0e6, "c")}, "pulses": {}, "waveforms": {}}
    for el in ("q1.xy", "q3.xy"):
        for op, z in (("x180", x180), ("y180", y180)):
            n = f"{el}.{op}"
            cfg["waveforms"][f"{n}.I"] = {"type": "arbitrary", "samples": list(np.real(z))}
            cfg["waveforms"][f"{n}.Q"] = {"type": "arbitrary", "samples": list(np.imag(z))}
            cfg["pulses"][f"{n}.p"] = {"operation": "control", "length": len(z), "waveforms": {"I": f"{n}.I", "Q": f"{n}.Q"}}
            cfg["elements"][el]["operations"][op] = f"{n}.p"
    return cfg


def _qm_pulses():
    x180, y180 = _sim_pulses()
    scale = 0.2 / np.max(np.abs(x180))  # a realistic DAC amplitude
    return scale * x180, scale * y180


def test_fractional_shift_is_exact_for_integers_and_smooth_for_fractions():
    t = np.arange(32)
    x = np.where((t >= 8) & (t < 24), 0.5 * (1 - np.cos(2 * np.pi * (t - 8) / 16)), 0.0)
    assert np.allclose(fractional_shift(x, 2), np.roll(x, 2), atol=1e-12)
    half = fractional_shift(x, 0.5).real
    analytic = np.where((t >= 8.5) & (t < 24.5), 0.5 * (1 - np.cos(2 * np.pi * (t - 8.5) / 16)), 0.0)
    assert np.max(np.abs(half - analytic)) < 0.01
    # the shift moves the centroid by exactly tau
    for tau in (-1.3, 0.25, 2.7):
        y = fractional_shift(x, tau).real
        assert np.sum(t * y) / np.sum(y) == pytest.approx(np.sum(t * x) / np.sum(x) + tau, abs=1e-4)


def test_y_parts_cancel_at_the_alignment_point():
    x180, y180 = _qm_pulses()
    assert axis_factor(x180, y180) == pytest.approx(1j)
    c_amp, via_phase, k = 0.09, 0.4, 0.1
    leak = c_amp * axis_factor(x180, y180) * np.exp(-1j * via_phase)  # c that the calibration assumes
    for kflip in (1, -1):
        for sign in (1, -1):
            l1, l2 = scan_waveforms(x180, y180, k=k, c_amp=c_amp, via_phase=via_phase, tau_ns=0.0, pad=4,
                                    kflip=kflip, sign=sign)
            assert len(l1) == len(l2) == 24
            assert np.allclose(l1 + leak * l2, sign * np.pad(x180, 4), atol=1e-9)
    # a shift leaves a residual with zero area (derivative-like)
    l1, l2 = scan_waveforms(x180, y180, k=k, c_amp=c_amp, via_phase=via_phase, tau_ns=0.3, pad=4, kflip=1, sign=1)
    resid = l1 + leak * l2 - np.pad(x180, 4)
    assert np.max(np.abs(resid)) > 1e-3
    assert abs(resid.sum()) < 1e-4 * np.abs(x180).sum()  # band-limited tails cut at the window
    _, silent = scan_waveforms(x180, y180, k=k, c_amp=c_amp, via_phase=via_phase, tau_ns=0.3, pad=4, kflip=0, sign=1)
    assert not silent.any()


def test_patch_config_builds_the_via_element_on_the_source_line():
    mach, q1, q3 = _mock_machine()
    x180, y180 = _qm_pulses()
    taus = [-1.0, 0.0, 0.5]
    cfg = patch_config(_config(x180, y180), q1, q3, k=0.1, c_amp=0.09, via_phase=0.0, tau_sw_ns=taus, pad_ns=4,
                       cal_via_scale=0.5)
    via = cfg["elements"]["q3.xy_via"]
    assert via["MWInput"]["port"] == ("con1", 6, 4)
    assert via["intermediate_frequency"] == pytest.approx(192.0e6 + (5.1446e9 - 5.1920e9))
    assert via["core"] not in ("a", "c")
    assert "q3.xy_cancel" not in cfg["elements"]
    # L1 carries kflip +-1 and 0, L2 only +-1, both signs, every tau; plus the calibration ops
    assert sum(op.startswith("skew_t") for op in cfg["elements"]["q1.xy"]["operations"]) == len(taus) * 3 * 2
    assert sum(op.startswith("skew_t") for op in via["operations"]) == len(taus) * 2 * 2
    assert {"skewcal_0", "skewcal_90"} <= set(via["operations"])
    with pytest.raises(ValueError, match="lower k"):
        patch_config(_config(x180, y180), q1, q3, k=0.5, c_amp=0.09, via_phase=0.0, tau_sw_ns=taus, pad_ns=4,
                     cal_via_scale=0.5)


def test_scan_program_plays_both_lines_in_lockstep_without_branches():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    prog, axes = build_scan_program(
        mach, q1, q3, tau_indices=[0, 1], tau_values=[-0.5, 0.5], repetitions=[1, 3], kflips=[1, -1, 0],
        pulse_cycles=scan_pulse_cycles(16, 4), num_shots=5, use_state_discrimination=True,
    )
    sc = generate_qua_script(prog).replace("'", '"')
    assert "reset_global_phase()" in sc and "strict_timing_()" in sc
    assert "if_(" not in sc and "switch_(" not in sc and "amp(" not in sc
    # per tau: (1 + 3) pairs x 2 signs, for each kflip on L1; L2 plays only kflip +-1
    assert len(re.findall(r'play\("skew_t0_k[pm0]_s[pm]", "q1\.xy"\)', sc)) == 4 * 2 * 3
    assert len(re.findall(r'play\("skew_t0_k[pm]_s[pm]", "q3\.xy_via"\)', sc)) == 4 * 2 * 2
    assert len(re.findall(r'play\("y90", "q1\.xy"\)', sc)) == 2 * 2 * 3
    assert list(axes["kflip"].values) == [1, -1, 0]
    assert list(axes["tau_sw_ns"].values) == [-0.5, 0.5]
    assert scan_op(1, -1, 1) == "skew_t1_km_sp"


def test_calibrate_program_structure():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    prog, axes = build_calibrate_program(mach, q1, q3, repetitions=[0, 2], cal_own_scale=0.1, num_shots=5,
                                         use_state_discrimination=True)
    sc = generate_qua_script(prog).replace("'", '"')
    assert len(re.findall(r'play\("skewcal_0", "q3\.xy_via"\)', sc)) == 2
    assert len(re.findall(r'play\("skewcal_90", "q3\.xy_via"\)', sc)) == 2
    assert "if_(" not in sc and "switch_(" not in sc
    assert list(axes["condition"].values) == list(CAL_CONDITIONS)
    assert list(axes["basis"].values) == list(BASES)


def test_parameters_parse_and_guard():
    p = XySkewPingpongParameters(targets=["q1", "q3"], tau_sw_ns="-1:1:0.5", repetitions="2,5")
    assert p.sensor_qubit == "q1" and p.source_qubit == "q3"
    assert p.tau_sw_ns == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert p.repetitions == [2, 5]
    assert p.kflips() == [1, -1, 0]
    with pytest.raises(ValueError, match="pad_ns"):
        XySkewPingpongParameters(targets=["q1", "q3"], tau_sw_ns=[-3.5, 3.5], pad_ns=4)


def _simulated(**kw):
    params = XySkewPingpongParameters(targets=["q1", "q3"], use_state_discrimination=True, **kw)
    exp = XySkewPingpong(SimpleNamespace(device=None), params)
    coords = exp.define_sweep()
    data = exp.simulate(coords)
    ds = xr.Dataset({k: v for k, v in data.items()}, coords={"target": params.targets, **coords})
    return params, ds["population"].sel(target="q1")


def test_offline_scan_recovers_the_simulated_skew():
    from scqo_qm.experiments.xy_skew_pingpong import SIM_SKEW_NS

    params, pop = _simulated(mode="scan", k=0.1, repetitions=[2], tau_sw_ns="-1.5:1.5:0.25")
    fit = analyze_scan(pop, window=params.fit_window_ns)
    assert fit["success"]
    assert fit["tau_sw_star_ns"] == pytest.approx(-SIM_SKEW_NS, abs=0.05)
    assert 0.0 < fit["per_repetitions"][2]["background_p1"] < 1.0


def test_offline_calibration_recovers_the_simulated_crosstalk():
    from scqo_qm.experiments.xy_skew_pingpong import SIM_C_AMP, SIM_C_AXIS

    params, pop = _simulated(mode="calibrate", repetitions="0:12:2")
    fit = analyze_calibration(pop, cal_own_scale=params.cal_own_scale, cal_via_scale=params.cal_via_scale)
    assert fit["success"], fit
    assert fit["c_amp"] == pytest.approx(SIM_C_AMP, rel=0.05)
    # the leak points along +Y (y180 = i x180 in the model) once L2 carries via_phase
    expected = (np.pi / 2 - SIM_C_AXIS + np.pi) % (2 * np.pi) - np.pi
    assert fit["via_phase"] == pytest.approx(expected, abs=np.deg2rad(3))


def test_zero_crossing_picks_the_one_nearest_the_center():
    tau = np.linspace(-3, 3, 25)
    s = np.sin(2.0 * (tau - 0.2))  # crossings at 0.2 and 0.2 +- pi/2
    zc = zero_crossing(tau, s, window=0.5)
    assert zc["tau0"] == pytest.approx(0.2, abs=0.02)
    assert zc["slope"] > 0


def _null_fit(c_amp, via_phase):
    params, pop = _simulated(mode="calibrate", repetitions="0:20:2", include_null=True, c_amp=c_amp,
                             via_phase=via_phase)
    return analyze_calibration(pop, cal_own_scale=params.cal_own_scale, cal_via_scale=params.cal_via_scale,
                               cal_null_k=params.cal_null_k, c_amp_played=c_amp, via_phase_played=via_phase)


def _true_via_phase():
    from scqo_qm.experiments.xy_skew_pingpong import SIM_C_AXIS

    return float((np.pi / 2 - SIM_C_AXIS + np.pi) % (2 * np.pi) - np.pi)


def test_null_check_vanishes_at_the_right_calibration():
    from scqo_qm.experiments.xy_skew_pingpong import SIM_C_AMP

    fit = _null_fit(SIM_C_AMP, _true_via_phase())
    assert fit["null_ratio"] < 0.1
    assert fit["c_amp_refined"] == pytest.approx(SIM_C_AMP, rel=0.1)


def test_null_check_flags_a_flipped_direction():
    from scqo_qm.experiments.xy_skew_pingpong import SIM_C_AMP

    fit = _null_fit(SIM_C_AMP, _true_via_phase() + np.pi)
    assert fit["null_ratio"] == pytest.approx(2.0, abs=0.2)


def test_null_check_refines_a_wrong_calibration():
    from scqo_qm.experiments.xy_skew_pingpong import SIM_C_AMP

    true_phase = _true_via_phase()
    fit = _null_fit(1.2 * SIM_C_AMP, true_phase + np.deg2rad(30))
    assert 0.3 < fit["null_ratio"] < 0.8
    assert fit["c_amp_refined"] == pytest.approx(SIM_C_AMP, rel=0.05)
    err = (fit["via_phase_refined"] - true_phase + np.pi) % (2 * np.pi) - np.pi
    assert abs(err) < np.deg2rad(4)


def test_null_condition_plays_both_lines_together():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    prog, axes = build_calibrate_program(mach, q1, q3, repetitions=[3], cal_own_scale=0.05, num_shots=2,
                                         conditions=(*CAL_CONDITIONS, "null"), use_state_discrimination=True)
    sc = generate_qua_script(prog).replace("'", '"')
    assert len(re.findall(r'play\("skewnull", "q1\.xy"\)', sc)) == 3
    assert len(re.findall(r'play\("skewnull", "q3\.xy_via"\)', sc)) == 3
    assert list(axes["condition"].values)[-1] == "null"
    p = XySkewPingpongParameters(targets=["q1", "q3"], mode="calibrate", include_null=True)
    assert p.cal_conditions()[-1] == "null"


def test_null_waveforms_cancel_with_the_assumed_crosstalk():
    from scqo_qm.experiments.xy_skew_pingpong import null_waveforms

    x180, y180 = _qm_pulses()
    l1, l2 = null_waveforms(x180, y180, k=0.03, c_amp=0.09, via_phase=0.4)
    leak = 0.09 * axis_factor(x180, y180) * np.exp(-1j * 0.4)
    assert np.allclose(l1 + leak * l2, 0, atol=1e-12)
    assert np.max(np.abs(l1)) > 0
