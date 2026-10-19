"""xy_skew_pingpong QM probe: config patch and QUA program structure (physics/analysis are tested in SCQO/scqat)."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
from qm.qua import amp, declare, declare_stream, fixed, play

from scqo.experiments.xy_skew_pingpong import CAL_CONDITIONS, BASES, XySkewPingpongParameters, _sim_pulses
from scqo_qm.experiments.xy_skew_pingpong import (
    QMXySkewPingpong,
    build_calibrate_program,
    build_scan_program,
    flip_op,
    l1_op,
    patch_config,
    scan_op,
    scan_pulse_cycles,
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


def _config():
    x180, y180 = _sim_pulses()
    scale = 0.2 / np.max(np.abs(x180))
    x180, y180 = scale * x180, scale * y180

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


def _iq(cfg, el, op):
    p = cfg["pulses"][cfg["elements"][el]["operations"][op]]
    return np.array(cfg["waveforms"][p["waveforms"]["I"]]["samples"]) + 1j * np.array(
        cfg["waveforms"][p["waveforms"]["Q"]]["samples"])


def test_patch_config_builds_the_via_element_on_the_source_line():
    mach, q1, q3 = _mock_machine()
    taus = [-1.0, 0.0, 0.5]
    cfg = patch_config(_config(), q1, q3, k=0.1, c_amp=0.09, via_phase=0.0, tau_sw_ns=taus, pad_ns=4, cal_via_scale=0.5)
    via = cfg["elements"]["q3.xy_via"]
    assert via["MWInput"]["port"] == ("con1", 6, 4)
    assert via["intermediate_frequency"] == pytest.approx(192.0e6 + (5.1446e9 - 5.1920e9))
    assert via["core"] not in ("a", "c")
    assert "q3.xy_cancel" not in cfg["elements"]
    # L1 does not depend on tau_sw: one op per (kflip, sign); L2 one per (tau, kflip +-1, sign)
    assert sum(op.startswith("skewl1_") for op in cfg["elements"]["q1.xy"]["operations"]) == 3 * 2
    assert not any(op.startswith("skew_t") for op in cfg["elements"]["q1.xy"]["operations"])
    assert sum(op.startswith("skew_t") for op in via["operations"]) == len(taus) * 2 * 2
    assert {"skewcal_0", "skewcal_90", "skewnull"} <= set(via["operations"])
    # every C(-) op is the exact negative of its C(+) op, on both lines; so are the flip ops
    for i in range(len(taus)):
        assert np.allclose(_iq(cfg, "q3.xy_via", scan_op(i, 1, 1)), -_iq(cfg, "q3.xy_via", scan_op(i, 1, -1)))
    for kflip in (1, -1, 0):
        assert np.allclose(_iq(cfg, "q1.xy", l1_op(kflip, 1)), -_iq(cfg, "q1.xy", l1_op(kflip, -1)))
    assert np.allclose(_iq(cfg, "q1.xy", flip_op(1)), -_iq(cfg, "q1.xy", flip_op(-1)))
    with pytest.raises(ValueError, match="lower k"):
        patch_config(_config(), q1, q3, k=0.5, c_amp=0.09, via_phase=0.0, tau_sw_ns=taus, pad_ns=4, cal_via_scale=0.5)


def test_loop_layout_runs_one_pair_loop_per_element():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    prog, axes = build_scan_program(
        mach, q1, q3, tau_indices=[0, 1], tau_values=[-0.5, 0.5], repetitions=[2, 4, 8, 16], kflips=[1, -1, 0],
        pulse_cycles=scan_pulse_cycles(16, 4), num_shots=5, use_state_discrimination=True,
    )
    sc = generate_qua_script(prog).replace("'", '"')
    assert re.search(r'reset_global_phase\(\)\s+align\("q1\.xy", "q3\.xy_via"\)', sc)
    assert "if_(" not in sc and "switch_(" not in sc and "amp(" not in sc and "strict_timing_" not in sc
    # N is a QUA loop: the program size does not grow with N
    assert "for_each_(" in sc and "[2, 4, 8, 16]" in sc
    # per (tau, kflip) one body: L1 always plays its pair, L2 only at kflip +-1
    assert len(re.findall(r'play\("skewl1_k[pm0]_s[pm]", "q1\.xy"\)', sc)) == 2 * 3 * 2
    assert len(re.findall(r'play\("skew_t[01]_k[pm]_s[pm]", "q3\.xy_via"\)', sc)) == 2 * 2 * 2
    # two separate pair loops (own variables), not one shared loop
    assert len(re.findall(r"with for_\(", sc)) == 1 + 2 * 3 + 2 * 2
    assert list(axes) == ["qubit", "tau_sw_ns", "kflip", "repetitions"]
    assert "buffer(2, 3, 4)" in sc


def test_unrolled_layout_plays_both_lines_in_lockstep_without_branches():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    prog, axes = build_scan_program(
        mach, q1, q3, tau_indices=[0, 1], tau_values=[-0.5, 0.5], repetitions=[1, 3], kflips=[1, -1, 0],
        pulse_cycles=scan_pulse_cycles(16, 4), num_shots=5, use_state_discrimination=True, layout="unrolled",
    )
    sc = generate_qua_script(prog).replace("'", '"')
    assert "reset_global_phase()" in sc and "strict_timing_()" in sc
    # both timelines are aligned between the phase reset and the pulse train
    assert re.search(r'reset_global_phase\(\)\s+align\("q1\.xy", "q3\.xy_via"\)\s+with strict_timing_', sc)
    assert "if_(" not in sc and "switch_(" not in sc and "amp(" not in sc and "timestamp" not in sc
    assert len(re.findall(r'play\("skewl1_k[pm0]_s[pm]", "q1\.xy"\)', sc)) == 2 * 4 * 2 * 3
    assert len(re.findall(r'play\("skew_t0_k[pm]_s[pm]", "q3\.xy_via"\)', sc)) == 4 * 2 * 2
    assert list(axes["kflip"].values) == [1, -1, 0]


def test_debug_timestamps_stamp_every_play():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    for layout, plays in (("loops", 2 * 2), ("unrolled", 2 * 2 * 2)):
        prog, _ = build_scan_program(
            mach, q1, q3, tau_indices=[0], tau_values=[0.0], repetitions=[2], kflips=[1, -1], pulse_cycles=6,
            num_shots=2, use_state_discrimination=True, debug_timestamps=True, layout=layout,
        )
        sc = generate_qua_script(prog).replace("'", '"')
        for name in ("ts_y90", "ts_l1", "ts_l2"):
            assert f'save_all("{name}")' in sc
        # every play of both lines carries a timestamp stream
        assert len(re.findall(r'play\("skewl1_k[pm]_s[pm]", "q1\.xy", timestamp_stream=', sc)) == plays
        assert len(re.findall(r'play\("skew_t0_k[pm]_s[pm]", "q3\.xy_via", timestamp_stream=', sc)) == plays


def test_calibrate_program_with_null_and_flip_conditions():
    from qm import generate_qua_script

    mach, q1, q3 = _mock_machine()
    conds = (*CAL_CONDITIONS, "null", "l1_flip", "l2_flip")
    prog, axes = build_calibrate_program(mach, q1, q3, repetitions=[0, 4], cal_own_scale=0.05, num_shots=2,
                                         conditions=conds, flip_tau_index=1, scan_cycles=6,
                                         use_state_discrimination=True)
    sc = generate_qua_script(prog).replace("'", '"')
    assert len(re.findall(r'play\("skewcal_0", "q3\.xy_via"\)', sc)) == 4
    assert len(re.findall(r'play\("skewnull", "q1\.xy"\)', sc)) == 4
    assert len(re.findall(r'play\("skewnull", "q3\.xy_via"\)', sc)) == 4
    # flip trains alternate + and - on one line only
    assert re.findall(r'play\("skewflip_s([pm])", "q1\.xy"\)', sc) == ["p", "m", "p", "m"]
    assert re.findall(r'play\("skew_t1_kp_s([pm])", "q3\.xy_via"\)', sc) == ["p", "m", "p", "m"]
    assert "if_(" not in sc and "switch_(" not in sc
    assert list(axes["condition"].values) == list(conds)
    assert list(axes["basis"].values) == list(BASES)


def test_the_qm_shell_is_the_registered_experiment():
    from scqo import experiments

    assert experiments.get("xy_skew_pingpong") is QMXySkewPingpong
    p = XySkewPingpongParameters(targets=["q1", "q3"], debug_timestamps=True, tau_sw_ns=[0.5, 0.0])
    exp = QMXySkewPingpong(SimpleNamespace(device=None), p)
    assert list(exp.define_sweep()["tau_sw_ns"]) == [0.5]
