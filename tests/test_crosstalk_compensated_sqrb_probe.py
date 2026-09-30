"""Tests for CrosstalkCompensatedSQRB QUA probe generation."""

from types import SimpleNamespace
import numpy as np
import pytest

from scqo_qm.experiments.crosstalk_compensated_sqrb import (
    CLIFFORD_3SUBSLOTS,
    QMCrosstalkCompensatedSQRB,
    build_program,
    compute_theoretical_phase_rate,
)


from qm.qua import declare, declare_stream, fixed


def _mock_machine():
    """Mock QUAM machine with probe and drive qubits."""
    def _decl(num_IQ_pairs=1):
        return (
            [declare(fixed) for _ in range(num_IQ_pairs)],
            [declare_stream() for _ in range(num_IQ_pairs)],
            [declare(fixed) for _ in range(num_IQ_pairs)],
            [declare_stream() for _ in range(num_IQ_pairs)],
            declare(int),
            declare_stream(),
        )

    q1_xy = SimpleNamespace(
        name="q1.xy",
        operations={"x180": SimpleNamespace(length=20)},
        wait=lambda *a: None,
        play=lambda *a: None,
    )
    q2_xy = SimpleNamespace(
        name="q2.xy",
        operations={"x180": SimpleNamespace(length=20)},
        wait=lambda *a: None,
        play=lambda *a: None,
    )

    q1_res = SimpleNamespace(name="q1_res", measure=lambda *a, **kw: None)
    q2_res = SimpleNamespace(name="q2_res", measure=lambda *a, **kw: None)

    q1 = SimpleNamespace(
        name="q1",
        xy=q1_xy,
        resonator=q1_res,
        reset=lambda *a: None,
        readout_state=lambda *a: None,
    )
    q2 = SimpleNamespace(
        name="q2",
        xy=q2_xy,
        resonator=q2_res,
        reset=lambda *a: None,
        readout_state=lambda *a: None,
    )

    mach = SimpleNamespace(
        declare_qua_variables=_decl,
        initialize_qpu=lambda **kw: None,
        qubits={"q1": q1, "q2": q2},
    )
    return mach, q1, q2


def test_theoretical_formula():
    f_d = 5.2e9
    f_p = 5.1e9
    slot_ns = 40.0
    rate = compute_theoretical_phase_rate(f_d, f_p, slot_ns)
    # Expected: 2 * pi * 100 MHz * 40 ns = 8 * pi
    assert np.isclose(rate, 2 * np.pi * 100e6 * 40e-9)


def test_build_program_benchmark_mode():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        depths=[1, 2, 4],
        num_sequences=2,
        num_shots=10,
        cancel_amp=0.035,
        init_phase=0.85,
        phase_rate_turns=0.25,
        cancel_amps=[0.01, 0.05],
        init_phases=[0.0, 1.0],
        cal_depth=10,
        reset_type="thermal",
        use_state_discrimination=False,
    )

    assert "condition" in sweep_axes
    assert "depth" in sweep_axes
    assert list(sweep_axes["condition"].values) == ["isolated", "simultaneous", "compensated"]
    assert len(sweep_axes["depth"]) == 3
    assert prog is not None


def test_build_program_calibrate_mode():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="calibrate",
        target_gate="x180",
        cal_stage_repetitions=[10, 20, 30],
        num_shots=10,
        cancel_amps=[0.01, 0.05, 0.1],
        init_phases=[-0.5, 0.0, 0.5],
        sub_phase_rate_turns=0.25,
        reset_type="thermal",
        use_state_discrimination=False,
    )

    assert "cancel_amp" in sweep_axes
    assert "init_phase" in sweep_axes
    assert len(sweep_axes["cancel_amp"]) == 3
    assert len(sweep_axes["init_phase"]) == 3
    assert prog is not None

    from qm import generate_qua_script
    script = generate_qua_script(prog)
    assert 'frame_rotation_2pi(-0.25, "q2.xy")' in script or "frame_rotation_2pi(-0.25, 'q2.xy')" in script
    assert 'frame_rotation_2pi(0.25, "q1.xy_cancel")' not in script and "frame_rotation_2pi(0.25, 'q1.xy_cancel')" not in script
    assert 'frame_rotation_2pi(-0.25, "q1.xy_cancel")' not in script and "frame_rotation_2pi(-0.25, 'q1.xy_cancel')" not in script


def test_clifford_3subslots_structure():
    assert len(CLIFFORD_3SUBSLOTS) == 24
    for idx, subslots in enumerate(CLIFFORD_3SUBSLOTS):
        assert len(subslots) == 3, f"Gate {idx} does not have 3 subslots"
        for slot in subslots:
            if slot is not None:
                assert len(slot) == 3
                op_name, pulse_kind, axis_phase = slot
                assert pulse_kind in ("x90", "x180")
                assert 0.0 <= axis_phase < 1.0


def test_build_program_with_sub_phase_rate():
    mach, q1, q2 = _mock_machine()
    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        mode="benchmark",
        depths=[1, 2],
        num_sequences=2,
        num_shots=5,
        cancel_amp=0.02,
        init_phase=0.5,
        sub_phase_rate_turns=0.08,
        cancel_amps=[0.01, 0.05],
        init_phases=[0.0, 1.0],
        cal_depth=5,
        reset_type="thermal",
    )
    assert prog is not None


def test_patch_preview_config():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2"),
    )

    cfg = {
        "elements": {
            "q1.xy": {"core": "a", "operations": {"x180": "q1_x180_pulse"}, "MWInput": {"port": ("con1", 6, 2)}},
            "q2.xy": {"core": "b", "operations": {"x180": "q2_x180_pulse"}, "MWInput": {"port": ("con1", 6, 3)}},
        }
    }

    patched = exp.patch_preview_config(cfg)
    cancel_key = "q1.xy_cancel" if "q1.xy_cancel" in patched["elements"] else "q1_xy_cancel"
    assert cancel_key in patched["elements"]
    assert patched["elements"][cancel_key]["core"] != patched["elements"]["q1.xy"]["core"]
    assert patched["elements"][cancel_key]["core"] not in ["a", "b"]
    assert patched["elements"][cancel_key]["operations"]["x180"] == "q2_x180_pulse"


def test_preview_program():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark"),
    )
    prog = exp.preview_program()
    assert prog is not None
    script = generate_qua_script(prog)
    assert "q1.xy_cancel" in script or "q1_xy_cancel" in script
    assert "cond_idx" not in script
    assert "case_(0)" not in script
    assert "case_(1)" not in script


def test_acquire_benchmark_batched(monkeypatch):
    import xarray as xr
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(
            probe_qubit="q1",
            drive_qubit="q2",
            mode="benchmark",
            depths=[2, 4, 8],
            benchmark_batch_size=1,
            num_random_sequences=2,
        ),
    )

    call_count = 0
    def mock_acquire(m, p, s, **kw):
        nonlocal call_count
        call_count += 1
        d_vals = s["depth"].values
        c_vals = s["condition"].values
        n_seq = len(s["sequence_idx"].values)
        return xr.Dataset(
            data_vars={"I": (("qubit", "condition", "sequence_idx", "depth"), np.zeros((1, len(c_vals), n_seq, len(d_vals))))},
            coords={
                "qubit": ["q1"],
                "condition": c_vals,
                "sequence_idx": s["sequence_idx"].values,
                "depth": d_vals,
            },
        )

    import scqo_qm.experiments.crosstalk_compensated_sqrb as mod
    monkeypatch.setattr(mod, "_acquire", mock_acquire)

    res = exp._acquire_benchmark(mach, None, None, num_shots=10, probe_qubit=q1, drive_qubit=q2)
    assert call_count == 9
    assert list(res.coords["depth"].values) == [2, 4, 8]
    assert list(res.coords["condition"].values) == ["isolated", "simultaneous", "compensated"]
    assert list(res.coords["sequence_idx"].values) == [0, 1]


def test_strict_timing_toggle():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    exp_strict = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark", strict_timing=True),
    )
    prog_strict = exp_strict.preview_program()
    script_strict = generate_qua_script(prog_strict)
    assert "strict_timing_" in script_strict

    exp_loose = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", mode="benchmark", strict_timing=False),
    )
    prog_loose = exp_loose.preview_program()
    script_loose = generate_qua_script(prog_loose)
    assert "strict_timing_" not in script_loose


def test_compensation_3subslot_decomposition():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    from qm import generate_qua_script

    assert len(CLIFFORD_3SUBSLOTS) == 24
    for gate_subslots in CLIFFORD_3SUBSLOTS:
        assert len(gate_subslots) == 3

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(
            probe_qubit="q1",
            drive_qubit="q2",
            mode="benchmark",
            strict_timing=True,
            phase_rate=0.5,
        ),
    )
    prog = exp.preview_program()
    script = generate_qua_script(prog)
    assert "q1.xy_cancel" in script or "q1_xy_cancel" in script
    assert "frame_rotation_2pi" in script





def _recording_machine():
    """Mock machine whose qubit macros log (event, element, op) in program order."""
    mach, q1, q2 = _mock_machine()
    events = []
    for q in (q1, q2):
        xy_name = q.xy.name
        q.xy.play = lambda op, _n=xy_name: events.append(("play", _n, op))
        q.xy.wait = lambda *a, _n=xy_name: events.append(("wait", _n, None))
        q.resonator.measure = lambda *a, _n=q.name, **kw: events.append(("measure", _n, None))
        q.reset = lambda *a, _n=q.name: events.append(("reset", _n, None))
    return mach, q1, q2, events


def _script(prog):
    from qm import generate_qua_script

    return generate_qua_script(prog).replace("'", '"')


def test_bidirectional_reads_both_qubits():
    mach, q1, q2 = _mock_machine()
    prog, sweep_axes = build_program(
        mach, q1, q2, mode="benchmark", depths=[1, 2], num_sequences=2, num_shots=5,
        cancel_amp=0.03, init_phase=0.5, sub_phase_rate_turns=0.1, bidirectional=True,
        drive_cancel_amp=0.02, drive_init_phase=np.pi / 2,
    )
    assert list(sweep_axes["qubit"].values) == ["q1", "q2"]
    script = _script(prog)
    for stream in ("I1", "Q1", "I2", "Q2"):
        assert f'"{stream}"' in script


def test_bidirectional_reverse_cancel_track():
    mach, q1, q2 = _mock_machine()
    prog, _ = build_program(
        mach, q1, q2, mode="benchmark", conditions=["compensated"], depths=[2], num_sequences=1,
        num_shots=1, cancel_amp=0.03, init_phase=0.0, sub_phase_rate_turns=0.1, bidirectional=True,
        drive_cancel_amp=0.02, drive_init_phase=np.pi / 2,
    )
    script = _script(prog)
    # Forward track on the probe line, reverse track on the drive line with the opposite rate
    assert 'frame_rotation_2pi(0.1, "q1.xy_cancel")' in script
    assert 'frame_rotation_2pi(-0.1, "q2.xy_cancel")' in script
    assert 'frame_rotation_2pi(0.25, "q2.xy_cancel")' in script  # drive_init_phase = pi/2
    assert "*amp(0.02)" in script.replace(" ", "")
    assert 'reset_frame("q2.xy_cancel")' in script


def test_unidirectional_has_no_reverse_track():
    mach, q1, q2 = _mock_machine()
    prog, sweep_axes = build_program(
        mach, q1, q2, mode="benchmark", conditions=["compensated"], depths=[2], num_sequences=1,
        num_shots=1, cancel_amp=0.03, sub_phase_rate_turns=0.1,
    )
    assert list(sweep_axes["qubit"].values) == ["q1"]
    assert "q2.xy_cancel" not in _script(prog)


def test_bidirectional_isolated_runs_each_qubit_alone():
    mach, q1, q2, events = _recording_machine()
    build_program(
        mach, q1, q2, mode="benchmark", conditions=["isolated"], depths=[1], num_sequences=1,
        num_shots=1, bidirectional=True, seed=3,
    )
    measures = [i for i, e in enumerate(events) if e[0] == "measure"]
    assert [events[i][1] for i in measures] == ["q1", "q2"]
    first, second = events[: measures[0]], events[measures[0] + 1 : measures[1]]
    # Pass 1: probe alone, then read probe. Pass 2: drive alone, then read drive.
    assert {e[1] for e in first if e[0] in ("play", "wait")} == {"q1.xy"}
    assert {e[1] for e in second if e[0] in ("play", "wait")} == {"q2.xy"}
    # Both qubits are reset before each pass (a leftover |1> would ZZ-shift the other)
    assert [e[1] for e in first if e[0] == "reset"] == ["q1", "q2"]
    assert [e[1] for e in second if e[0] == "reset"] == ["q1", "q2"]


def test_bidirectional_simultaneous_reads_both_after_one_pass():
    mach, q1, q2, events = _recording_machine()
    build_program(
        mach, q1, q2, mode="benchmark", conditions=["simultaneous"], depths=[1], num_sequences=1,
        num_shots=1, bidirectional=True, seed=3, readout_scheme="simultaneous",
    )
    assert [e[1] for e in events if e[0] == "measure"] == ["q1", "q2"]
    assert [e[1] for e in events if e[0] == "reset"] == ["q1", "q2"]


def _two_line_config():
    ops = ("x180", "y180", "x90", "-x90", "y90", "-y90")
    return {
        "elements": {
            "q1.xy": {"core": "a", "operations": {op: f"q1_{op}_pulse" for op in ops}, "MWInput": {"port": ("con1", 6, 2)}},
            "q2.xy": {"core": "b", "operations": {op: f"q2_{op}_pulse" for op in ops}, "MWInput": {"port": ("con1", 6, 3)}},
        }
    }


def test_cancel_element_mirrors_every_aggressor_clifford_op():
    from scqo_qm.experiments.crosstalk_compensated_sqrb import CANCEL_MIRROR_OPS, _patch_cancel_element

    mach, q1, q2 = _mock_machine()
    assert set(CANCEL_MIRROR_OPS) == {"x180", "y180", "x90", "-x90", "y90", "-y90"}
    cfg = _patch_cancel_element(_two_line_config(), q1, q2, "x180")
    ops = cfg["elements"]["q1.xy_cancel"]["operations"]
    for op in CANCEL_MIRROR_OPS:
        assert ops[op] == f"q2_{op}_pulse"
    # Still on the victim's port
    assert cfg["elements"]["q1.xy_cancel"]["MWInput"]["port"] == ("con1", 6, 2)


def test_patch_preview_config_bidirectional():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", bidirectional=True),
    )
    els = exp.patch_preview_config(_two_line_config())["elements"]
    fwd, rev = els["q1.xy_cancel"], els["q2.xy_cancel"]
    assert rev["MWInput"]["port"] == ("con1", 6, 3)
    assert rev["operations"]["x90"] == "q1_x90_pulse"
    assert fwd["operations"]["x90"] == "q2_x90_pulse"
    cores = [els[k]["core"] for k in ("q1.xy", "q2.xy", "q1.xy_cancel", "q2.xy_cancel")]
    assert len(set(cores)) == 4


def test_phase_rates_are_opposite_by_default():
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", bidirectional=True, phase_rate=0.6),
    )
    fwd, rev = exp._phase_rates_turns(q1, q2)
    assert np.isclose(fwd, 0.6 / (2 * np.pi))
    assert np.isclose(rev, -fwd)

    exp.params.drive_phase_rate = 1.0
    _, rev = exp._phase_rates_turns(q1, q2)
    assert np.isclose(rev, 1.0 / (2 * np.pi))


def test_acquire_benchmark_bidirectional(monkeypatch):
    import xarray as xr
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters
    import scqo_qm.experiments.crosstalk_compensated_sqrb as mod

    mach, q1, q2 = _mock_machine()
    params = dict(
        probe_qubit="q1", drive_qubit="q2", mode="benchmark", depths=[8],
        num_random_sequences=2, max_gates_per_batch=20,
    )

    def run(bidirectional):
        calls = []

        def mock_acquire(m, p, s, **kw):
            calls.append(list(s["qubit"].values))
            n_q = len(s["qubit"].values)
            return xr.Dataset(
                data_vars={"I": (("qubit", "condition", "sequence_idx", "depth"),
                                 np.zeros((n_q, 1, len(s["sequence_idx"]), 1)))},
                coords={"qubit": s["qubit"].values, "condition": s["condition"].values,
                        "sequence_idx": s["sequence_idx"].values, "depth": s["depth"].values},
            )

        monkeypatch.setattr(mod, "_acquire", mock_acquire)
        exp = QMCrosstalkCompensatedSQRB(
            SimpleNamespace(machine=mach, device=None),
            CrosstalkCompensatedSQRBParameters(bidirectional=bidirectional, **params),
        )
        res = exp._acquire_benchmark(mach, None, None, num_shots=1, probe_qubit=q1, drive_qubit=q2)
        return calls, res

    calls_uni, _ = run(False)
    calls_bi, res_bi = run(True)
    assert all(c == ["q1"] for c in calls_uni)
    assert all(c == ["q1", "q2"] for c in calls_bi)
    # depth 8 = 9 gates/seq: both sequences fit one chunk at a budget of 20, but bidirectional
    # doubles the load to 18 so each sequence gets its own chunk (3 conditions each)
    assert len(calls_uni) == 3
    assert len(calls_bi) == 6
    assert list(res_bi.coords["qubit"].values) == ["q1", "q2"]
    assert list(res_bi.coords["sequence_idx"].values) == [0, 1]


@pytest.mark.parametrize("scheme, passes, reads", [
    ("separate_passes", 2, [["q1"], ["q2"]]),
    ("sequential", 1, [["q1", "q2"]]),
    ("simultaneous", 1, [["q1", "q2"]]),
])
def test_bidirectional_readout_scheme(scheme, passes, reads):
    mach, q1, q2, events = _recording_machine()
    build_program(
        mach, q1, q2, mode="benchmark", conditions=["compensated"], depths=[1], num_sequences=1,
        num_shots=1, bidirectional=True, seed=3, readout_scheme=scheme,
    )
    # a pass = reset of both qubits, gates, then its readouts
    chunks, cur = [], []
    for kind, who, _ in events:
        if kind == "reset" and cur and cur[-1][0] == "measure":
            chunks.append(cur)
            cur = []
        cur.append((kind, who))
    chunks.append(cur)
    assert len(chunks) == passes
    assert [[w for k, w in c if k == "measure"] for c in chunks] == reads
    # every pass plays both qubits' gates
    for c in chunks:
        assert {w for k, w in c if k == "play"} == {"q1.xy", "q2.xy"}


def test_sequential_readout_aligns_between_measurements():
    mach, q1, q2 = _mock_machine()
    seq, _ = build_program(mach, q1, q2, mode="benchmark", conditions=["compensated"], depths=[1],
                           num_sequences=1, num_shots=1, bidirectional=True, readout_scheme="sequential")
    sim, _ = build_program(mach, q1, q2, mode="benchmark", conditions=["compensated"], depths=[1],
                           num_sequences=1, num_shots=1, bidirectional=True, readout_scheme="simultaneous")
    assert _script(seq).count("align()") == _script(sim).count("align()") + 1
