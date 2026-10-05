"""Tests for CrosstalkCompensatedSQRB QUA probe generation (FPGA loops, aggressor-tone cancels)."""

import re
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from qm.qua import declare, declare_stream, fixed, program

from scqo_qm.experiments.crosstalk_compensated_sqrb import (
    CANCEL_MIRROR_OPS,
    CLIFFORD_3SUBSLOTS,
    MAX_ARRAY_GATES,
    QMCrosstalkCompensatedSQRB,
    _passes,
    _patch_cancel_element,
    _patch_composite_cliffords,
    _played_elements,
    build_program,
    clifford_sequence,
    patch_config,
    recovery_gates,
    sequence_chunks,
    subslot_phase_step,
)


def _mock_machine():
    """Mock QUAM machine with probe (q1) and drive (q2) qubits."""
    def _decl(num_IQ_pairs=1):
        return (
            [declare(fixed) for _ in range(num_IQ_pairs)],
            [declare_stream() for _ in range(num_IQ_pairs)],
            [declare(fixed) for _ in range(num_IQ_pairs)],
            [declare_stream() for _ in range(num_IQ_pairs)],
            declare(int),
            declare_stream(),
        )

    def qubit(name, rf):
        xy = SimpleNamespace(name=f"{name}.xy", operations={"x180": SimpleNamespace(length=16)}, RF_frequency=rf)
        res = SimpleNamespace(name=f"{name}.res", measure=lambda *a, **kw: None)
        return SimpleNamespace(name=name, xy=xy, resonator=res, reset=lambda *a, **kw: None, readout_state=lambda *a: None)

    q1, q2 = qubit("q1", 5.143e9), qubit("q2", 5.191e9)
    mach = SimpleNamespace(declare_qua_variables=_decl, initialize_qpu=lambda **kw: None, qubits={"q1": q1, "q2": q2})
    return mach, q1, q2


def _recording_machine():
    """Mock machine whose qubit macros log (event, qubit) in program order."""
    mach, q1, q2 = _mock_machine()
    events = []
    for q in (q1, q2):
        q.resonator.measure = lambda *a, _n=q.name, **kw: events.append(("measure", _n))
        q.reset = lambda *a, _n=q.name, **kw: events.append(("reset", _n))
    return mach, q1, q2, events


def _script(prog):
    from qm import generate_qua_script

    return generate_qua_script(prog).replace("'", '"')


def _program(**kw):
    mach, q1, q2 = _mock_machine()
    args = dict(depths=[1, 8], num_sequences=2, num_shots=2, cancel_amp=0.03, init_phase=0.5)
    args.update(kw)
    return build_program(mach, q1, q2, **args)


def _full_config():
    """Two xy lines with every Clifford op defined (4-sample pulses), IF and cores set."""
    cfg = {"elements": {}, "pulses": {}, "waveforms": {}}
    for name, port, core, if_hz in (("q1.xy", 2, "a", 243e6), ("q2.xy", 4, "c", 191e6)):
        ops = {}
        for k, op in enumerate(CANCEL_MIRROR_OPS):
            pulse = f"{name}.{op}.pulse"
            cfg["waveforms"][f"{pulse}.I"] = {"type": "arbitrary", "samples": [0.1 * (k + 1)] * 4}
            cfg["waveforms"][f"{pulse}.Q"] = {"type": "constant", "sample": 0.0}
            cfg["pulses"][pulse] = {"operation": "control", "length": 4, "waveforms": {"I": f"{pulse}.I", "Q": f"{pulse}.Q"}}
            ops[op] = pulse
        cfg["elements"][name] = {
            "core": core, "operations": ops, "intermediate_frequency": if_hz, "MWInput": {"port": ("con1", 6, port)},
        }
    return cfg


# --------------------------------------------------------------------------- sequences


def test_clifford_3subslots_structure():
    assert len(CLIFFORD_3SUBSLOTS) == 24
    assert set(CANCEL_MIRROR_OPS) == {"x180", "y180", "x90", "-x90", "y90", "-y90"}


def test_sequences_are_prefixes_with_correct_recovery():
    from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

    gates = clifford_sequence(42, 3, 200)
    assert gates[:50] == clifford_sequence(42, 3, 50)
    assert gates != clifford_sequence(42, 4, 200)
    depths = [1, 7, 128, 200]
    for d, inv in zip(depths, recovery_gates(gates, depths)):
        state = 0
        for g in gates[:d] + [inv]:
            state = int(c1_table[state, g])
        assert state == 0


def test_programs_declare_the_sequences_of_their_chunk():
    prog, axes = _program(depths=[3, 5], sequence_indices=[4, 5], num_sequences=6)
    arrays = [list(map(int, a.split(","))) for a in re.findall(r"declare\(int, value=\[([0-9, ]+)\]\)", _script(prog))]
    # one spare slot per sequence: the recovery of the depth being played goes there
    gp = clifford_sequence(42, 4, 5) + [0] + clifford_sequence(42, 5, 5) + [0]
    gd = clifford_sequence(1042, 4, 5) + [0] + clifford_sequence(1042, 5, 5) + [0]
    assert gp in arrays and gd in arrays
    assert list(axes["sequence_idx"].values) == [4, 5]


def test_sequence_chunks_respect_program_length_and_array_memory():
    assert sequence_chunks(30, 1024, 7) == [list(range(i, min(i + 7, 30))) for i in range(0, 30, 7)]
    by_memory = MAX_ARRAY_GATES // (2 * 4097)
    assert all(len(c) <= by_memory for c in sequence_chunks(10, 4096, 7))
    assert sum(sequence_chunks(10, 4096, 7), []) == list(range(10))


# --------------------------------------------------------------------------- program


def test_axes_and_streams_unidirectional_and_bidirectional():
    _, axes = _program()
    assert list(axes["qubit"].values) == ["q1"]
    assert list(axes["condition"].values) == ["isolated", "simultaneous", "compensated"]
    prog, axes = _program(bidirectional=True, drive_cancel_amp=0.02)
    assert list(axes["qubit"].values) == ["q1", "q2"]
    sc = _script(prog)
    for stream in ("I1", "Q1", "I2", "Q2"):
        assert f'"{stream}"' in sc
    with pytest.raises(ValueError, match="depths"):
        _program(depths=[0, 4])


def test_passes_and_played_elements():
    assert _passes("isolated", False) == [("isolated", [0])]
    assert _passes("isolated", True) == [("isolated", [0]), ("isolated_drive", [1])]
    assert _passes("compensated", True) == [("compensated", [0, 1])]

    mach, q1, q2 = _mock_machine()
    kw = dict(probe_qubit=q1, drive_qubit=q2, cancel_elem="q1.xy_cancel", reverse_cancel_elem="q2.xy_cancel",
              cancel_amp=0.03, drive_cancel_amp=0.02)
    assert _played_elements("isolated", **kw) == [("q1.xy", "p", None)]
    assert _played_elements("isolated_drive", **kw) == [("q2.xy", "d", None)]
    assert _played_elements("simultaneous", **kw) == [("q1.xy", "p", None), ("q2.xy", "d", None)]
    assert _played_elements("compensated", **kw) == [
        ("q1.xy", "p", None), ("q2.xy", "d", None), ("q1.xy_cancel", "d", 0.03), ("q2.xy_cancel", "p", 0.02),
    ]


def test_bidirectional_isolated_reads_each_qubit_after_its_own_pass():
    mach, q1, q2, events = _recording_machine()
    build_program(mach, q1, q2, depths=[1], num_sequences=1, num_shots=1, conditions=["isolated"], bidirectional=True)
    assert [e for e in events if e[0] == "measure"] == [("measure", "q1"), ("measure", "q2")]
    mach, q1, q2, events = _recording_machine()
    build_program(mach, q1, q2, depths=[1], num_sequences=1, num_shots=1, conditions=["simultaneous"], bidirectional=True)
    assert [e for e in events if e[0] == "measure"] == [("measure", "q1"), ("measure", "q2")]
    assert [e for e in events if e[0] == "reset"] == [("reset", "q1"), ("reset", "q2")]


def test_every_element_has_its_own_loop():
    """Parallel loops (each on its own variable) play 48 ns Cliffords back to back on hardware."""
    sc = _script(_program(conditions=["compensated"], bidirectional=True, drive_cancel_amp=0.02)[0])
    lines = sc.splitlines()
    elem_loops = []
    for i, line in enumerate(lines):
        m = re.match(r"\s*with for_\((v\d+),", line)
        if m:
            window = " ".join(ln for ln in lines[i + 1:i + 4] if "play(" in ln)
            for elem in ("q1.xy_cancel", "q2.xy_cancel", "q1.xy", "q2.xy"):
                if f'"{elem}")' in window:
                    elem_loops.append((m.group(1), elem))
                    break
    assert sorted(e for _, e in elem_loops) == ["q1.xy", "q1.xy_cancel", "q2.xy", "q2.xy_cancel"]
    assert len({v for v, _ in elem_loops}) == 4
    assert sc.count("unsafe=True):") == 4  # the recovery is the loop's last Clifford
    # the loop index is the bare loop variable (a computed index costs ~70 ns per Clifford)
    assert not re.search(r"\[\(v\d+\+\d+\)\]", sc)


def test_program_size_does_not_grow_with_depth():
    small = _script(_program(depths=[1, 4], conditions=["compensated"])[0])
    deep = _script(_program(depths=[1, 4, 1024, 2000], conditions=["compensated"])[0])
    assert small.count("play(") == deep.count("play(")


def test_cancel_copies_play_scaled_with_their_phase_and_no_frame_steps():
    sc = _script(_program(conditions=["compensated"], bidirectional=True, init_phase=np.pi / 2,
                          drive_cancel_amp=0.02, drive_init_phase=-np.pi)[0])
    assert 'frame_rotation_2pi(0.25, "q1.xy_cancel")' in sc
    assert 'frame_rotation_2pi(-0.5, "q2.xy_cancel")' in sc
    # only the initial phases: an aggressor-tone copy needs no per-pulse frame steps
    assert len(re.findall(r'frame_rotation_2pi\([^)]*"q[12]\.xy_cancel"\)', sc)) == 2
    assert "*amp(0.03)" in sc.replace(" ", "") and "*amp(0.02)" in sc.replace(" ", "")
    assert "reset_global_phase()" in sc


def test_recovery_is_written_into_the_loop_and_restored():
    sc = _script(_program(conditions=["compensated"])[0])
    # slot d of each gate array takes the recovery before the shots and gets its gate back after
    writes = re.findall(r"assign\((a\d+)\[(v\d+)\], (a\d+)\[", sc)
    restores = re.findall(r"assign\((a\d+)\[(v\d+)\], (v\d+)\)", sc)
    assert len(writes) == 2 and len(restores) == 2
    assert {(a, i) for a, i, _ in writes} == {(a, i) for a, i, _ in restores}


def test_debug_timestamps_bracket_each_element_loop():
    sc = _script(_program(conditions=["simultaneous"], debug_timestamps=True)[0])
    for edge in ("ts0", "ts1"):
        for elem in ("q1.xy", "q2.xy"):
            assert f'"{edge}_simultaneous_{elem}"' in sc


# --------------------------------------------------------------------------- config


def test_cancel_element_is_a_copy_of_the_aggressor_at_its_frequency():
    mach, q1, q2 = _mock_machine()
    cfg = _patch_cancel_element(_full_config(), q1, q2)
    cancel = cfg["elements"]["q1.xy_cancel"]
    for op in CANCEL_MIRROR_OPS:
        assert cancel["operations"][op] == f"q2.xy.{op}.pulse"
    assert cancel["MWInput"]["port"] == ("con1", 6, 2)
    assert np.isclose(cancel["intermediate_frequency"], 243e6 + (5.191e9 - 5.143e9))
    assert cancel["core"] not in ("a", "c")
    q2.xy.RF_frequency = 5.6e9
    with pytest.raises(ValueError, match="400 MHz"):
        _patch_cancel_element(_full_config(), q1, q2)


def test_patch_config_orders_cancel_cores_by_victim_name():
    """The phase reference is per core: the tomography (same rule) calibrates on the same core."""
    mach, q1, q2 = _mock_machine()
    a = patch_config(_full_config(), q1, q2, bidirectional=True)
    b = patch_config(_full_config(), q2, q1, bidirectional=True)
    for name in ("q1.xy_cancel", "q2.xy_cancel"):
        assert a["elements"][name]["core"] == b["elements"][name]["core"]
    assert "clif23" in a["elements"]["q2.xy_cancel"]["operations"]


def test_composite_cliffords_concatenate_the_three_subslots():
    n = 4
    shapes = {"x180": 1.0, "y180": 1j, "x90": 0.5, "-x90": -0.5, "y90": 0.5j, "-y90": -0.5j}
    cfg = {"elements": {"q.xy": {"operations": {}}, "a.xy": {"operations": {}}}, "pulses": {}, "waveforms": {}}
    for op, z in shapes.items():
        cfg["waveforms"][f"{op}.I"] = {"type": "constant", "sample": float(np.real(z))}
        cfg["waveforms"][f"{op}.Q"] = {"type": "arbitrary", "samples": [float(np.imag(z))] * n}
        cfg["pulses"][f"{op}.p"] = {"operation": "control", "length": n, "waveforms": {"I": f"{op}.I", "Q": f"{op}.Q"}}
        cfg["elements"]["a.xy"]["operations"][op] = f"{op}.p"
    _patch_composite_cliffords(cfg, "q.xy", "a.xy")
    for g, subslots in enumerate(CLIFFORD_3SUBSLOTS):
        pulse = cfg["pulses"][cfg["elements"]["q.xy"]["operations"][f"clif{g}"]]
        assert pulse["length"] == 3 * n
        z = np.asarray(cfg["waveforms"][pulse["waveforms"]["I"]]["samples"]) + 1j * np.asarray(
            cfg["waveforms"][pulse["waveforms"]["Q"]]["samples"])
        for j, slot in enumerate(subslots):
            assert np.allclose(z[j * n:(j + 1) * n], 0.0 if slot is None else shapes[slot[0]])


# --------------------------------------------------------------------------- victim tone


def test_subslot_phase_step_is_the_wrapped_leak_advance():
    mach, q1, q2 = _mock_machine()
    # (5.191 - 5.143) GHz x 16 ns = 0.768 turns -> -0.232
    assert np.isclose(subslot_phase_step(q1, q2), 0.768 - 1.0)
    assert np.isclose(subslot_phase_step(q2, q1), 1.0 - 0.768)


def test_victim_tone_cancel_keeps_the_victim_frequency():
    mach, q1, q2 = _mock_machine()
    q2.xy.RF_frequency = 5.6e9  # an aggressor-tone IF would be out of band
    cfg = _patch_cancel_element(_full_config(), q1, q2, tone="victim")
    assert cfg["elements"]["q1.xy_cancel"]["intermediate_frequency"] == 243e6
    assert cfg["elements"]["q1.xy_cancel"]["operations"]["x90"] == "q2.xy.x90.pulse"
    with pytest.raises(ValueError, match="tone"):
        _patch_cancel_element(_full_config(), q1, q2, tone="drive")


def test_victim_tone_composites_carry_the_in_clifford_frame_steps():
    """Subslot j of a victim-tone cancel composite = aggressor pulse x e^{j 2 pi j s}."""
    mach, q1, q2 = _mock_machine()
    agg = patch_config(_full_config(), q1, q2, bidirectional=True)
    vic = patch_config(_full_config(), q1, q2, bidirectional=True, cancel_tone="victim")

    def iq(cfg, elem, g):
        pulse = cfg["pulses"][cfg["elements"][elem]["operations"][f"clif{g}"]]
        return np.asarray(cfg["waveforms"][pulse["waveforms"]["I"]]["samples"]) + 1j * np.asarray(
            cfg["waveforms"][pulse["waveforms"]["Q"]]["samples"])

    g = 22  # x90, y90, x90: all three subslots played
    for victim, aggressor in ((q1, q2), (q2, q1)):
        elem, s = f"{victim.xy.name}_cancel", subslot_phase_step(victim, aggressor)
        a, v = iq(agg, elem, g), iq(vic, elem, g)
        for j in range(3):
            assert np.allclose(v[4 * j:4 * (j + 1)], a[4 * j:4 * (j + 1)] * np.exp(2j * np.pi * j * s))
        # the qubits' own composites are untouched
        assert np.allclose(iq(agg, victim.xy.name, g), iq(vic, victim.xy.name, g))


def test_victim_tone_cancel_loops_turn_the_frame_every_clifford():
    sc = _script(_program(conditions=["compensated"], bidirectional=True, drive_cancel_amp=0.02,
                          cancel_tone="victim")[0])
    mach, q1, q2 = _mock_machine()
    for victim, aggressor in ((q1, q2), (q2, q1)):
        step = 3 * subslot_phase_step(victim, aggressor)
        hits = re.findall(rf'frame_rotation_2pi\(([-0-9.e]+), "{victim.xy.name}_cancel"\)', sc)
        assert any(np.isclose(float(h), step) for h in hits)
    for elem in ("q1.xy", "q2.xy"):
        assert not re.search(rf'frame_rotation_2pi\([^)]*"{re.escape(elem)}"\)', sc)
    # the aggressor tone has only the initial phases
    sc = _script(_program(conditions=["compensated"], bidirectional=True, drive_cancel_amp=0.02)[0])
    assert len(re.findall(r'frame_rotation_2pi\([^)]*"q[12]\.xy_cancel"\)', sc)) == 2
    with pytest.raises(ValueError, match="cancel_tone"):
        _program(cancel_tone="drive")


# --------------------------------------------------------------------------- experiment


def _experiment(**params):
    from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRBParameters

    mach, q1, q2 = _mock_machine()
    exp = QMCrosstalkCompensatedSQRB(
        SimpleNamespace(machine=mach, device=None),
        CrosstalkCompensatedSQRBParameters(probe_qubit="q1", drive_qubit="q2", **params),
    )
    return exp, mach, q1, q2


def test_patch_preview_config_adds_cancels_and_composites():
    exp, mach, q1, q2 = _experiment(bidirectional=True)
    exp._select_pair = lambda: (q1, q2)
    cfg = exp.patch_preview_config(_full_config())
    assert {"q1.xy_cancel", "q2.xy_cancel"} <= set(cfg["elements"])
    assert "clif0" in cfg["elements"]["q1.xy"]["operations"]


def test_acquire_benchmark_runs_each_condition_in_short_programs(monkeypatch):
    import scqo_qm.experiments.crosstalk_compensated_sqrb as mod

    calls = []

    def mock_acquire(m, p, s, **kw):
        dims = [len(s[k].values) for k in ("qubit", "condition", "sequence_idx", "depth")]
        calls.append((list(s["condition"].values), list(s["sequence_idx"].values)))
        return xr.Dataset(
            data_vars={"I": (("qubit", "condition", "sequence_idx", "depth"), np.zeros(dims))},
            coords={k: s[k].values for k in ("qubit", "condition", "sequence_idx", "depth")},
        )

    from contextlib import contextmanager

    opened = []

    @contextmanager
    def fake_shared_qm(machine, config=None, *, timeout):
        opened.append(object())
        yield opened[-1]

    qms = []
    monkeypatch.setattr(mod, "shared_qm", fake_shared_qm)
    monkeypatch.setattr(mod, "_acquire", mock_acquire)
    monkeypatch.setattr(
        mod, "_acquire_chunk_with_retry", lambda m, p, s, **kw: (qms.append(kw.get("qm")), mock_acquire(m, p, s, **kw))[1]
    )
    exp, mach, q1, q2 = _experiment(bidirectional=True, depths=[1, 64, 1024], num_random_sequences=30)
    monkeypatch.setattr(QMCrosstalkCompensatedSQRB, "_program_kwargs", lambda self: dict(bidirectional=True))
    res = exp._acquire_benchmark(mach, num_shots=1, probe_qubit=q1, drive_qubit=q2)
    assert len(calls) == 3 * 5  # 3 conditions x ceil(30 / 7) programs
    # one QM for the whole benchmark, handed to every program
    assert len(opened) == 1 and qms == [opened[0]] * len(calls)
    assert all(len(seqs) <= 7 for _, seqs in calls)
    assert dict(res.sizes) == {"qubit": 2, "condition": 3, "sequence_idx": 30, "depth": 3}
    assert list(res.coords["condition"].values) == ["isolated", "simultaneous", "compensated"]
    assert list(res.coords["sequence_idx"].values) == list(range(30))


def test_chunk_retry_on_transient_qop_error(monkeypatch):
    import scqo_qm.experiments.crosstalk_compensated_sqrb as mod

    calls = []

    def flaky(m, p, s, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("FailedToExecuteJobException: Internal error. Compilation failed.")
        return "ok"

    monkeypatch.setattr(mod, "_acquire", flaky)
    assert mod._acquire_chunk_with_retry(None, None, None, pause_s=0) == "ok"
    assert len(calls) == 2

    def broken(m, p, s, **kw):
        raise ValueError("a real bug")

    monkeypatch.setattr(mod, "_acquire", broken)
    with pytest.raises(ValueError):
        mod._acquire_chunk_with_retry(None, None, None, pause_s=0)
