"""Tests for CrosstalkCancellationTomography QUA probe generation."""

from types import SimpleNamespace
import numpy as np
import pytest

from scqo_qm.experiments.crosstalk_cancellation_tomography import (
    CrosstalkCancellationTomography,
    build_program,
)

from qm.qua import declare, declare_stream, fixed, play, amp


def _mock_machine():
    """Mock QUAM machine with probe and drive qubits."""
    def _decl():
        return [declare(fixed)], [declare_stream()], [declare(fixed)], [declare_stream()], declare(int), declare_stream()

    q1_xy = SimpleNamespace(
        name="q1.xy",
        operations={
            "x180": SimpleNamespace(length=16),
            "x90": SimpleNamespace(length=16),
            "-y90": SimpleNamespace(length=16),
        },
        wait=lambda *a, **kw: None,
        play=lambda op, amplitude_scale=None, **kw: play(
            op * amp(amplitude_scale) if amplitude_scale is not None else op, "q1.xy"
        ),
    )
    q2_xy = SimpleNamespace(
        name="q2.xy",
        operations={"x180": SimpleNamespace(length=16)},
        wait=lambda *a, **kw: None,
        play=lambda op, amplitude_scale=None, **kw: play(
            op * amp(amplitude_scale) if amplitude_scale is not None else op, "q2.xy"
        ),
    )

    q1_res = SimpleNamespace(name="q1_res", measure=lambda *a, **kw: None)
    q2_res = SimpleNamespace(name="q2_res", measure=lambda *a, **kw: None)

    q1 = SimpleNamespace(
        name="q1",
        xy=q1_xy,
        resonator=q1_res,
        reset=lambda *a, **kw: None,
        readout_state=lambda *a: None,
    )
    q2 = SimpleNamespace(
        name="q2",
        xy=q2_xy,
        resonator=q2_res,
        reset=lambda *a, **kw: None,
        readout_state=lambda *a: None,
    )

    mach = SimpleNamespace(
        declare_qua_variables=_decl,
        initialize_qpu=lambda **kw: None,
        qubits={"q1": q1, "q2": q2},
    )
    return mach, q1, q2


def test_build_program_structure():
    mach, q1, q2 = _mock_machine()

    prog, sweep_axes = build_program(
        mach,
        q1,
        q2,
        target_gate="x180",
        cancel_amp=0.035,
        init_phase=0.85,
        sub_phase_rate_turns=0.12,
        repetitions=[0, 1, 2, 5],
        conditions=["noise_only", "cancel_only", "compensated"],
        bases=["z", "x", "y"],
        num_shots=100,
        use_state_discrimination=True,
    )

    assert prog is not None
    assert "qubit" in sweep_axes
    assert "condition" in sweep_axes
    assert "repetitions" in sweep_axes
    assert "basis" in sweep_axes

    assert sweep_axes["qubit"].values.tolist() == ["q1"]
    assert sweep_axes["condition"].values.tolist() == ["noise_only", "cancel_only", "compensated"]
    assert sweep_axes["repetitions"].values.tolist() == [0, 1, 2, 5]
    assert sweep_axes["basis"].values.tolist() == ["z", "x", "y"]


def test_cancel_follows_the_drive_frame_with_a_global_phase_reset():
    import re
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    prog, _ = build_program(
        mach, q1, q2, target_gate="x180", cancel_amp=0.035, init_phase=0.0, sub_phase_rate_turns=0.12,
        repetitions=[3], conditions=["noise_only", "cancel_only", "compensated"], bases=["z"],
        num_shots=1, bidirectional=True,
    )
    sc = generate_qua_script(prog).replace("'", '"')

    def steps(elem):
        return [float(v) for v in re.findall(rf'frame_rotation_2pi\(([-0-9.e]+), "{re.escape(elem)}"\)', sc)]

    assert "reset_global_phase()" in sc
    # the drive and its copy step together every pulse (3 conditions x 3 pulses)
    assert steps("q1.xy_cancel").count(-0.12) == 9
    assert steps("q2.xy").count(-0.12) == 9
    # the idle reverse cancel element never plays, so it never steps
    assert steps("q2.xy_cancel") == []


def test_bases_share_one_pulse_train_without_a_branch():
    import re
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    prog, axes = build_program(
        mach, q1, q2, target_gate="x180", cancel_amp=0.035, init_phase=0.0, sub_phase_rate_turns=0.12,
        repetitions=[0, 2, 5], conditions=["compensated"], bases=["z", "x", "y"], num_shots=1,
        bidirectional=True,
    )
    sc = generate_qua_script(prog).replace("'", '"')
    # one pulse train per repetition count, whatever the number of bases
    assert len(re.findall(r'play\("x180", "q2\.xy"\)', sc)) == 2 + 5
    # the pre-rotation is amplitude-switched per basis: a play in a real-time branch hung the job
    assert "if_(" not in sc and "switch_(" not in sc
    assert "[0.0, 1.0, 0.0]" in sc and "[0.0, 0.0, 1.0]" in sc
    assert list(axes["basis"].values) == ["z", "x", "y"]
    with pytest.raises(ValueError, match="bases"):
        build_program(mach, q1, q2, repetitions=[1], conditions=["compensated"], bases=["w"], num_shots=1)


def test_removed_tone_and_reset_options_are_gone():
    mach, q1, q2 = _mock_machine()
    for kw in ({"phase_reset": "if_only"}, {"drive_init_phase": 0.1}):
        with pytest.raises(TypeError):
            build_program(mach, q1, q2, repetitions=[1], conditions=["compensated"], bases=["z"], num_shots=1, **kw)


def test_victim_tone_cancel_holds_still_while_the_drive_steps():
    import re
    from qm import generate_qua_script

    mach, q1, q2 = _mock_machine()
    prog, _ = build_program(
        mach, q1, q2, target_gate="x180", cancel_amp=0.035, init_phase=0.0, sub_phase_rate_turns=0.12,
        repetitions=[3], conditions=["noise_only", "cancel_only", "compensated"], bases=["z"],
        num_shots=1, bidirectional=True, cancel_tone="victim",
    )
    sc = generate_qua_script(prog).replace("'", '"')
    assert re.findall(r'frame_rotation_2pi\(([-0-9.e]+), "q2\.xy"\)', sc).count("-0.12") == 9
    assert -0.12 not in [float(v) for v in re.findall(r'frame_rotation_2pi\(([-0-9.e]+), "q1\.xy_cancel"\)', sc)]
    with pytest.raises(ValueError, match="cancel_tone"):
        build_program(mach, q1, q2, repetitions=[1], conditions=["compensated"], bases=["z"], num_shots=1,
                      cancel_tone="drive")
