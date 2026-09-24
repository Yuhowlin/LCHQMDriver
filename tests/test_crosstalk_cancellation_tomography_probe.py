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


