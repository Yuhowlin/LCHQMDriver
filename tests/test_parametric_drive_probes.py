"""BOTH parametric-drive probes' own decisions.

``qubit_parametric_drive_amp`` sweeps (amplitude, frequency) at a fixed driving
time; ``qubit_parametric_drive_time`` sweeps (frequency, driving time) at a fixed
amplitude. Their shared half lives in ``experiments/_parametric.py``.

Covers the pure guards (the 4 ns drive-time grid for one time and for a whole
axis, the generated-config oscillator patch, the stub-level flux refusals) and —
on the live ``quam_state`` — that both programs build, the axes come out
canonical in the right ORDER, the time probe stretches its z pulse with a
real-time QUA variable, and each ``probe()``'s 3-tuple binds the
oscillator-patched config into its acquire callable (the reason these shells use
the 3-tuple shape at all: the shared fetch path regenerates a config whose z
elements carry no oscillator).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.experiments._parametric import (
    drive_time_cycles,
    drive_time_cycles_array,
    ensure_flux_oscillators,
)
from scqo_qm.experiments.qubit_parametric_drive_amp import build_program
from scqo_qm.experiments.qubit_parametric_drive_time import (
    build_program as build_time_program,
)


# ------------------------------------------------------------------ pure guards

def test_drive_time_grid():
    assert drive_time_cycles(16) == 4
    assert drive_time_cycles(2000) == 500


@pytest.mark.parametrize("bad", [0, 8, 12, 15, 18, 2001, -4])
def test_drive_time_off_grid_refused_by_name(bad):
    with pytest.raises(ValueError, match="drive_time_ns"):
        drive_time_cycles(bad)


def test_drive_time_axis_to_cycles():
    axis = np.array([16.0, 44.0, 2000.0])
    assert drive_time_cycles_array(axis).tolist() == [4, 11, 500]


def test_drive_time_axis_refuses_one_off_grid_point_by_name():
    """scqo builds this axis at grid_ns=4, so an off-grid point means the axis
    was tampered with — refuse the whole run rather than truncate that point."""
    with pytest.raises(ValueError, match="drive_time_ns"):
        drive_time_cycles_array(np.array([16.0, 45.0, 2000.0]))


def test_drive_time_axis_refuses_an_empty_sweep():
    with pytest.raises(ValueError, match="empty"):
        drive_time_cycles_array(np.array([]))


def test_drive_time_axis_keeps_duplicates_rather_than_shortening_the_sweep():
    """No de-duplication on purpose: an exact-grid axis has no rounding
    collisions to absorb, and silently shortening a sweep axis would
    desynchronize it from the stream buffers built from len(cycles)."""
    assert drive_time_cycles_array(np.array([16.0, 16.0])).tolist() == [4, 4]


def test_oscillator_patch_seeds_missing_and_respects_declared():
    config = {"elements": {
        "q1.z": {},                                    # no oscillator -> seeded
        "q2.z": {"intermediate_frequency": 12e6},      # declared -> untouched
    }}
    out = ensure_flux_oscillators(config, ["q1.z", "q2.z"], 50e6)
    assert out is config  # in-place, the same dict the acquire callable carries
    assert config["elements"]["q1.z"]["intermediate_frequency"] == 50e6
    assert config["elements"]["q2.z"]["intermediate_frequency"] == 12e6


def test_oscillator_patch_refuses_an_unknown_element():
    with pytest.raises(ValueError, match="q9.z"):
        ensure_flux_oscillators({"elements": {}}, ["q9.z"], 50e6)


def _stub_qubit(name="q1", *, z="default"):
    if z == "default":
        z = SimpleNamespace(
            name=f"{name}.z",
            operations={"const": SimpleNamespace(amplitude=0.25)},
            opx_output=SimpleNamespace(output_mode="direct"),
            flux_point="zero",
        )
    return SimpleNamespace(name=name, z=z)


def test_build_refuses_a_qubit_without_a_z_line():
    with pytest.raises(ValueError, match="no z line"):
        build_program(
            None, [_stub_qubit(z=None)],
            amps_v=np.linspace(0.0, 0.1, 5), freqs_hz=np.linspace(50e6, 150e6, 5),
            drive_time_ns=2000, num_shots=10, reset_type="thermal",
            use_state_discrimination=True)


def test_build_refuses_a_window_past_the_rail():
    # direct-mode rail is 0.5 V; a 0.6 V excursion would clip silently.
    with pytest.raises(ValueError, match="full scale"):
        build_program(
            None, [_stub_qubit()],
            amps_v=np.linspace(0.0, 0.6, 5), freqs_hz=np.linspace(50e6, 150e6, 5),
            drive_time_ns=2000, num_shots=10, reset_type="thermal",
            use_state_discrimination=True)


def test_build_refuses_an_off_grid_drive_time_before_any_qua():
    with pytest.raises(ValueError, match="drive_time_ns"):
        build_program(
            None, [_stub_qubit()],
            amps_v=np.linspace(0.0, 0.1, 5), freqs_hz=np.linspace(50e6, 150e6, 5),
            drive_time_ns=1999, num_shots=10, reset_type="thermal",
            use_state_discrimination=True)


def test_time_build_refuses_a_qubit_without_a_z_line():
    with pytest.raises(ValueError, match="no z line"):
        build_time_program(
            None, [_stub_qubit(z=None)],
            amp_v=0.05, freqs_hz=np.linspace(50e6, 150e6, 5),
            times_ns=np.arange(16, 100, 4), num_shots=10, reset_type="thermal",
            use_state_discrimination=True)


def test_time_build_refuses_an_amplitude_past_the_rail():
    # direct-mode rail is 0.5 V; a 0.6 V excursion would clip silently. The
    # amplitude is FIXED here, but it goes through the same one-element guard.
    with pytest.raises(ValueError, match="full scale"):
        build_time_program(
            None, [_stub_qubit()],
            amp_v=0.6, freqs_hz=np.linspace(50e6, 150e6, 5),
            times_ns=np.arange(16, 100, 4), num_shots=10, reset_type="thermal",
            use_state_discrimination=True)


def test_time_build_refuses_an_off_grid_time_axis_before_any_qua():
    with pytest.raises(ValueError, match="drive_time_ns"):
        build_time_program(
            None, [_stub_qubit()],
            amp_v=0.05, freqs_hz=np.linspace(50e6, 150e6, 5),
            times_ns=np.array([16.0, 17.0]), num_shots=10, reset_type="thermal",
            use_state_discrimination=True)


# ------------------------------------------------------------------ requires QUAM

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(
        str(Path(__file__).resolve().parents[1] / "quam_state"))


AMPS = np.linspace(0.0, 0.1, 6)
FREQS = np.linspace(50e6, 150e6, 11)


@pytest.mark.parametrize("discriminate", [True, False], ids=["state", "iq"])
def test_program_builds_on_live_state_with_canonical_axes(machine, discriminate):
    from scqo_qm.experiments._lib import select_qubits

    qubits = select_qubits(machine, ["q1"], multiplexed=True)
    prog, axes = build_program(
        machine, qubits,
        amps_v=AMPS, freqs_hz=FREQS.astype(int), drive_time_ns=2000,
        num_shots=10, reset_type="thermal",
        use_state_discrimination=discriminate)
    assert list(axes) == ["qubit", "parametric_amp_v", "parametric_freq_hz"]
    assert axes["parametric_amp_v"].attrs["units"] == "V"
    assert axes["parametric_freq_hz"].attrs["units"] == "Hz"

    # the program must serialize against the PATCHED config (the z oscillator
    # the update_frequency needs), and the patch must land on the z element
    from qm import generate_qua_script

    z_name = machine.qubits["q1"].z.name
    config = ensure_flux_oscillators(
        machine.generate_config(), [z_name], float(FREQS[0]))
    assert config["elements"][z_name]["intermediate_frequency"] == float(FREQS[0])
    script = generate_qua_script(prog, config)
    assert "update_frequency" in script


def test_preview_compiles_against_the_patched_config(machine, tmp_path, monkeypatch):
    """backend.preview must hand the SAME oscillator-amended config to the
    script dump (and the gateway simulation) that a real run executes against —
    the live gateway refuses the unpatched one by name ('Can not change the
    intermediate frequency of quantum Element q1.z because its' initial value
    was none', observed 2026-08-20)."""
    import qm

    from scqo.roster import parse_components
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments.qubit_parametric_drive_amp import QMQubitParametricDriveAmp

    captured = {}
    real = qm.generate_qua_script

    def capture(prog, config):
        captured["config"] = config
        return real(prog, config)

    monkeypatch.setattr(qm, "generate_qua_script", capture)

    backend = QMBackend(machine, roster=parse_components(roster_toml_for(machine)))
    exp = QMQubitParametricDriveAmp(
        backend, QMQubitParametricDriveAmp.Parameters(
            targets=["q1"], start_parametric_amp_v=0.0, end_parametric_amp_v=0.1,
            start_parametric_freq_hz=50e6, end_parametric_freq_hz=150e6,
            num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    backend.preview(exp, tmp_path, no_simulate=True)
    z_name = machine.qubits["q1"].z.name
    assert captured["config"]["elements"][z_name]["intermediate_frequency"] == 50e6
    assert (tmp_path / "qua_script.py").exists()


def test_probe_binds_the_patched_config_into_its_acquire(machine):
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo.roster import parse_components
    from scqo_qm.experiments.qubit_parametric_drive_amp import QMQubitParametricDriveAmp

    backend = QMBackend(machine, roster=parse_components(roster_toml_for(machine)))
    exp = QMQubitParametricDriveAmp(
        backend, QMQubitParametricDriveAmp.Parameters(
            targets=["q1"], start_parametric_amp_v=0.0, end_parametric_amp_v=0.1,
            start_parametric_freq_hz=50e6, end_parametric_freq_hz=150e6,
            num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    res = exp.probe()
    assert isinstance(res, tuple) and len(res) == 3
    prog, axes, acquire_fn = res
    # the canonical frequency axis was re-declared as the played integer grid
    played = axes["parametric_freq_hz"].values
    assert np.array_equal(exp.sweep_axes["parametric_freq_hz"], played.astype(float))
    # the acquire callable carries the oscillator-patched config
    config = acquire_fn.keywords["config"]
    z_name = machine.qubits["q1"].z.name
    assert config["elements"][z_name]["intermediate_frequency"] == float(played[0])


TIMES = np.arange(16.0, 16.0 + 8 * 28.0, 28.0)  # 8 points, on the 4 ns grid


@pytest.mark.parametrize("discriminate", [True, False], ids=["state", "iq"])
def test_time_program_builds_on_live_state_with_canonical_axes(machine, discriminate):
    """Frequency OUTER, time INNER — the order is the contract's, and it is what
    lets update_frequency run once per frequency instead of once per point."""
    from scqo_qm.experiments._lib import select_qubits

    qubits = select_qubits(machine, ["q1"], multiplexed=True)
    prog, axes = build_time_program(
        machine, qubits,
        amp_v=0.05, freqs_hz=FREQS.astype(int), times_ns=TIMES,
        num_shots=10, reset_type="thermal",
        use_state_discrimination=discriminate)
    assert list(axes) == ["qubit", "parametric_freq_hz", "drive_time_ns"]
    assert axes["parametric_freq_hz"].attrs["units"] == "Hz"
    assert axes["drive_time_ns"].attrs["units"] == "ns"
    # the axis is stored in NANOSECONDS, exactly as handed in (the 4 ns grid is
    # what makes the ns -> cycles conversion lossless, so nothing re-declares it)
    assert np.array_equal(axes["drive_time_ns"].values, TIMES)

    from qm import generate_qua_script

    z_name = machine.qubits["q1"].z.name
    config = ensure_flux_oscillators(
        machine.generate_config(), [z_name], float(FREQS[0]))
    script = generate_qua_script(prog, config)
    assert "update_frequency" in script


def test_time_program_stretches_the_pulse_with_a_qua_variable(machine):
    """The duration must be a real-time QUA variable, not a python int baked per
    point: one program covers the whole map. A regression here would show up as
    a literal cycle count in the play() call."""
    from qm import generate_qua_script

    from scqo_qm.experiments._lib import select_qubits

    qubits = select_qubits(machine, ["q1"], multiplexed=True)
    prog, _ = build_time_program(
        machine, qubits,
        amp_v=0.05, freqs_hz=FREQS.astype(int), times_ns=TIMES,
        num_shots=10, reset_type="thermal", use_state_discrimination=True)
    z_name = machine.qubits["q1"].z.name
    config = ensure_flux_oscillators(
        machine.generate_config(), [z_name], float(FREQS[0]))
    script = generate_qua_script(prog, config)
    plays = [ln for ln in script.splitlines()
             if "play(" in ln and z_name in ln and "duration=" in ln]
    assert plays, f"no z play() with a duration in the dumped script"
    # every cycle count on the axis is distinct, so a baked duration would have
    # to appear as one of these literals
    assert not any(f"duration={c}" in ln for ln in plays
                   for c in drive_time_cycles_array(TIMES)), (
        "the driving time was baked per point instead of swept in real time")


def test_time_probe_binds_the_patched_config_into_its_acquire(machine):
    from scqo.roster import parse_components
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments.qubit_parametric_drive_time import QMQubitParametricDriveTime

    backend = QMBackend(machine, roster=parse_components(roster_toml_for(machine)))
    exp = QMQubitParametricDriveTime(
        backend, QMQubitParametricDriveTime.Parameters(
            targets=["q1"], parametric_amp_v=0.05,
            start_parametric_freq_hz=50e6, end_parametric_freq_hz=150e6,
            num_freq_points=5, num_time_points=8, num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    times_before = np.array(exp.sweep_axes["drive_time_ns"], dtype=float)
    prog, axes, acquire_fn = exp.probe()
    # the frequency axis was re-declared as the played integer grid; the TIME
    # axis was not touched at all (it was already exact on the 4 ns grid)
    played = axes["parametric_freq_hz"].values
    assert np.array_equal(exp.sweep_axes["parametric_freq_hz"], played.astype(float))
    assert np.array_equal(exp.sweep_axes["drive_time_ns"], times_before)
    config = acquire_fn.keywords["config"]
    z_name = machine.qubits["q1"].z.name
    assert config["elements"][z_name]["intermediate_frequency"] == float(played[0])


def test_time_preview_compiles_against_the_patched_config(machine, tmp_path, monkeypatch):
    """patch_preview_config comes from the shared mixin, so the time shell gets
    the same gateway-verified amendment the amp shell has."""
    import qm

    from scqo.roster import parse_components
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments.qubit_parametric_drive_time import QMQubitParametricDriveTime

    captured = {}
    real = qm.generate_qua_script

    def capture(prog, config):
        captured["config"] = config
        return real(prog, config)

    monkeypatch.setattr(qm, "generate_qua_script", capture)

    backend = QMBackend(machine, roster=parse_components(roster_toml_for(machine)))
    exp = QMQubitParametricDriveTime(
        backend, QMQubitParametricDriveTime.Parameters(
            targets=["q1"], parametric_amp_v=0.05,
            start_parametric_freq_hz=50e6, end_parametric_freq_hz=150e6,
            num_freq_points=5, num_time_points=8, num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    backend.preview(exp, tmp_path, no_simulate=True)
    z_name = machine.qubits["q1"].z.name
    assert captured["config"]["elements"][z_name]["intermediate_frequency"] == 50e6
    assert (tmp_path / "qua_script.py").exists()


def test_a_reversed_window_still_plays_ascending_and_seeds_the_low_edge(machine):
    """The scqo window takes its edges in either order and normalises the axis
    ascending, so the DRIVER never sees a descending sweep. The oscillator patch
    is seeded from the first PLAYED frequency, which is therefore the LOW edge —
    the other tests pass their edges ascending and would not notice a regression
    that leaked the raw start_ value through."""
    from scqo.roster import parse_components
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo_qm.experiments.qubit_parametric_drive_time import QMQubitParametricDriveTime

    backend = QMBackend(machine, roster=parse_components(roster_toml_for(machine)))
    exp = QMQubitParametricDriveTime(
        backend, QMQubitParametricDriveTime.Parameters(
            targets=["q1"], parametric_amp_v=0.05,
            start_parametric_freq_hz=150e6, end_parametric_freq_hz=50e6,   # REVERSED
            start_drive_time_ns=500.0, end_drive_time_ns=16.0,             # REVERSED
            num_freq_points=5, num_time_points=8, num_averages=10))
    exp.sweep_axes = exp.define_sweep()
    _prog, axes, acquire_fn = exp.probe()

    played = axes["parametric_freq_hz"].values
    assert np.all(np.diff(played) > 0), "the driver was handed a descending axis"
    assert float(played[0]) == 50e6
    times = axes["drive_time_ns"].values
    assert np.all(np.diff(times) > 0)
    assert float(times[0]) == 16.0
    z_name = machine.qubits["q1"].z.name
    seed = acquire_fn.keywords["config"]["elements"][z_name]["intermediate_frequency"]
    assert seed == 50e6, "the oscillator patch must be seeded from the LOW edge"
