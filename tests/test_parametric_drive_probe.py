"""The parametric-drive probe's own decisions.

Covers the pure guards (the 4 ns drive-time grid, the generated-config
oscillator patch, the stub-level flux refusals) and — on the live
``quam_state`` — that the program builds, the axes come out canonical, and
``probe()``'s 3-tuple binds the oscillator-patched config into its acquire
callable (the reason this shell uses the 3-tuple shape at all: the shared
fetch path regenerates a config whose z elements carry no oscillator).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.experiments.qubit_parametric_drive import (
    build_program,
    drive_time_cycles,
    ensure_flux_oscillators,
)


# ------------------------------------------------------------------ pure guards

def test_drive_time_grid():
    assert drive_time_cycles(16) == 4
    assert drive_time_cycles(2000) == 500


@pytest.mark.parametrize("bad", [0, 8, 12, 15, 18, 2001, -4])
def test_drive_time_off_grid_refused_by_name(bad):
    with pytest.raises(ValueError, match="drive_time_ns"):
        drive_time_cycles(bad)


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


def test_probe_binds_the_patched_config_into_its_acquire(machine):
    from scqo_qm.backend.qm_backend import QMBackend
    from scqo_qm.backend.roster_gen import roster_toml_for
    from scqo.roster import parse_components
    from scqo_qm.experiments.qubit_parametric_drive import QMQubitParametricDrive

    backend = QMBackend(machine, roster=parse_components(roster_toml_for(machine)))
    exp = QMQubitParametricDrive(
        backend, QMQubitParametricDrive.Parameters(
            targets=["q1"], min_parametric_amp_v=0.0, max_parametric_amp_v=0.1,
            min_parametric_freq_hz=50e6, max_parametric_freq_hz=150e6,
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
