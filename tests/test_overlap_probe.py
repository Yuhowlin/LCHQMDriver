"""``qubit_spectroscopy_overlap``: the drive and the readout tone really do overlap.

On this backend "overlap" is the ABSENCE of something — the ``align()`` that sits
between the drive and the measurement in ``qubit_spectroscopy.py``. A missing
barrier leaves no trace anywhere except in the emitted QUA, and re-adding one
would silently turn the experiment back into its sequential sibling while every
fit still converged. So this reads the generated QUA program and asserts the
shape directly, with the sequential probe built alongside as the contrast.

Live-QUAM: the pre-tone plays a REAL ``readout`` operation through
``Channel.play``, and whether a ``BaseReadoutPulse`` can be played without being
measured is a property of the actual tree, not of a stub.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

quam_config = pytest.importorskip("quam_config")
pytest.importorskip("qm")

from conftest import recording_device  # noqa: E402
from test_qm_backend import roster_toml_for  # noqa: E402

from scqo_qm.backend.qm_backend import QMBackend  # noqa: E402
from scqo.roster import parse_components  # noqa: E402

TARGET = "q4"


@pytest.fixture(scope="module")
def machine():
    return quam_config.Quam.load(str(Path(__file__).resolve().parents[1] / "quam_state"))


@pytest.fixture(scope="module")
def live_roster(machine):
    return parse_components(roster_toml_for(machine))


@pytest.fixture(scope="module")
def config(machine):
    return machine.generate_config()


def _body(prog, config) -> list[str]:
    """The stripped lines of one sweep point's QUA, drive onward.

    Everything before the shared ``align()`` is setup (flux offsets, the reset
    wait); everything from there to the closing ``align()`` is the concurrent
    block this file is about.
    """
    from qm import generate_qua_script

    lines = [ln.strip() for ln in generate_qua_script(prog, config).splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.startswith("update_frequency")]
    assert starts, "expected the detuning update that opens each sweep point"
    tail = lines[starts[0]:]
    stop = next(i for i, ln in enumerate(tail) if ln.startswith("save("))
    return tail[:stop]


def _build(machine, live_roster, name, **params):
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401  (registers the QM probes)

    backend = QMBackend(machine, roster=live_roster)
    cls = get(name)
    kwargs = {k: v for k, v in
              dict(num_freq_points=5, num_averages=10, **params).items()
              if k in cls.Parameters.model_fields}
    exp = cls(backend, cls.Parameters(targets=[TARGET], **kwargs))
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    prog, _axes = exp.probe()
    return exp, prog


def test_no_align_between_the_drive_and_the_measurement(machine, live_roster, config):
    """THE claim, and its contrast. In the sequential probe an ``align()`` stands
    between the saturation play and the measure — a hard barrier that makes the
    drive finish first. Here there is none, so both elements run from the one
    shared align above them and the tones are concurrent."""
    _e, overlap = _build(machine, live_roster, "qubit_spectroscopy_overlap")
    _s, sequential = _build(machine, live_roster, "qubit_spectroscopy")

    for prog, expect_barrier in ((sequential, True), (overlap, False)):
        body = _body(prog, config)
        drive = next(i for i, ln in enumerate(body) if ln.replace(chr(34), chr(39)).startswith("play('saturation'"))
        measure = next(i for i, ln in enumerate(body) if ln.replace(chr(34), chr(39)).startswith("measure('readout'"))
        assert drive < measure
        between = [ln for ln in body[drive + 1:measure] if ln.startswith("align(")]
        assert bool(between) is expect_barrier, body[drive:measure + 1]


def test_the_adc_lead_is_a_pre_tone_on_the_resonator(machine, live_roster, config):
    """QUAM's ``measure()`` has no acquisition-delay argument, so the lead is the
    same readout operation played back-to-back into it — one seamless tone whose
    tail is what gets integrated. Durations are QUA clock cycles."""
    _e, prog = _build(machine, live_roster, "qubit_spectroscopy_overlap",
                      acq_start_ns=400.0, drive_len_ns=600.0)
    body = _body(prog, config)

    pre = [ln for ln in body if ln.replace(chr(34), chr(39)).startswith("play('readout', '")]
    assert len(pre) == 1, f"expected one readout pre-tone, got {pre}"
    assert re.search(r"duration=(\d+)", pre[0]).group(1) == "100"  # 400 ns / 4
    # ...and it is played BEFORE the measurement, on the same element
    assert body.index(pre[0]) < next(
        i for i, ln in enumerate(body) if ln.replace(chr(34), chr(39)).startswith("measure('readout'"))

    drive = next(ln for ln in body if ln.replace(chr(34), chr(39)).startswith("play('saturation'"))
    assert re.search(r"duration=(\d+)", drive).group(1) == "150"  # 600 ns / 4


def test_zero_lead_emits_no_pre_tone_and_the_drive_spans_the_tone(
        machine, live_roster, config):
    """The default. acq_start_ns=0 is today's timing (the ADC opens with the
    readout pulse), and drive_len_ns=None means the drive covers the whole tone
    — here the readout operation's own length, since there is no lead."""
    exp, prog = _build(machine, live_roster, "qubit_spectroscopy_overlap")
    body = _body(prog, config)

    assert not [ln for ln in body if ln.replace(chr(34), chr(39)).startswith("play('readout', '")]
    readout_ns = exp.device.channel(TARGET, "readout").readout_duration_s * 1e9
    drive = next(ln for ln in body if ln.replace(chr(34), chr(39)).startswith("play('saturation'"))
    assert int(re.search(r"duration=(\d+)", drive).group(1)) == round(readout_ns / 4)


def test_targets_with_different_readout_windows_are_refused(machine, live_roster):
    """One multiplexed program plays ONE set of times. Taking the first target's
    silently would put every other target's ADC in the wrong place — a weaker,
    shifted peak on exactly the qubits nobody was watching."""
    from scqo.experiments import get

    import scqo_qm.experiments  # noqa: F401

    backend = QMBackend(machine, roster=live_roster)
    cls = get("qubit_spectroscopy_overlap")
    others = [q for q in ("q4", "q5") if q in live_roster.entities]
    if len(others) < 2:
        pytest.skip("needs two qubits on the live tree")

    exp = cls(backend, cls.Parameters(targets=others, num_freq_points=5, num_averages=10))
    exp.device = recording_device(backend, live_roster)
    exp.sweep_axes = exp.define_sweep()
    view = exp.device.channel(others[1], "readout")
    before = view.readout_duration_s
    view.readout_duration_s = before + 400e-9
    try:
        with pytest.raises(ValueError, match="different concurrent-tone windows"):
            exp.probe()
    finally:
        view.readout_duration_s = before
