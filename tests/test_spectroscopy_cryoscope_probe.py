"""``qubit_spectroscopy_cryoscope`` probe helpers — the long-time cryoscope's gates.

The probe builds a single-qubit spectroscopy sequence that parks an idle-relative
flux pulse and drives at each wait-time into it. Two pure helpers are checked here
(no QOP, no config): ``validate_inputs`` (more than one target, a missing flux
line, and a parked flux whose ``idle + excursion`` clips the port or needs an
``amplitude_scale`` QUA cannot express — a legal call returns the ``const``
reference amplitude the volts->scale conversion divides by) and the RUN-SCOPED
spectroscopy tone: ``drive_amp_for_area`` (the area-preserving absolute amplitude —
a longer pulse is a proportionally weaker one), ``make_drive_pulse`` (the three
envelopes and what each one's average/peak ratio is), and ``check_drive_amp``
(a tone louder than the loudest already on the xy line refused by name).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.experiments.qubit_spectroscopy_cryoscope import (
    DRIVE_OP,
    _mean_per_amplitude,
    check_drive_amp,
    drive_amp_for_area,
    drop_drive_op,
    make_drive_pulse,
    validate_inputs,
)


class _Qubits(list):
    """Minimal stand-in for the probe's BatchableList (len / index / names)."""

    def get_names(self):
        return [q.name for q in self]


def _z(const_amp: float = 0.2, *, output_mode: str = "direct",
       joint_offset: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(
        name="q0_z",
        opx_output=SimpleNamespace(output_mode=output_mode),
        operations={"const": SimpleNamespace(amplitude=const_amp)},
        joint_offset=joint_offset,
    )


def _xy(*, x180_amp: float = 0.2, x180_len: float = 16.0,
        sat_amp: float = 0.4, sat_len: float = 1000.0) -> SimpleNamespace:
    return SimpleNamespace(operations={
        "x180": SimpleNamespace(amplitude=x180_amp, length=x180_len),
        "saturation": SimpleNamespace(amplitude=sat_amp, length=sat_len),
    })


def _qubit(name: str = "q0", *, xy: SimpleNamespace | None = None,
           **z_kwargs) -> SimpleNamespace:
    return SimpleNamespace(name=name, z=_z(**z_kwargs), xy=xy or _xy())


def test_legal_call_returns_the_const_reference():
    amp_ref = validate_inputs(_Qubits([_qubit()]), flux_amp_v=0.1, flux_point="joint")
    assert amp_ref == pytest.approx(0.2)  # the stored const amplitude


def test_more_than_one_target_refused_by_name():
    with pytest.raises(ValueError, match="one target at a time"):
        validate_inputs(_Qubits([_qubit("q0"), _qubit("q1")]), flux_amp_v=0.1, flux_point="joint")


def test_missing_flux_line_refused():
    with pytest.raises(ValueError, match="no flux line"):
        validate_inputs(_Qubits([SimpleNamespace(name="q0", z=None)]), flux_amp_v=0.1, flux_point="joint")


def test_idle_plus_excursion_over_rail_refused():
    # direct-mode rail is 0.5 V; 0.4 V idle + 0.2 V excursion = 0.6 V clips.
    with pytest.raises(ValueError, match="full scale"):
        validate_inputs(_Qubits([_qubit(joint_offset=0.4)]), flux_amp_v=0.2, flux_point="joint")


def test_amplitude_scale_out_of_range_refused():
    # const 0.2 V, excursion 0.5 V -> scale 2.5 >= QUA's +/-2 range (rail fine at 0.5).
    with pytest.raises(ValueError, match="amplitude_scale"):
        validate_inputs(_Qubits([_qubit(const_amp=0.2)]), flux_amp_v=0.5, flux_point="joint")


def test_drive_amp_holds_the_x180_rotation_area():
    """The absolute amplitude is the x180 rotation area spread over the tone: a
    longer pulse is a proportionally weaker one, and drive_amp_factor multiplies."""
    x180_area = 0.5 * 0.2 * 16  # cosine average/peak x amplitude x length
    amp = drive_amp_for_area(x180_area, 1.0, 400)  # square envelope
    assert amp == pytest.approx(x180_area / 400)
    assert drive_amp_for_area(x180_area, 1.0, 800) == pytest.approx(amp / 2)
    assert drive_amp_for_area(x180_area, 1.0, 400, 1.5) == pytest.approx(amp * 1.5)
    # a half-area envelope needs twice the peak to deliver the same rotation
    assert drive_amp_for_area(x180_area, 0.5, 400) == pytest.approx(amp * 2)


def test_drive_amp_reproduces_the_stretched_saturation_amplitude():
    """`square` is the same PHYSICS as the retired stretched-`saturation` path (the
    stored amplitude times the area-preserving scale), now expressed absolutely."""
    x180_area, sat_amp, length = 0.5 * 0.2 * 16, 0.4, 400
    legacy_scale = x180_area / (sat_amp * length)  # what resolve_drive_scale returned
    assert drive_amp_for_area(x180_area, 1.0, length) == pytest.approx(
        sat_amp * legacy_scale)


@pytest.mark.parametrize("shape, expected_ratio", [
    ("square", 1.0),        # constant waveform
    ("cosine", 0.5),        # raised cosine (Hann), mean over a whole period
    ("gaussian", 0.4622),   # sigma = length/4, subtracted
])
def test_make_drive_pulse_envelopes(shape, expected_ratio):
    """Each shape renders at the requested length, and its mean envelope per unit
    NOMINAL amplitude is the factor the amplitude arithmetic divides by."""
    pulse = make_drive_pulse(shape, 400)
    assert pulse.length == 400
    assert _mean_per_amplitude(pulse, default=-1.0) == pytest.approx(
        expected_ratio, abs=0.005)


def test_subtracted_gaussian_is_measured_against_its_NOMINAL_amplitude():
    """The regression this guards: a subtracted gaussian's rendered PEAK is only
    amplitude x (1 - exp(-2)) at sigma = length/4, but ``amplitude`` is what gets
    stored — measuring the envelope against the peak (``_area_ratio``) instead of
    the nominal amplitude leaves the tone 13.5% under a pi pulse, silently costing
    contrast. Pin that the built pulse delivers the requested area exactly."""
    pulse = make_drive_pulse("gaussian", 400, 0.25)
    samples = np.real(np.asarray(pulse.waveform_function()))
    assert samples.max() == pytest.approx(1 - np.exp(-2.0), abs=0.01)  # < nominal 1.0

    x180_area = 0.5 * 0.2 * 16
    ratio = _mean_per_amplitude(pulse, default=-1.0)
    pulse.amplitude = drive_amp_for_area(x180_area, ratio, 400)
    rendered = np.real(np.asarray(pulse.waveform_function()))
    assert rendered.sum() == pytest.approx(x180_area, rel=1e-3)


@pytest.mark.parametrize("shape", ["square", "cosine", "gaussian"])
def test_every_shape_delivers_the_same_rotation_area(shape):
    """Shape changes the LINESHAPE, never the rotation: all three tones integrate
    to the same x180 area at the same length, so switching drive_shape cannot
    silently turn the spectroscopy pulse into a sub- or over-rotation."""
    x180_area = 0.5 * 0.2 * 16
    pulse = make_drive_pulse(shape, 400)
    pulse.amplitude = drive_amp_for_area(
        x180_area, _mean_per_amplitude(pulse, default=-1.0), 400)
    rendered = np.real(np.asarray(pulse.waveform_function()))
    area = float(rendered.sum()) if rendered.ndim else float(rendered) * 400
    assert area == pytest.approx(x180_area, rel=1e-3)


def test_make_drive_pulse_gaussian_sigma_frac_narrows_the_envelope():
    """A narrower sigma packs less area under the same nominal amplitude, so the
    tone needs a proportionally larger amplitude to stay a pi pulse."""
    wide = _mean_per_amplitude(make_drive_pulse("gaussian", 400, 0.25), default=-1.0)
    narrow = _mean_per_amplitude(make_drive_pulse("gaussian", 400, 0.10), default=-1.0)
    assert narrow < wide


def test_unknown_drive_shape_refused_by_name():
    with pytest.raises(ValueError, match="drive_shape"):
        make_drive_pulse("lorentzian", 400)


def test_drive_amp_above_the_loudest_stored_tone_refused_by_name():
    """Clipping is silent on the QM simulator, so a tone louder than anything the
    xy line already stores is refused before instrument time — naming the knobs."""
    xy = _xy(x180_amp=0.2, sat_amp=0.4)
    check_drive_amp(0.4, xy, name="q0 drive")  # exactly the ceiling is fine
    with pytest.raises(ValueError, match="drive_amp_factor"):
        check_drive_amp(0.5, xy, name="q0 drive")


def test_drop_drive_op_is_idempotent_and_survives_a_missing_line():
    """run()'s finally must never raise: dropping a tone that was never installed,
    or one on a qubit with no xy line, is a no-op."""
    q = _qubit()
    q.xy.operations[DRIVE_OP] = object()
    drop_drive_op(q)
    assert DRIVE_OP not in q.xy.operations
    drop_drive_op(q)  # already gone
    drop_drive_op(SimpleNamespace(name="q0", xy=None))
