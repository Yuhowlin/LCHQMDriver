"""``qubit_spectroscopy_cryoscope`` probe helpers — the long-time cryoscope's gates.

The probe builds a single-qubit spectroscopy sequence that parks an idle-relative
flux pulse and drives at each wait-time into it. Two pure helpers are checked here
(no QOP, no config): ``validate_inputs`` (more than one target, a missing flux
line, and a parked flux whose ``idle + excursion`` clips the port or needs an
``amplitude_scale`` QUA cannot express — a legal call returns the ``const``
reference amplitude the volts->scale conversion divides by) and
``resolve_drive_scale`` (the area-preserving spectroscopy-drive amplitude: a
longer pulse is a proportionally weaker one, and a scale QUA's ``amp()`` cannot
express is refused by name).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.probes.qubit_spectroscopy_cryoscope import (
    resolve_drive_scale,
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


def test_drive_scale_holds_the_x180_rotation_area():
    # sample-less stubs fall back to the lab envelopes (cosine x180 -> 0.5,
    # square drive -> 1.0): x180 rotation area = 0.5 * 0.2 * 16; saturation amp
    # 0.4 over 400 ns -> 1.6 / 160 = 0.01.
    q = _qubit()
    scale = resolve_drive_scale(q, operation="saturation", operation_len_ns=400)
    assert scale == pytest.approx(0.5 * 0.2 * 16 / (0.4 * 400))
    # a longer pulse is a proportionally weaker one (area held constant).
    longer = resolve_drive_scale(q, operation="saturation", operation_len_ns=800)
    assert longer == pytest.approx(scale / 2)
    # operation_amp is a plain extra multiplier on top.
    boosted = resolve_drive_scale(q, operation="saturation", operation_len_ns=400,
                                  operation_amp=1.5)
    assert boosted == pytest.approx(scale * 1.5)


def test_drive_scale_samples_the_real_envelopes():
    """Ops that can render their waveform are weighted by the SAMPLED envelope
    (average/peak of the I quadrature), not the fallback: a cosine-envelope DRAG
    x180 delivers half its peak x length — miss that factor and the played tone is
    an exact 2*pi pulse whose spectroscopy response splits into two lobes (seen on
    hardware, 2026-08-03)."""
    t = np.arange(16.0)
    cos_env = 0.2 * (1 - np.cos(2 * np.pi * t / 16)) / 2  # DragCosine I quadrature
    x180 = SimpleNamespace(
        amplitude=0.2, length=16.0,
        # complex I + iQ, pinning the real-part-only reduction (Q integrates ~0)
        waveform_function=lambda: cos_env + 0.05j * np.gradient(cos_env))
    sat = SimpleNamespace(amplitude=0.4, length=1000.0,
                          waveform_function=lambda: 0.4)  # SquarePulse: a scalar
    q = SimpleNamespace(name="q0", xy=SimpleNamespace(
        operations={"x180": x180, "saturation": sat}))
    scale = resolve_drive_scale(q, operation="saturation", operation_len_ns=400)
    # cosine mean/peak over a whole period is exactly 0.5; square is 1.0
    assert scale == pytest.approx(0.5 * 0.2 * 16 / (0.4 * 400))


def test_drive_scale_out_of_range_refused_by_name():
    # a pathological config (tiny stored saturation amplitude) drives the
    # area-preserving scale past QUA's +/-2 amp() range -> refused before building.
    q = _qubit(xy=_xy(x180_amp=0.5, x180_len=16, sat_amp=0.002))
    with pytest.raises(ValueError, match="amplitude_scale"):
        resolve_drive_scale(q, operation="saturation", operation_len_ns=16)
