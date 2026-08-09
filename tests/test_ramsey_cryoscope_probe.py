"""``qubit_ramsey_cryoscope.validate_inputs`` — the ramsey cryoscope probe's pre-flight gate.

The probe builds a single-qubit phase-tomography sequence that sweeps EVERY
nanosecond and plays an idle-relative flux pulse. All the ways that can be wrong
are pure checks over stub qubits (no QOP, no config, no baking): more than one
target, a duration axis that is not ``arange(1, max+1)``, a missing flux line,
and a flux window whose ``idle + excursion`` clips the port or needs an
``amplitude_scale`` QUA cannot express. A legal call returns the ``const``
reference amplitude the volts->scale conversion divides by.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scqo_qm.probes.qubit_ramsey_cryoscope import validate_inputs


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


def _qubit(name: str = "q0", **z_kwargs) -> SimpleNamespace:
    return SimpleNamespace(name=name, z=_z(**z_kwargs))


def _durations(max_ns: int = 16) -> np.ndarray:
    return np.arange(1, max_ns + 1, dtype=float)


def test_legal_call_returns_the_const_reference():
    qubits = _Qubits([_qubit()])
    amp_ref = validate_inputs(qubits, _durations(240), flux_amp_v=0.1, flux_point="joint")
    assert amp_ref == pytest.approx(0.2)  # the stored const amplitude


def test_more_than_one_target_refused_by_name():
    qubits = _Qubits([_qubit("q0"), _qubit("q1")])
    with pytest.raises(ValueError, match="one target at a time"):
        validate_inputs(qubits, _durations(), flux_amp_v=0.1, flux_point="joint")


def test_non_arange_durations_refused():
    qubits = _Qubits([_qubit()])
    sparse = np.array([1.0, 2.0, 4.0, 8.0, 16.0])  # not every ns
    with pytest.raises(ValueError, match="EVERY nanosecond"):
        validate_inputs(qubits, sparse, flux_amp_v=0.1, flux_point="joint")


def test_missing_flux_line_refused():
    qubits = _Qubits([SimpleNamespace(name="q0", z=None)])
    with pytest.raises(ValueError, match="no flux line"):
        validate_inputs(qubits, _durations(), flux_amp_v=0.1, flux_point="joint")


def test_idle_plus_excursion_over_rail_refused():
    # direct-mode rail is 0.5 V; 0.4 V idle + 0.2 V excursion = 0.6 V clips.
    qubits = _Qubits([_qubit(joint_offset=0.4)])
    with pytest.raises(ValueError, match="full scale"):
        validate_inputs(qubits, _durations(), flux_amp_v=0.2, flux_point="joint")


def test_amplitude_scale_out_of_range_refused():
    # const 0.2 V, excursion 0.5 V -> scale 2.5 >= QUA's +/-2 range (rail is fine
    # at idle 0 + 0.5 = 0.5, exactly the rail).
    qubits = _Qubits([_qubit(const_amp=0.2)])
    with pytest.raises(ValueError, match="amplitude_scale"):
        validate_inputs(qubits, _durations(), flux_amp_v=0.5, flux_point="joint")
