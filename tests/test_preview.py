"""``QMBackend.preview``: the self-acquiring census + the refusal dispatch.

QUAM-free (conftest stub): preview's refusal for a ``probe_self_acquires``
shell must fire BEFORE ``probe()`` — the stub machine cannot build a QUA
program, so getting past the refusal would blow up, which is exactly the
ordering proof. The census keeps the declarations in lockstep with the shells
that really acquire inside ``probe()``: a new self-acquiring shell MUST
declare the attribute, or preview would reach the instrument before the
backend's defensive ready-Dataset check could fire. The live-state script
dump is ``test_qm_backend.py::test_preview_writes_qua_script`` (needs the
``machine`` fixture, so it does not live here).
"""

from __future__ import annotations

import pytest

from conftest import make_experiment

import customized.scqo.experiments  # noqa: F401  (import side effect: @register)
from customized.scqo.backend import SELF_ACQUIRING_ATTR
from scqo.experiments import catalog, get

#: the shells whose probe() executes on the instrument and returns a ready
#: Dataset — the ONLY legitimate carriers of the opt-out attribute
SELF_ACQUIRING = {
    "pair_swap_chevron",
    "qc_n_swap_amp",
    "qubit_drag_equator",
    "qubit_drag_alternating",
    "qubit_ramsey_cryoscope",
    "qubit_xyz_delay",
}


def _qm_shells() -> dict[str, type]:
    """Every registered experiment whose class comes from THIS driver."""
    return {entry["name"]: get(entry["name"]) for entry in catalog()
            if get(entry["name"]).__module__.startswith("customized.")}


def test_declarations_match_the_self_acquiring_set_exactly():
    shells = _qm_shells()
    assert set(SELF_ACQUIRING) <= set(shells), "census names a non-QM shell"
    declared = {name for name, cls in shells.items()
                if getattr(cls, SELF_ACQUIRING_ATTR, None)}
    assert declared == SELF_ACQUIRING  # both directions: no stray, no missing


def test_every_reason_is_a_nonempty_string():
    for name in sorted(SELF_ACQUIRING):
        reason = getattr(get(name), SELF_ACQUIRING_ATTR)
        assert isinstance(reason, str) and reason.strip(), name


def test_refusal_fires_before_probe_and_creates_nothing(backend, roster,
                                                        tmp_path):
    from customized.scqo.experiments.qubit_drag_alternating import (
        QMQubitDragAlternating,
    )

    exp = make_experiment(QMQubitDragAlternating, backend, roster,
                          QMQubitDragAlternating.Parameters(targets=["q1"]))
    out_dir = tmp_path / "prev"
    with pytest.raises(ValueError,
                       match="qubit_drag_alternating cannot be previewed"):
        backend.preview(exp, out_dir)
    assert not out_dir.exists()
