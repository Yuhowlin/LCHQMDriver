"""The mixed fixed/tunable QUAM root (scqo_qm.quam_builder ... mixed_quam).

The vendor roots force a device to be ALL fixed-frequency or ALL flux-tunable:
``FluxTunableQuam``'s flux helpers assume every qubit has a flux line, and
``set_all_fluxes`` -- reached from ``initialize_qpu``, which every probe calls --
does an unguarded ``target.z.settle()``. ``MixedTransmonQuam`` guards every flux
access through ``flux_line()`` so the three shapes (all-fixed, all-tunable, mixed)
all work.

Stub components, no qm/quam machinery: the point is which attributes get touched,
which is exactly what breaks. ``align()`` and the flux-line calls are recorded
rather than executed, since the real ones emit QUA and need a program() scope.
"""

from types import SimpleNamespace

import pytest

from scqo_qm.quam_builder.architecture.superconducting.qpu.mixed_quam import (
    MixedTransmonQuam,
    flux_line,
)


class _FluxLine:
    """Records which flux operations were asked for."""

    def __init__(self):
        self.calls = []
        self.joint_offset = 0.11
        self.independent_offset = 0.22

    def __getattr__(self, name):
        if name.startswith("to_") or name == "settle":
            return lambda *a, **k: self.calls.append(name)
        raise AttributeError(name)


def _tunable(name):
    """A flux-tunable qubit: has ``z``."""
    return SimpleNamespace(name=name, z=_FluxLine(), align=lambda: None)


def _fixed(name):
    """A fixed-frequency qubit: NO ``z`` attribute at all, which is how
    FixedFrequencyTransmon differs from its FluxTunableTransmon subclass."""
    return SimpleNamespace(name=name, align=lambda: None)


def _machine(**qubits):
    m = MixedTransmonQuam()
    m.qubits = dict(qubits)
    m.active_qubit_names = list(qubits)
    return m


# ------------------------------------------------------------------- the predicate
def test_flux_line_reads_absent_and_none_the_same_way():
    """Two ways to have no flux line, one meaning: nothing to bias."""
    assert flux_line(_fixed("q1")) is None            # no attribute at all
    assert flux_line(SimpleNamespace(z=None)) is None  # tunable, unwired
    assert flux_line(_tunable("q2")) is not None
    assert flux_line(SimpleNamespace()) is None        # a PAIR never has one


# ------------------------------------------------------------------- the mixed case
def test_set_all_fluxes_does_not_touch_a_fixed_qubit():
    """The regression this class exists for: unguarded target.z.settle() raises
    AttributeError on a fixed-frequency qubit, and every probe reaches it."""
    fixed, tunable = _fixed("q1"), _tunable("q2")
    m = _machine(q1=fixed, q2=tunable)

    bias = m.set_all_fluxes("joint", fixed)  # must not raise

    assert bias is None, "a qubit with no flux line has no bias to report"
    # the tunable neighbour is still biased -- skipping is per qubit, not global
    assert "to_joint_idle" in tunable.z.calls


def test_a_tunable_target_is_still_settled_alongside_a_fixed_qubit():
    fixed, tunable = _fixed("q1"), _tunable("q2")
    m = _machine(q1=fixed, q2=tunable)

    bias = m.set_all_fluxes("joint", tunable)

    assert bias == pytest.approx(0.11)  # z.joint_offset
    assert "settle" in tunable.z.calls


def test_all_fixed_device_is_a_clean_no_op():
    """chipA's shape. Every helper must survive a device with no flux at all."""
    m = _machine(q1=_fixed("q1"), q2=_fixed("q2"))
    m.set_all_fluxes("joint", m.qubits["q1"])
    m.apply_all_flux_to_min()
    m.apply_all_flux_to_joint_idle()
    m.apply_all_flux_to_zero()  # unguarded in the vendor class


def test_all_tunable_device_biases_everything():
    """5Q4C's shape -- the vendor behaviour must be unchanged."""
    a, b = _tunable("q1"), _tunable("q2")
    m = _machine(q1=a, q2=b)
    m.apply_all_flux_to_min()
    assert "to_min" in a.z.calls and "to_min" in b.z.calls


def test_inactive_tunable_qubits_go_to_min_not_joint():
    a, b = _tunable("q1"), _tunable("q2")
    m = _machine(q1=a, q2=b)
    m.active_qubit_names = ["q1"]
    m.apply_all_flux_to_joint_idle()
    assert "to_joint_idle" in a.z.calls
    assert b.z.calls == ["to_min"]


# ------------------------------------------------------------------- refusals
def test_independent_flux_point_refuses_a_fixed_qubit_by_name():
    """Skipping is right for 'set the fluxes'; it is NOT right for a caller who
    explicitly asked for a flux-specific operating point."""
    fixed = _fixed("q1")
    m = _machine(q1=fixed, q2=_tunable("q2"))
    with pytest.raises(TypeError, match="independent"):
        m.set_all_fluxes("independent", fixed)


def test_pairwise_flux_point_refuses_a_qubit():
    m = _machine(q1=_tunable("q1"))
    with pytest.raises(TypeError, match="pairwise"):
        m.set_all_fluxes("pairwise", m.qubits["q1"])


def test_refusals_raise_rather_than_assert():
    """The vendor used bare asserts, which vanish under python -O. These are real
    caller errors and must survive optimisation."""
    import scqo_qm.quam_builder.architecture.superconducting.qpu.mixed_quam as mod
    import inspect

    src = inspect.getsource(mod.MixedTransmonQuam.set_all_fluxes)
    assert "assert " not in src, "a refusal regressed to assert"
    assert src.count("raise TypeError") == 2
