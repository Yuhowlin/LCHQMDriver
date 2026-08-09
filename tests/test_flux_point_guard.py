"""The flux-point invariant: the knob scqo governs must BE the bias that plays.

scqo's ``idle_flux`` reads and writes ``q.z.<flux_point>_offset``
(``quam_fields.get/set_idle_flux``), while every probe's ``initialize_qpu``
applies the ``"joint"`` point — ``joint_offset`` for qubits and
``decouple_offset`` for couplers. When a state file DECLARES a different point
the two silently address different numbers, and ``idle_flux`` becomes a knob the
hardware never sees.

That was live on 5Q4C until 2026-07-29: all five z lines declared
``"independent"`` while every run biased them at ``joint_offset``, so the arch
fits' AND the resonator flux map's accepted sweet spots were both inert. The
failure is silent by nature — the write succeeds, the state file changes, the
next run is simply unaffected — which is why it needs an explicit guard rather
than a convention.
"""

from types import SimpleNamespace

from scqo_qm.quam_fields import (
    GOVERNED_COUPLER_FLUX_POINT,
    GOVERNED_FLUX_POINT,
    flux_point_problems,
    get_idle_flux,
)

from conftest import _flux_line, make_stub_machine


def test_the_stub_tree_is_compliant():
    """Guard for every other case here: the shared fixture must start clean, or a
    later assertion could pass on the wrong problem."""
    assert flux_point_problems(make_stub_machine()) == []


def test_a_non_joint_qubit_is_named_with_both_offsets():
    """The 5Q4C bug, reproduced. The message must carry BOTH numbers: the
    operator's question is 'which value is live', and naming only the field
    leaves them to look it up."""
    machine = make_stub_machine()
    machine.qubits["q1"].z = _flux_line(joint=0.107, flux_point="independent")
    machine.qubits["q1"].z.independent_offset = 0.084
    machine.active_qubits = list(machine.qubits.values())

    problems = flux_point_problems(machine)
    assert len(problems) == 1
    assert "q1" in problems[0]
    assert "independent" in problems[0]
    assert "0.084" in problems[0]   # what idle_flux would read/write
    assert "0.107" in problems[0]   # what the hardware actually holds


def test_the_two_offsets_really_do_differ_under_the_bug():
    """Why the guard is worth having: with a non-joint declaration, the value
    scqo serves as idle_flux is NOT the one the joint path applies."""
    z = _flux_line(joint=0.107, flux_point="independent")
    z.independent_offset = 0.084
    qubit = SimpleNamespace(z=z)
    assert get_idle_flux(qubit) == 0.084      # what scqo reads and writes
    assert z.joint_offset == 0.107            # what initialize_qpu applies


def test_a_non_off_coupler_is_named():
    """The coupler leg has its OWN vocabulary; the joint path always decouples
    (``apply_all_couplers_to_min`` -> ``decouple_offset``), so anything but
    'off' addresses a different stored number."""
    machine = make_stub_machine()
    machine.qubit_pairs["coupler_q1_q2"].coupler.flux_point = "on"

    problems = flux_point_problems(machine)
    assert len(problems) == 1
    assert "coupler_q1_q2" in problems[0]
    assert GOVERNED_COUPLER_FLUX_POINT in problems[0]


def test_an_inactive_flux_tunable_qubit_is_named():
    """The other half of the same defect: ``apply_all_flux_to_joint_idle`` parks
    INACTIVE qubits at ``min_offset``, so a compliant flux_point is not enough —
    an inactive qubit's idle_flux still reports a bias it does not hold."""
    machine = make_stub_machine()
    machine.active_qubits = [machine.qubits["q1"], machine.qubits["q3"]]

    problems = flux_point_problems(machine)
    assert len(problems) == 1
    assert "q2" in problems[0]
    assert "active_qubit_names" in problems[0]


def test_a_fixed_frequency_qubit_is_not_flagged():
    """q3 has no z subtree at all. No flux line, nothing to govern — and a
    mostly-fixed chip must not drown the operator in irrelevant problems."""
    machine = make_stub_machine()
    assert not hasattr(machine.qubits["q3"], "z")
    assert flux_point_problems(machine) == []


def test_an_unknowable_active_set_is_not_an_accusation():
    """A tree that cannot answer 'which qubits are active' must SKIP that clause,
    not flag every qubit. Treating a missing attribute as an empty set would turn
    the guard into a wall of false alarms on any partial tree."""
    machine = make_stub_machine()
    del machine.active_qubits
    assert flux_point_problems(machine) == []


def test_every_problem_names_the_fix():
    """The guard aborts session construction, so each line has to be actionable
    on its own — the operator is looking at a stack trace, not at this test."""
    machine = make_stub_machine()
    machine.qubits["q1"].z = _flux_line(joint=0.1, flux_point="min")
    machine.qubit_pairs["coupler_q1_q2"].coupler.flux_point = "arbitrary"
    machine.active_qubits = list(machine.qubits.values())

    problems = flux_point_problems(machine)
    assert len(problems) == 2
    assert all("state.json" in p for p in problems)
    assert any(GOVERNED_FLUX_POINT in p for p in problems)
    assert any(GOVERNED_COUPLER_FLUX_POINT in p for p in problems)
