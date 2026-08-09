"""The flux-headroom invariant: a port must be able to EMIT what the tree declares.

The per-sweep helpers in ``probes/_flux_limits.py`` refuse ONE probe when it asks
for more volts than its port can put out. This is the other half — a whole-tree
audit run once by the scqo backend factory, so a config that cannot work is
reported completely and up front ("these three ports need amplified mode")
instead of one probe dying on the first one it happens to touch.

Every failure it catches is silent on hardware: the DAC clips, the fit still
converges, and the QM simulator shows nothing.

This is also the ONE place the ``const`` = rail/2 convention is enforced. The
per-sweep helpers deliberately do not, because an undersized ``const`` does not
clip anything — it only caps reach — and enforcing it there would refuse configs
that genuinely work today (the live 5Q4C couplers sit at 0.15 V).
"""

from scqo_qm.quam_fields import flux_headroom_problems, flux_headroom_warnings

from conftest import _coupler, _flux_line, make_stub_machine


def _z(amplitude, *, mode="direct", **offsets):
    z = _flux_line(**offsets)
    z.opx_output = type("Port", (), {"output_mode": mode})()
    z.operations = {"const": type("Pulse", (), {"amplitude": amplitude})()}
    return z


def _machine_with(z):
    machine = make_stub_machine()
    machine.qubits["q1"].z = z
    return machine


def test_the_stub_tree_is_compliant():
    """Guard for every other case here: the shared fixture must start clean, or a
    later assertion could pass on the wrong problem."""
    assert flux_headroom_problems(make_stub_machine()) == []
    assert flux_headroom_warnings(make_stub_machine()) == []


def test_the_two_severities_are_split_by_whether_the_DAC_LIES():
    """The load-bearing distinction. An over-rail op makes the hardware emit
    something OTHER than what was asked, silently — fatal. An undersized const
    emits exactly what was asked, there is just less range available — advisory.
    Conflating them would block every session on the live 5Q4C couplers, which
    sit at 0.15 V and have always run."""
    clipping = _machine_with(_z(0.6))
    assert len(flux_headroom_problems(clipping)) == 1
    assert flux_headroom_warnings(clipping) == []

    undersized = _machine_with(_z(0.15))
    assert flux_headroom_problems(undersized) == []
    assert len(flux_headroom_warnings(undersized)) == 1


# ------------------------------------------------------------- stored op peaks

def test_a_stored_op_past_the_rail_is_named_with_the_remedy():
    problems = flux_headroom_problems(_machine_with(_z(0.6)))
    assert len(problems) == 1
    assert "qubits.q1.z.operations['const']" in problems[0]
    assert "0.6" in problems[0] and "0.5" in problems[0]
    assert 'output_mode="amplified"' in problems[0]  # the actual fix


def test_the_SAME_amplitude_is_fine_on_an_amplified_port():
    """The whole point of reading the rail per port: 1.25 V is the correct
    amplified-mode const and would be refused outright by a hardcoded 0.5."""
    assert flux_headroom_problems(_machine_with(_z(1.25, mode="amplified"))) == []


def test_a_non_numeric_amplitude_is_skipped_not_guessed():
    """QUAM references ('#./const') and unset fields are legitimate; inventing a
    verdict for them would make the audit cry wolf on every real state file."""
    z = _z(0.25)
    z.operations["cardinal"] = type("Pulse", (), {"amplitude": "#./const"})()
    assert flux_headroom_problems(_machine_with(z)) == []


# ------------------------------------------------------------- the convention

def test_an_undersized_const_is_reported_with_the_reach_it_costs():
    """The live 5Q4C couplers' 0.15 V. It clips nothing — it just means no sweep
    on that line can pass +/-0.3 V — so it is an advisory here and not raised by
    the per-sweep helpers."""
    advisories = flux_headroom_warnings(_machine_with(_z(0.15)))
    assert len(advisories) == 1
    assert "0.15" in advisories[0]
    assert "0.25" in advisories[0]        # the value to set
    assert "0.3" in advisories[0]         # the reach it actually costs
    assert "Nothing clips" in advisories[0]


def test_the_convention_follows_the_port_mode():
    """rail/2 is 0.25 on direct and 1.25 on amplified — the same stored number is
    correct on one and undersized on the other."""
    assert flux_headroom_warnings(_machine_with(_z(0.25))) == []
    assert flux_headroom_warnings(_machine_with(_z(1.25, mode="amplified"))) == []
    advisories = flux_headroom_warnings(_machine_with(_z(0.25, mode="amplified")))
    assert len(advisories) == 1 and "1.25" in advisories[0]


def test_an_over_rail_const_is_reported_once_not_twice():
    """It breaks the convention AND clips. The clipping is the actionable one;
    also emitting a reach advisory would bury it."""
    machine = _machine_with(_z(0.6))
    problems = flux_headroom_problems(machine)
    assert len(problems) == 1
    assert "full scale" in problems[0]
    assert flux_headroom_warnings(machine) == []


# ----------------------------------------------------------- standing offsets

def test_an_idle_offset_past_the_rail_is_reported():
    """The bias alone already clips, before any pulse rides on it."""
    problems = flux_headroom_problems(_machine_with(_z(0.25, joint=0.7)))
    assert len(problems) == 1
    assert "joint_offset" in problems[0] and "0.7" in problems[0]


def test_every_named_point_is_audited_not_just_the_declared_one():
    """A parked-at-min run uses min_offset; auditing only the declared point
    would pass a tree that clips the moment someone selects another."""
    z = _z(0.25)
    z.min_offset = -0.9
    problems = flux_headroom_problems(_machine_with(z))
    assert len(problems) == 1 and "min_offset" in problems[0]


# ----------------------------------------------------------------- couplers

def test_couplers_are_audited_with_their_OWN_offset_names():
    """A TunableCoupler carries decouple_offset/interaction_offset, not
    joint_offset — auditing only the qubit vocabulary would skip every coupler."""
    machine = make_stub_machine()
    coupler = _coupler("c12")
    coupler.opx_output = type("Port", (), {"output_mode": "direct"})()
    coupler.operations = {"const": type("Pulse", (), {"amplitude": 0.25})()}
    coupler.interaction_offset = 0.8
    machine.qubit_pairs["coupler_q1_q2"].coupler = coupler

    problems = flux_headroom_problems(machine)
    assert len(problems) == 1
    assert "coupler" in problems[0] and "interaction_offset" in problems[0]


# ------------------------------------------------------- reporting completely

def test_every_bad_port_is_reported_in_one_pass():
    """The reason this exists as an audit rather than only as per-probe refusals:
    an operator fixing an amplified-mode migration wants the whole list, not the
    first port a probe happened to touch."""
    machine = make_stub_machine()
    machine.qubits["q1"].z = _z(0.6)
    machine.qubits["q2"].z = _z(0.25, joint=0.8)
    problems = flux_headroom_problems(machine)
    assert len(problems) == 2
    assert any("q1" in p for p in problems) and any("q2" in p for p in problems)


def test_the_live_5Q4C_config_is_not_blocked_by_the_audit():
    """Regression for the severity split. The shipped 5Q4C state has three
    couplers at const 0.15 V; if those were fatal, every `scqo run` on the QM
    backend would SystemExit at session construction."""
    machine = make_stub_machine()
    for pair_name in machine.qubit_pairs:
        coupler = _coupler(pair_name)
        coupler.opx_output = type("Port", (), {"output_mode": "direct"})()
        coupler.operations = {"const": type("Pulse", (), {"amplitude": 0.15})()}
        machine.qubit_pairs[pair_name].coupler = coupler
    assert flux_headroom_problems(machine) == []
    assert flux_headroom_warnings(machine) != []
