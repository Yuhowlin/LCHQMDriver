"""The QM parity-monitor shells' pure decisions (no QUA, no instrument).

Covers BOTH variants — continuous (`qubit_parity_switch_continuous`) and
discrete (`qubit_parity_switch_discrete`). Everything here is arithmetic the
shells own and the probes trust:

* ns -> QUA clock cycles for the fixed idle (4 ns clock, 16 ns floor);
* the depletion wait's precedence AND the fact that 0 survives the floor —
  "measured, needs no settle" is a real answer, and rounding it up to 16 ns
  would silently lengthen every shot;
* the shot/cycle period, which is the telegraph timebase: the neutral layer
  divides a per-sample switching probability by it, so a period that disagrees
  with what the program plays scales every reported rate linearly;
* the discrete pad math: cycle_period_ns - sequence, refused by name when
  negative, 0 when None/exact, snapped UP through the 16 ns floor otherwise.

The absence of a qubit reset is a property of the QUA probe modules, checked by
reading them — a QUA program cannot be built without an instrument-grade
config, and a probe that silently regained a `qubit.reset()` would be
undetectable in the data.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from scqo_qm.experiments.qubit_parity_switch_continuous import _cycles

_PROBES_DIR = Path(__file__).resolve().parents[1] / "scqo_qm" / "experiments"
PROBE_CONTINUOUS = _PROBES_DIR / "qubit_parity_switch_continuous.py"
PROBE_DISCRETE = _PROBES_DIR / "qubit_parity_switch_discrete.py"

BOTH_PROBES = pytest.mark.parametrize(
    "probe", [PROBE_CONTINUOUS, PROBE_DISCRETE],
    ids=["continuous", "discrete"])


def _called_names(probe: Path) -> set[str]:
    """Every callee spelling in the probe, as source text.

    An AST walk, not a substring scan: the module docstrings DESCRIBE the calls
    that must not appear (``readout_state``, ``.average()``, ``qubit.reset``)
    and explain why, so a text search over the file finds its own
    documentation and reports the opposite of the truth.
    """
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    return {ast.unparse(node.func) for node in ast.walk(tree)
            if isinstance(node, ast.Call)}


def _stub_machine(*, y90=40, x90=40, readout=2000):
    qubit = SimpleNamespace(
        xy=SimpleNamespace(operations={
            "y90": SimpleNamespace(length=y90),
            "x90": SimpleNamespace(length=x90)}),
        resonator=SimpleNamespace(operations={
            "readout": SimpleNamespace(length=readout)}),
    )
    return SimpleNamespace(qubits={"q1": qubit})


class TestCycleConversion:

    def test_snaps_to_the_4ns_clock(self):
        assert _cycles(2000) == 500
        assert _cycles(2002) == 500     # 2002 -> 500.5 -> 500 cycles = 2000 ns
        assert _cycles(2006) == 502     # rounds up

    def test_floors_at_the_16ns_minimum_wait(self):
        assert _cycles(4) == 4          # 1 cycle requested, 4 is the floor
        assert _cycles(16) == 4
        assert _cycles(0) == 4


class TestShotPeriod:

    def test_sums_the_scheduled_durations(self):
        from scqo_qm.experiments.qubit_parity_switch_continuous import (
            QMQubitParitySwitchContinuous,
        )

        # 500 cycles idle (2000 ns) + 250 cycles depletion (1000 ns)
        # + 40 + 40 ns pi/2 + 2000 ns readout = 5080 ns
        period = QMQubitParitySwitchContinuous._shot_period_s(
            _stub_machine(), "q1", idle_cycles=500, depletion_cycles=250)
        assert period == pytest.approx(5080e-9)

    def test_zero_depletion_shortens_the_period(self):
        from scqo_qm.experiments.qubit_parity_switch_continuous import (
            QMQubitParitySwitchContinuous,
        )

        with_wait = QMQubitParitySwitchContinuous._shot_period_s(
            _stub_machine(), "q1", idle_cycles=500, depletion_cycles=250)
        without = QMQubitParitySwitchContinuous._shot_period_s(
            _stub_machine(), "q1", idle_cycles=500, depletion_cycles=0)
        assert with_wait - without == pytest.approx(1000e-9)


class TestDiscreteCyclePeriod:
    """The discrete shell's pad math: the cycle period IS the telegraph
    timebase, so the pad is computed to hit the request exactly and the
    reported period reflects any grid snap."""

    @staticmethod
    def _shell():
        from scqo_qm.experiments.qubit_parity_switch_discrete import (
            QMQubitParitySwitchDiscrete,
        )
        return QMQubitParitySwitchDiscrete

    def test_sequence_counts_two_readouts(self):
        # 500 cycles idle (2000 ns) + 250 cycles depletion (1000 ns)
        # + 40 + 40 ns pi/2 + 2 x 2000 ns readout = 7080 ns
        seq = self._shell()._sequence_ns(
            _stub_machine(), "q1", idle_cycles=500, depletion_cycles=250)
        assert seq == pytest.approx(7080.0)

    def test_none_means_no_pad(self):
        assert self._shell()._tau_wait_cycles(None, 7080.0, "q1") == 0

    def test_pad_hits_the_requested_period_exactly(self):
        # 10000 - 7080 = 2920 ns -> 730 cycles; 7080 + 730 * 4 = 10000 exactly
        cycles = self._shell()._tau_wait_cycles(10000.0, 7080.0, "q1")
        assert cycles == 730
        assert 7080.0 + cycles * 4.0 == pytest.approx(10000.0)

    def test_exact_fit_pads_nothing(self):
        assert self._shell()._tau_wait_cycles(7080.0, 7080.0, "q1") == 0

    def test_too_short_period_refused_by_name(self):
        with pytest.raises(ValueError, match="cycle_period_ns"):
            self._shell()._tau_wait_cycles(5000.0, 7080.0, "q1")

    def test_sub_16ns_remainder_snaps_up(self):
        """A positive remainder under the 16 ns wait floor cannot be played as
        asked; it snaps UP to one minimum wait and the reported period (which
        the shell computes AFTER the snap) says so."""
        cycles = self._shell()._tau_wait_cycles(7090.0, 7080.0, "q1")
        assert cycles == 4              # 10 ns remainder -> the 16 ns floor
        assert 7080.0 + cycles * 4.0 == pytest.approx(7096.0)


class TestProbeHasNoQubitReset:
    """The defining absence, for BOTH variants. Continuous: only the RESONATOR
    is waited out between shots (a reset would sever the XOR chain). Discrete:
    M1's projection IS the initialization."""

    @BOTH_PROBES
    def test_probe_never_calls_qubit_reset(self, probe):
        calls = _called_names(probe)
        assert not [c for c in calls if c.endswith(".reset")], calls
        # and takes no reset argument at all — there is nothing to pass
        tree = ast.parse(probe.read_text(encoding="utf-8"))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {a.arg for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef)
                  for a in [*n.args.args, *n.args.kwonlyargs]}
        assert "reset_type" not in names

    @BOTH_PROBES
    def test_probe_waits_the_resonator_between_shots(self, probe):
        assert "qubit.resonator.wait" in _called_names(probe)

    @BOTH_PROBES
    def test_probe_never_averages_the_shots(self, probe):
        """Every measurement is a sample of the telegraph; averaging destroys
        the signal. The legacy exclude/ node carried a `.average()` that was a
        no-op only because one buffer was ever produced."""
        calls = _called_names(probe)
        assert not [c for c in calls if c.endswith(".average")], calls
        assert any(c.endswith(".buffer") for c in calls), calls

    @BOTH_PROBES
    def test_probe_does_not_use_readout_state(self, probe):
        """`readout_state()` ends with its OWN depletion wait, which would
        double-count the governed wait and desynchronize the timebase from the
        period the shell reports (see the probes' docstrings)."""
        calls = _called_names(probe)
        assert not [c for c in calls if c.endswith("readout_state")], calls
        # ... and does the measurement itself instead
        assert "qubit.resonator.measure" in calls


def _played(probe: Path) -> list:
    tree = ast.parse(probe.read_text(encoding="utf-8"))
    return [node.args[0].value for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func).endswith("xy.play")
            and node.args and isinstance(node.args[0], ast.Constant)]


def test_continuous_probe_plays_y90_then_x90():
    """Order matters: the SECOND pulse is the 90-degree-shifted one, so the
    measured quantity is sin (odd in parity) rather than cos (even)."""
    assert _played(PROBE_CONTINUOUS) == ["y90", "x90"], _played(PROBE_CONTINUOUS)


def test_discrete_probe_plays_x90_then_y90():
    """The discrete variant follows the reference protocol's order (x90 first);
    still a 90-degree-shifted pair, so still sin — the swap only flips the
    telegraph's sign, which the PSD cannot see."""
    assert _played(PROBE_DISCRETE) == ["x90", "y90"], _played(PROBE_DISCRETE)


def test_discrete_probe_measures_twice_per_cycle_into_one_stream():
    """Both measurements ride the SAME streams as a buffered pair: the shared
    measure-and-save helper is invoked twice per loop iteration and the stream
    is buffered `(2)` innermost (then the cycle count), with meas_idx 0 = M1."""
    tree = ast.parse(PROBE_DISCRETE.read_text(encoding="utf-8"))
    helper_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "_measure_and_save"]
    assert len(helper_calls) == 2, "expected exactly two measurements per cycle"
    # the innermost buffer is the length-2 measurement pair
    pair_buffers = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and ast.unparse(n.func).endswith(".buffer")
                    and n.args and isinstance(n.args[0], ast.Constant)
                    and n.args[0].value == 2]
    assert pair_buffers, "no .buffer(2) found — the meas_idx axis is missing"
