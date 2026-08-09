"""QM discrete charge-parity monitor for scqo - supplies only ``probe()``.

Parameters (including ``cycle_period_ns``), the REFUSE gates, the
within-cycle telegraph-PSD fit and the ``parity_rate_hz`` writeback are all
inherited from ``scqo.experiments.QubitParitySwitchDiscrete``. This class
converts the resolved ns values into QUA clock cycles, computes the pad wait
that stretches each cycle to exactly the requested period — refusing BY NAME
when ``cycle_period_ns`` is shorter than the scheduled sequence — and reports
the exact scheduled cycle period back to the neutral layer.

PER-MEASUREMENT contract: both measurements of every cycle are recorded
individually — the probe's streams are ``buffer(2).buffer(num_shots)`` with
NO ``.average()``.

NOT MULTIPLEXED, for the same reason as the continuous shell: each qubit's
cycle period is its own telegraph timebase.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitParitySwitchDiscrete
from scqo.experiments._depletion import depletion_wait_ns

from .qubit_parity_switch_continuous import _cycles

_PERIOD_TOO_SHORT = (
    "cycle_period_ns = {want:.0f} ns is shorter than the scheduled sequence "
    "on {target}: M1 + depletion + x90 + idle + y90 + M2 already takes "
    "{sequence:.0f} ns. Raise cycle_period_ns to at least that (or leave it "
    "None for the minimal, unpadded cycle)."
)


@register
class QMQubitParitySwitchDiscrete(QubitParitySwitchDiscrete):
    """Build a two-measurement-per-cycle parity-monitor QUA program on the QM
    OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_parity_switch_discrete as parity_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        # one qubit per batch: independent cycle periods (module docstring)
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)

        idle_cycles: dict[str, int] = {}
        depletion_cycles: dict[str, int] = {}
        tau_wait_cycles: dict[str, int] = {}
        self.probe_shot_period_s: dict[str, float] = {}
        for target in self.params.targets:
            idle_cycles[target] = _cycles(self.resolved_idle_ns(target))
            # THE precedence point; never None — define_sweep refused a target
            # without a governed value. 0 is legal and means no wait.
            depletion_ns = float(depletion_wait_ns(self, target))
            depletion_cycles[target] = 0 if depletion_ns <= 0 else _cycles(depletion_ns)
            sequence_ns = self._sequence_ns(
                machine, target, idle_cycles[target], depletion_cycles[target])
            tau_wait_cycles[target] = self._tau_wait_cycles(
                self.params.cycle_period_ns, sequence_ns, target)
            # the EXACT scheduled cycle period, after the pad's grid snap —
            # this is the telegraph timebase the rate is divided by.
            self.probe_shot_period_s[target] = (
                sequence_ns + tau_wait_cycles[target] * 4.0) * 1e-9

        return parity_probe.build_program(
            machine,
            qubits,
            idle_cycles=idle_cycles,
            depletion_cycles=depletion_cycles,
            tau_wait_cycles=tau_wait_cycles,
            num_shots=self.resolved_num_shots(),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )

    @staticmethod
    def _sequence_ns(machine: Any, target: str, idle_cycles: int,
                     depletion_cycles: int) -> float:
        """The scheduled per-cycle sequence duration WITHOUT the pad: M1 +
        depletion + x90 + idle + y90 + M2, from the same numbers passed to
        ``build_program``. The QUA ``align()``s add tens of ns of sequencer
        overhead not counted here — constant, so it biases the rate by a
        sub-percent fraction rather than distorting the spectrum."""
        qubit = machine.qubits[target]
        pi2 = qubit.xy.operations["x90"].length + qubit.xy.operations["y90"].length
        readout = qubit.resonator.operations["readout"].length
        return ((idle_cycles + depletion_cycles) * 4.0
                + float(pi2) + 2.0 * float(readout))

    @staticmethod
    def _tau_wait_cycles(cycle_period_ns: float | None, sequence_ns: float,
                         target: str) -> int:
        """The pad that stretches the cycle to ``cycle_period_ns``, in 4 ns
        cycles. None or an exact fit pads nothing; a period SHORTER than the
        sequence is refused by name; a positive sub-16 ns remainder snaps UP
        to the 16 ns floor (the reported period reflects the snap)."""
        if cycle_period_ns is None:
            return 0
        remainder = float(cycle_period_ns) - float(sequence_ns)
        if remainder < 0:
            raise ValueError(_PERIOD_TOO_SHORT.format(
                want=float(cycle_period_ns), sequence=float(sequence_ns),
                target=target))
        if remainder == 0:
            return 0
        return _cycles(remainder)
