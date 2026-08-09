"""QM continuous charge-parity monitor for scqo - supplies only ``probe()``.

Parameters, the REFUSE gates (a stored ``parity_delta_f_hz`` for the fixed
idle, a governed depletion wait for the shot cadence, stored blob centers for
the trace discrimination), the telegraph-PSD fit and the ``parity_rate_hz``
writeback are all inherited from
``scqo.experiments.QubitParitySwitchContinuous``. This class only converts the
resolved ns values into QUA clock cycles and reports the scheduled shot period
back to the neutral layer.

PER-SHOT contract: every shot is recorded individually — the probe's streams
are ``buffer(num_shots)`` with NO ``.average()``.

NOT MULTIPLEXED, on purpose. Each qubit gets its own batch, so its shot cadence
is its own; in a multiplexed batch the ``align()``s tie every qubit's period to
the slowest member, and that period is exactly what each qubit's switching rate
is divided by.
"""

from __future__ import annotations

from typing import Any

from scqo import register
from scqo.experiments import QubitParitySwitchContinuous
from scqo.experiments._depletion import depletion_wait_ns

#: QUA plays waits on a 4 ns clock with a 16 ns (4-cycle) floor.
_MIN_CYCLES = 4


def _cycles(ns: float) -> int:
    """ns -> QUA clock cycles, floored at the 16 ns minimum wait."""
    return max(_MIN_CYCLES, int(round(float(ns) / 4.0)))


@register
class QMQubitParitySwitchContinuous(QubitParitySwitchContinuous):
    """Build a per-shot parity-monitor QUA program on the QM OPX."""

    def probe(self) -> Any:
        from scqo_qm.experiments._lib import select_qubits
        from scqo_qm.probes import qubit_parity_switch_continuous as parity_probe

        machine = self.backend.machine  # type: ignore[attr-defined]
        # one qubit per batch: independent shot cadences (see the module docstring)
        qubits = select_qubits(machine, self.params.targets, multiplexed=False)

        idle_cycles: dict[str, int] = {}
        depletion_cycles: dict[str, int] = {}
        self.probe_shot_period_s: dict[str, float] = {}
        for target in self.params.targets:
            idle_cycles[target] = _cycles(self.resolved_idle_ns(target))
            # THE precedence point (per-run override else the standing knob,
            # which binds to QUAM's resonator.depletion_time); never None —
            # define_sweep refused a target without a governed value. 0 is
            # legal and means no wait, so it must survive the floor.
            depletion_ns = float(depletion_wait_ns(self, target))
            depletion_cycles[target] = 0 if depletion_ns <= 0 else _cycles(depletion_ns)
            self.probe_shot_period_s[target] = self._shot_period_s(
                machine, target, idle_cycles[target], depletion_cycles[target])

        return parity_probe.build_program(
            machine,
            qubits,
            idle_cycles=idle_cycles,
            depletion_cycles=depletion_cycles,
            # resolved by the neutral layer from record_time_s (or an explicit
            # override); params.num_shots is None in the normal case
            num_shots=self.resolved_num_shots(),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )

    @staticmethod
    def _shot_period_s(machine: Any, target: str, idle_cycles: int,
                       depletion_cycles: int) -> float:
        """The scheduled shot-to-shot period from the durations the program
        above actually plays: depletion + y90 + idle + x90 + readout.

        This is the telegraph timebase — the neutral layer converts a per-shot
        switching probability into Hz with it — so it is summed from the same
        numbers passed to ``build_program`` rather than re-derived from knobs.
        The QUA ``align()``s between the blocks add tens of ns of sequencer
        overhead that is not counted here; on a multi-microsecond period that
        is well under a percent, and it is constant, so it biases the rate by
        the same fraction rather than distorting the spectrum.
        """
        qubit = machine.qubits[target]
        pi2 = qubit.xy.operations["y90"].length + qubit.xy.operations["x90"].length
        readout = qubit.resonator.operations["readout"].length
        return ((idle_cycles + depletion_cycles) * 4.0 + float(pi2) + float(readout)) * 1e-9
