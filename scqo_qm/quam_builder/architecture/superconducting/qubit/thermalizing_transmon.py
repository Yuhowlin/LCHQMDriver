"""Transmons whose thermal-reset wait is an ABSOLUTE stored value.

QUAM's ``BaseTransmon.thermalization_time`` is a derived read-only property —
``thermalization_time_factor * T1`` (falling back to ``factor * 10 us`` when T1
is unknown). That makes the wait unsettable: there is nowhere to put an absolute
number, and the factor is an int, so "wait 200 us" is only expressible when it
happens to be an integer multiple of the current T1.

scqo needs an absolute per-qubit wait it can own: ``thermalization_time_s`` is a
neutral KNOB on the drive channel (scqo_state.json, pushed to the vendor), and
``qubit_relaxation`` proposes ``thermalization_factor x T1`` for it. These
subclasses give that number a real home, ``thermalization_time_ns``, and
override the property to prefer it.

Everything downstream is untouched: ``qubit.reset("thermal")`` still calls
``reset_qubit_thermal`` which still waits ``thermalization_time // 4`` cycles,
so every probe (scqo shells AND the qualibrate nodes) honours the calibrated
value with no edit. When nothing has been set the property falls through to
QUAM's derived behaviour, so a qubit that scqo has never touched behaves exactly
as before.

Selecting these classes is a per-qubit decision in the device's ``state.json``
(each qubit serializes its own ``__class__``) — NOT the ``qubit_type`` binding
in ``quam_config/my_quam.py``, which is the lab's charge-tunable toggle and is
governed separately.
"""

from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.qubit.fixed_frequency_transmon import (
    FixedFrequencyTransmon,
)
from quam_builder.architecture.superconducting.qubit.flux_tunable_transmon import (
    FluxTunableTransmon,
)

__all__ = ["ThermalizingFixedFrequencyTransmon", "ThermalizingFluxTunableTransmon"]

#: QUA `wait` counts 4 ns clock cycles, so the realized wait is always a
#: multiple of 4 ns. Rounding (rather than refusing off-grid values, as
#: pi_duration_s does) is safe here: this is a policy wait, not a calibrated
#: pulse length, so a sub-cycle difference changes nothing measurable.
_CYCLE_NS = 4


def _quantized(thermalization_time_ns) -> int:
    return int(thermalization_time_ns / _CYCLE_NS) * _CYCLE_NS


@quam_dataclass
class ThermalizingFixedFrequencyTransmon(FixedFrequencyTransmon):
    """Fixed-frequency transmon with an absolute, scqo-owned reset wait (ns)."""

    thermalization_time_ns: float = None

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns — the stored absolute value
        when scqo has calibrated one, else QUAM's derived factor x T1."""
        if self.thermalization_time_ns is not None:
            return _quantized(self.thermalization_time_ns)
        return super().thermalization_time


@quam_dataclass
class ThermalizingFluxTunableTransmon(FluxTunableTransmon):
    """Flux-tunable transmon with an absolute, scqo-owned reset wait (ns)."""

    thermalization_time_ns: float = None

    @property
    def thermalization_time(self):
        """The transmon thermalization time in ns — the stored absolute value
        when scqo has calibrated one, else QUAM's derived factor x T1."""
        if self.thermalization_time_ns is not None:
            return _quantized(self.thermalization_time_ns)
        return super().thermalization_time
