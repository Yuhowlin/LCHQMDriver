from quam.core import quam_dataclass
from quam.components.macro import QubitMacro


@quam_dataclass
class ParametricReset(QubitMacro):
    """Parametric reset on a qubit's own flux (z) line.

    Mirrors the technique in `LCH_qubit_parametric_drive_{fixed,freq}_time`: set the z-line
    IF to a calibrated sideband/drive frequency, reset the IF phase for coherence, then play
    a dedicated `parametric_reset` z pulse (its amplitude and length live in the pulse op).
    Bare-callable so it works as a named macro: `qubit.macros["reset"].apply()`.

    The dedicated `parametric_reset` operation must exist on the qubit's z line
    (`qubit.z.operations["parametric_reset"]`). Amplitude/length intentionally live in that
    pulse op, not on this macro (optional `amplitude_scale`/`duration` overrides could be
    added later without breaking the bare `.apply()` call).
    """

    drive_frequency: int  # parametric drive frequency in Hz (z-line IF)
    flux_pulse: str = "parametric_reset"  # z operation to play

    def apply(self):
        self.qubit.z.update_frequency(self.drive_frequency)
        self.qubit.z.reset_if_phase()
        self.qubit.z.play(self.flux_pulse)
