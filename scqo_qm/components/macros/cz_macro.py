from quam.core import quam_dataclass
from quam.components.pulses import Pulse
# from quam.components.macro import QubitPairMacro
from scqo_qm.components.macros.two_qubit_pair_macro import QubitPairMacro

@quam_dataclass
class CZImplementation(QubitPairMacro):
    """CZ Operation for a qubit pair"""

    # flux_pulse: Pulse
    flux_pulse: str

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0

    def apply(self, *, amplitude_scale=None):
        self.qubit_control.z.play(self.flux_pulse,amplitude_scale=amplitude_scale)
        self.coupler.play(self.flux_pulse)
        # wait(operation_gap_ns//4)
        
        self.qubit_pair.align()
        self.qubit_control.xy.frame_rotation_2pi(self.phase_shift_control)
        self.qubit_target.xy.frame_rotation_2pi(self.phase_shift_target)
        self.qubit_pair.align()
        # Copy from quam
        # self.flux_pulse.play(amplitude_scale=amplitude_scale)
        # self.qubit_control.align(self.qubit_target)
        # self.qubit_control.xy.frame_rotation(self.phase_shift_control)
        # self.qubit_target.xy.frame_rotation(self.phase_shift_target)
        # self.qubit_pair.align()
