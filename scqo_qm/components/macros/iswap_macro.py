from quam.core import quam_dataclass
from quam.components.pulses import Pulse
# from quam.components.macro import QubitPairMacro
from scqo_qm.components.macros.two_qubit_pair_macro import QubitPairMacro

@quam_dataclass
class ISwapImplementation(QubitPairMacro):
    """ISWAP Operation for a qubit pair"""

    # flux_pulse: Pulse
    flux_pulse: str

    phase_shift_control: float = 0.0
    phase_shift_target: float = 0.0

    def apply(self, *, ctrl_amp=None, cplr_amp=None):

        # Bare call (ctrl_amp/cplr_amp = None) plays each flux pulse at its defined
        # amplitude; an explicit amp rescales relative to the pulse's defined amplitude.
        # This keeps the bare `apply()` usable as a named macro while preserving the
        # existing explicit-amplitude callers (e.g. LCH_iswap_fixed_time_search).
        if ctrl_amp is None:
            self.qubit_control.z.play(self.flux_pulse)
        else:
            ctrl_amp_ref = self.qubit_control.z.operations[self.flux_pulse].amplitude
            self.qubit_control.z.play(self.flux_pulse, amplitude_scale=ctrl_amp / ctrl_amp_ref)

        if cplr_amp is None:
            self.coupler.play(self.flux_pulse)
        else:
            cplr_amp_ref = self.coupler.operations[self.flux_pulse].amplitude
            self.coupler.play(self.flux_pulse, amplitude_scale=cplr_amp / cplr_amp_ref)

        # wait(operation_gap_ns//4)
        
        self.qubit_pair.align()

        # Copy from quam
        # self.flux_pulse.play(amplitude_scale=amplitude_scale)
        # self.qubit_control.align(self.qubit_target)
        # self.qubit_control.xy.frame_rotation(self.phase_shift_control)
        # self.qubit_target.xy.frame_rotation(self.phase_shift_target)
        # self.qubit_pair.align()
