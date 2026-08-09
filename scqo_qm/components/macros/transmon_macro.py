from quam.core import quam_dataclass
from quam.components.pulses import Pulse
# from quam.components.macro import QubitPairMacro
from typing import Tuple

from quam.components.macro import QubitMacro

from qm.qua import *



@quam_dataclass
class ROZZ(QubitMacro):
    """REF
    Characterization of tunable coupler without a dedicated readout resonator in superconducting circuits
    """

    # flux_pulse: Pulse

    # phase_shift_control: float = 0.0
    # phase_shift_target: float = 0.0

    def apply(self, pulse_name: str, qua_vars=None, stream=None, use_state_discrimination: bool = False, xy_duration_ratio: float = 1.0): 
        
        duration = int(self.qubit.xy.operations["x180"].length*xy_duration_ratio)
        # print(f"Duration of x180 pulse: {duration} ns {duration*xy_duration_ratio}")
        self.qubit.xy.play("x180",duration=duration // 4, amplitude_scale=1/xy_duration_ratio )

        self.qubit.align()
        self.qubit.wait(40//4)  # wait for the qubits to be in the right state before measurement
        # wait(operation_gap_ns//4)
        
        self.qubit.align()
        if use_state_discrimination:
            self.qubit.readout_state(qua_vars, pulse_name)
            save(qua_vars, stream)
        else:
            self.qubit.resonator.measure(pulse_name, qua_vars=qua_vars)

            # save data
            save(qua_vars[0], stream[0])
            save(qua_vars[1], stream[1])
