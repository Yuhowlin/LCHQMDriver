from quam.core import quam_dataclass
from quam.components.pulses import Pulse
# from quam.components.macro import QubitPairMacro
from scqo_qm.components.macros.two_qubit_pair_macro import QubitPairMacro
from typing import Tuple

from quam.components.macro import QubitMacro

from scqo_qm.quam_builder.architecture.superconducting import qubit
from qm.qua import *

@quam_dataclass
class ROCPLRDispersive(QubitPairMacro):
    """ISWAP Operation for a qubit pair"""

    # flux_pulse: Pulse

    # phase_shift_control: float = 0.0
    # phase_shift_target: float = 0.0

    def apply(self, pulse_name: str, qua_vars=None, stream=None, use_state_discrimination: bool = False): 

        self.qubit_control.xy.play("x180")

        self.qubit_pair.align()
        self.qubit_pair.wait(40//4)  # wait for the qubits to be in the right state before measurement
        # wait(operation_gap_ns//4)
        
        self.qubit_pair.align()
        if use_state_discrimination:
            self.qubit_control.readout_state(qua_vars, pulse_name)
            save(qua_vars, stream)
        else:
            self.qubit_control.resonator.measure(pulse_name, qua_vars=qua_vars)

            # save data
            save(qua_vars[0], stream[0])
            save(qua_vars[1], stream[1])



@quam_dataclass
class ROZZ(QubitMacro):
    """ISWAP Operation for a qubit pair"""

    # flux_pulse: Pulse

    # phase_shift_control: float = 0.0
    # phase_shift_target: float = 0.0

    def apply(self, pulse_name: str, qua_vars=None, stream=None, use_state_discrimination: bool = False): 

        self.qubit.xy.play("x180")

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
