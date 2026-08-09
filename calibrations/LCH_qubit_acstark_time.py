# %% {Imports}
import warnings

import numpy as np
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_qubit_acstark_time import Parameters, analysis
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qubit_acstark_time as probe

# %% {Description}
description = """
        TIME-DEPENDENT READOUT-RESONATOR PHOTON / AC-STARK vs TIME (customized)
A readout-resonator test pulse populates the resonator; the qubit is then probed at a
swept delay after the test pulse while the drive frequency is swept across the qubit
line. Locating the AC-Stark-shifted qubit line per delay traces the resonator photon
number filling up and ringing down in time.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.qubit_acstark_time`; the scqat estimate adapter
(`ReadoutPulsePhotonEstimator`) lives in `customized.node.LCH_qubit_acstark_time`.
"""


node = QualibrationNode[Parameters, Quam](
    name="LCH_qubit_acstark_time",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.qubits = ["q1"]
    node.parameters.simulate = False
    node.parameters.ro_operation = "readout"
    node.parameters.test_operation = "readout"
    node.parameters.xy_time_resolution_in_ns = 16
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    p = node.parameters
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    # Check if the qubits have a z-line attached
    if any([q.z is None for q in qubits]):
        warnings.warn("Found qubits without a flux line. Skipping")

    # Qubit detuning sweep with respect to their resonance frequencies
    dfs = np.linspace(p.min_frequency_in_mhz * u.MHz, p.max_frequency_in_mhz * u.MHz, p.num_frequency_points)
    # XY-probe delay sweep relative to the resonator drive onset (multiples of 4 ns; may be negative)
    res = p.xy_time_resolution_in_ns * u.ns
    delay_time_array = (np.arange(p.xy_delay_start_in_ns, p.xy_delay_end_in_ns + 1, res) // 4) * 4  # in ns

    (
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
    ) = probe.build_program(
        node.machine,
        qubits,
        dfs=dfs,
        delay_time_array=delay_time_array,
        num_shots=p.num_shots,
        reset_type=p.reset_type,
        xy_operation=p.xy_operation,
        xy_operation_amplitude_factor=p.xy_operation_amplitude_factor,
        ro_operation=p.ro_operation,
        test_operation=p.test_operation,
        rr_depletion_time=p.rr_depletion_time,
        use_state_discrimination=p.use_state_discrimination,
        multiplexed=p.multiplexed,
        simulate=p.simulate,
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (run half): execute on the QOP and store the raw dataset as "ds_raw"."""
    node.results["ds_raw"] = probe.acquire(
        node.machine,
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
        num_shots=node.parameters.num_shots,
        timeout=node.parameters.timeout,
        log=node.log,
    )


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """estimate: run scqat's ReadoutPulsePhotonEstimator per qubit and store figures."""
    node.results["fit_results"], node.results["figures"] = analysis.estimate(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.outcomes = {
        qubit_name: ("successful" if fit_result.get("success") else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Figures are produced in analyse_data."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
