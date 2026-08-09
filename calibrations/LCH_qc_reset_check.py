# %% {Imports}
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_qc_reset_check import Parameters, analysis, plotting
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qc_reset_check as probe


# %% {Description}
description = """
        RESET CHECK (power-Rabi-shaped reset diagnostic, customized)
Verify that a qubit reset macro actually works. Sweep the drive amplitude (as a pre-factor of
the current `operation` amplitude) to prepare a continuum of excited populations, then - after
the amplitude-modified drive - play the reset macro under test and read out.

To make "success" self-evident in one run, each amplitude point is measured twice along a
2-value `reset` axis:
    - reset="off": init -> x180(a) -> readout            (baseline: full Rabi oscillation)
    - reset="on" : init -> x180(a) -> reset macro -> readout  (should be flat near ground)
The node overlays the two curves per qubit; the gap between them is the reset quality.

The reset macro is invoked bare (`qubit.macros[reset_operation].apply()` with no extra args),
so the chosen macro must be callable that way (its amplitudes/pulses baked into the QUAM macro
definition, e.g. ParametricReset). If drive_qubit is set, only that qubit is driven/reset while
all qubits are still read out.

Analysis is intentionally an empty estimator: no fit and no state writeback; the node renders
the overlay for visual inspection.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.qc_reset_check`; the (no-op) estimate adapter and the plot live in
`customized.node.LCH_qc_reset_check`.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_qc_reset_check",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.qubits = ["q2"]
    node.parameters.drive_qubit = None
    node.parameters.reset_operation = "reset"
    node.parameters.multiplexed = True
    node.parameters.min_amp_factor = 0.0
    node.parameters.max_amp_factor = 1.5
    node.parameters.amp_factor_step = 0.05
    node.parameters.simulate = False
    node.parameters.num_shots = 1000
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    p = node.parameters
    node.namespace["qubits"] = qubits = get_qubits(node)
    amps = np.arange(p.min_amp_factor, p.max_amp_factor, p.amp_factor_step)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        amps=amps,
        operation=p.operation,
        reset_operation=p.reset_operation,
        num_shots=p.num_shots,
        reset_type=p.reset_type,
        use_state_discrimination=p.use_state_discrimination,
        drive_qubit=p.drive_qubit,
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
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """estimate (no-op) + render the reset off/on overlay per measured qubit."""
    node.results["fit_results"] = analysis.estimate(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.results["figures"] = plotting.plot_reset_check(
        node.results["ds_raw"],
        node.namespace["qubits"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.outcomes = {
        q.name: ("successful" if node.results["fit_results"][q.name]["success"] else "failed")
        for q in node.namespace["qubits"]
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
