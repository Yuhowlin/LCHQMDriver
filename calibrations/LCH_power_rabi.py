# %% {Imports}
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_power_rabi import Parameters, analysis, update
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qubit_power_rabi as probe


# %% {Description}
description = """
        POWER RABI (single pulse, customized)
A trimmed power-Rabi: sweep the qubit drive amplitude (as a pre-factor of the current
pulse amplitude) and fit a cosine to recalibrate the pi-pulse amplitude.

Differences vs 04b_power_rabi:
    - No error-amplification (N_pi) loop: the operation is played exactly once per
      amplitude point.
    - Uniform stream processing: always I_st[i].buffer(len(amps)).average() regardless
      of the operation.
    - drive_qubit selection: if drive_qubit is None, all qubits in qubits are driven;
      otherwise only that qubit is driven while all qubits are still read out.

Analysis is delegated to scqat's PowerRabiEstimator (cosine fit).

State update:
    - The qubit pulse amplitude of the selected operation
    (qubit.xy.operations[operation].amplitude); x90 is set to half when operation is x180
    and update_x90 is enabled.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.power_rabi` (shared with scqo); the scqat analysis adapter and
update policy live in `customized.node.LCH_power_rabi`.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_power_rabi",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.qubits = ["q1"]
    node.parameters.drive_qubit = None
    node.parameters.multiplexed = True
    # node.parameters.operation = "x180"
    node.parameters.min_amp_factor = 0.0
    node.parameters.max_amp_factor = 1.99
    node.parameters.amp_factor_step = 0.005
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the core."""
    node.namespace["qubits"] = qubits = get_qubits(node)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        amps=amps,
        operation=node.parameters.operation,
        num_shots=node.parameters.num_shots,
        reset_type=node.parameters.reset_type,
        use_state_discrimination=node.parameters.use_state_discrimination,
        drive_qubit=node.parameters.drive_qubit,
        simulate=node.parameters.simulate,
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
    """estimate: fit each qubit's power-Rabi cosine with scqat (via the core)."""
    node.results["fit_results"], node.results["figures"] = analysis.fit(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Figures are produced by the scqat estimator in analyse_data."""
    pass


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """update: recalibrate the pi-pulse amplitude from the fitted prefactor (via the core)."""
    operation = node.parameters.operation
    drive_qubit = node.parameters.drive_qubit
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            # When a single qubit is driven, only it is meaningfully calibrated; skip the
            # others so a cross-driven neighbour's oscillation can't rewrite its pi_amp.
            if drive_qubit is not None and q.name != drive_qubit:
                continue
            if node.outcomes[q.name] == "failed":
                continue

            q_update = update.compute_update(
                node.results["fit_results"][q.name], q.xy.operations[operation].amplitude
            )
            update.apply_update(q, operation, q_update, update_x90=node.parameters.update_x90)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
