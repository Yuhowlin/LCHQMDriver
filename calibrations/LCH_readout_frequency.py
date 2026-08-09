# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_readout_frequency import Parameters, analysis, update
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import readout_frequency as probe


# %% {Description}
description = """
        Ask LCH

        This node is a thin qualibrate shell: the acquisition probe lives in
        `scqo_qm.probes.readout_frequency` (shared with scqo); the scqat analysis
        adapter and update policy live in `customized.node.LCH_readout_frequency`.
"""


node = QualibrationNode[Parameters, Quam](
    name="LCH_readout_frequency",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1"]
    node.parameters.num_shots = 4000
    node.parameters.multiplexed = True
    node.parameters.start_freq_in_mhz = -2
    node.parameters.end_freq_in_mhz = 2
    node.parameters.frequency_step_in_mhz = 0.2

    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the core."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(node.parameters.start_freq_in_mhz * u.MHz, node.parameters.end_freq_in_mhz * u.MHz, step)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        dfs=dfs,
        num_shots=node.parameters.num_shots,
        reset_type=node.parameters.reset_type,
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
    """estimate: find the readout frequency that maximises single-shot fidelity (via the
    core). Figures are deferred to plot_data, so the fit is skipped only once."""
    fit_results, sep_results, estimator = analysis.fit(node.results["ds_raw"])
    node.results["fit_results"] = fit_results
    node.namespace["sep_results"] = sep_results
    node.namespace["estimator"] = estimator

    for q_name, fit in fit_results.items():
        node.log(
            f"Results for qubit {q_name}: "
            f"optimal detuning: {1e-6 * fit['best_detuning']:.3f} MHz | "
            f"fidelity: {fit['best_fidelity']:.4f} | "
            f"{'SUCCESS!' if fit['success'] else 'FAIL!'}"
        )
    node.outcomes = {
        qubit_name: ("successful" if fit["success"] else "failed")
        for qubit_name, fit in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.plot)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Redraw the scqat readout-frequency fidelity figures from the stored fit results."""
    node.results["figures"] = analysis.figures(
        node.namespace["estimator"], node.namespace["sep_results"]
    )
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """update: shift each qubit's readout frequency by the optimal detuning (via the core)."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            q_update = update.compute_update(node.results["fit_results"][q.name])
            update.apply_update(q, q_update)

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
