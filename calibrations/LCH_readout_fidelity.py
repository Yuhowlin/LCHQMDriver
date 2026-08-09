# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import readout_fidelity as probe
from customized.node.LCH_readout_fidelity import (
    Parameters,
)


# %% {Description}
description = """
        Ask LCH
"""
# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_readout_fidelity",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """
    Allow the user to locally set the node parameters for debugging purposes, or
    execution in the Python IDE.
    """
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q2", "q3"]
    node.parameters.multiplexed = False
    node.parameters.num_shots = 4000
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        operation=node.parameters.operation,
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
    """Characterise single-shot readout fidelity at the current settings. For each
    qubit scqat's StateDiscriminationEstimator fits the IQ blobs; the readout fidelity
    is the mean of the confusion-matrix diagonal (direct_counts[k, k]). Figures are
    deferred to plot_data, so the fit is skipped only once.

    ds_raw already carries the I/Q vars and shot_idx/prepared_state coords that scqat
    expects (see sweep_axes above), so no renaming is needed."""
    from scqat.parsers import repetition_data
    from scqat.estimators.state_discrimination import StateDiscriminationEstimator

    estimator = StateDiscriminationEstimator()
    node.namespace["estimator"] = estimator
    node.namespace["sep_results"] = {}
    node.results["fit_results"] = {}

    for sq in repetition_data(node.results["ds_raw"], repetition_dim="qubit"):
        qubit_name = sq["qubit"].values.item()
        results = estimator.analyze(sq, output_dir=None, skip_figures=True)[0]
        dc = np.asarray(results["direct_counts"])
        n = min(dc.shape[0], dc.shape[1])
        fidelity = float(np.mean([dc[k, k] for k in range(n)]))
        # SNR = |mean_1 - mean_0| / sigma: the IQ-plane separation of the two readout
        # blobs in units of one blob's (GMM) standard deviation.
        tp = results["trained_paras"]
        centers = np.asarray(tp["mean"], dtype=float)
        sigma = float(tp["std"])
        snr = (
            float(np.linalg.norm(centers[0] - centers[1]) / sigma)
            if centers.shape[0] >= 2 and sigma > 0 else float("nan")
        )
        node.results["fit_results"][qubit_name] = {
            "fidelity": fidelity,
            "snr": snr,
            "success": bool(np.isfinite(fidelity) and fidelity >= 0.5),
        }
        node.namespace["sep_results"][qubit_name] = (sq, results)

    for q_name, fit in node.results["fit_results"].items():
        node.log(
            f"Results for qubit {q_name}: "
            f"readout fidelity: {fit['fidelity']:.4f} | "
            f"SNR: {fit['snr']:.2f} | "
            f"{'SUCCESS!' if fit['success'] else 'FAIL!'}"
        )
    node.outcomes = {
        qubit_name: ("successful" if fit["success"] else "failed")
        for qubit_name, fit in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.plot)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Redraw the scqat state-discrimination figures for each qubit from the stored
    (dataset, results) pairs."""
    estimator = node.namespace["estimator"]
    node.results["figures"] = {}
    for qubit_name, (sq, results) in node.namespace["sep_results"].items():
        node.results["figures"][qubit_name] = estimator.generate_figures(sq, results)
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No state write-back: this node only characterises single-shot readout fidelity
    at the current settings, so there is no single scalar to commit. Deriving the
    readout angle / threshold / confusion matrix from the GMM is a separate
    calibration (cf. the readout-frequency and readout-power nodes, which do write
    their optimum back)."""
    pass

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
