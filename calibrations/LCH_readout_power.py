# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import readout_power as probe
from customized.node.LCH_readout_power import (
    Parameters,
)


# %% {Description}
description = """
        Ask LCH
"""


node = QualibrationNode[Parameters, Quam](
    name="LCH_readout_power",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    amps = np.linspace(node.parameters.start_amp, node.parameters.end_amp, node.parameters.num_amps)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        amps=amps,
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
    """Find the readout amplitude that maximises single-shot fidelity. For each qubit
    the swept-amplitude state-discrimination data is handed to scqat's
    ReadoutPowerFidelityEstimator, which returns the optimal amp_prefactor (a
    multiplier on the current readout amplitude) constrained by ``outliers_threshold``.
    Figures are deferred to plot_data, so the fit is skipped only once.

    ds_raw already carries the I/Q vars and shot_idx/amp_prefactor/prepared_state
    coords that scqat expects (see sweep_axes above), so no renaming is needed.

    Note: scqat's ReadoutPowerFidelityEstimator does NOT port qcat's power-specific
    linear mean-drift refit (fit_means_vs_amp_prefactor); the core per-amplitude
    state-discrimination sweep is preserved."""
    from scqat.parsers import repetition_data
    from scqat.estimators.readout_fidelity import ReadoutPowerFidelityEstimator

    estimator = ReadoutPowerFidelityEstimator()
    node.namespace["estimator"] = estimator
    node.namespace["sep_results"] = {}
    node.results["fit_results"] = {}

    for sq in repetition_data(node.results["ds_raw"], repetition_dim="qubit"):
        qubit_name = sq["qubit"].values.item()
        results = estimator.analyze(
            sq, output_dir=None, skip_figures=True,
            outliers_threshold=node.parameters.outliers_threshold,
        )[0]
        best = results["best_sweep_value"]
        node.results["fit_results"][qubit_name] = {
            "best_amp_prefactor": float(best) if best is not None else float("nan"),
            "best_fidelity": float(results["best_fidelity"]) if results["best_fidelity"] is not None else float("nan"),
            "success": bool(results["success"]),
        }
        node.namespace["sep_results"][qubit_name] = (sq, results)

    for q_name, fit in node.results["fit_results"].items():
        node.log(
            f"Results for qubit {q_name}: "
            f"optimal amp prefactor: {fit['best_amp_prefactor']:.4f} | "
            f"fidelity: {fit['best_fidelity']:.4f} | "
            f"{'SUCCESS!' if fit['success'] else 'FAIL!'}"
        )
    node.outcomes = {
        qubit_name: ("successful" if fit["success"] else "failed")
        for qubit_name, fit in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.plot)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Redraw the scqat readout-power fidelity figures for each qubit from the stored
    (dataset, results) pairs."""
    estimator = node.namespace["estimator"]
    node.results["figures"] = {}
    for qubit_name, (sq, results) in node.namespace["sep_results"].items():
        node.results["figures"][qubit_name] = estimator.generate_figures(sq, results)
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Scale each qubit's readout amplitude by the optimal prefactor found above."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            op = q.resonator.operations["readout"]
            prefactor = node.results["fit_results"][q.name]["best_amp_prefactor"]
            op.amplitude = float(op.amplitude * prefactor)

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
