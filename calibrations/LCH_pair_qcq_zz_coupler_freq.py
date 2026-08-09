# %% {Imports}
from dataclasses import asdict

import numpy as np
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

from scqo_qm.probes import pair_qcq_zz_coupler_freq as probe
from customized.node.LCH_pair_qcq_zz_coupler_freq import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_decay_rate_data,
    plot_joint_states,
    plot_raw_data,
    plot_zz_vs_coupler,
    process_raw_dataset,
)

# %% {Initialisation}
description = """
RESIDUAL ZZ COUPLING vs COUPLER FREQUENCY (QCQ pair)

This calibration measures the residual ZZ interaction in a qubit-coupler-qubit (QCQ) pair as a
function of the coupler frequency, in order to find the coupler operating point where ZZ -> 0.

The coupler has no RF-frequency field: its frequency is tuned via the gated `const` flux-pulse
amplitude. The swept quantity is therefore the coupler bias amplitude (in volts), which sets the
coupler frequency during the interaction.

For each qubit pair, the sequence (a Hahn echo with virtual detuning):
1. Resets the control and target qubits
2. Prepares the measured qubit in a superposition
3. Applies two coupler pulses (at the swept coupler bias) separated by echo pi-pulses on both qubits
4. Applies a virtual phase rotation corresponding to the virtual detuning
5. Measures the selected qubit using either state discrimination or IQ readout

The experiment is repeated for a range of coupler bias amplitudes and interaction times.
Each coupler-flux slice is a Ramsey-like decaying oscillation, so it is fitted with scqat's Ramsey
estimator to extract:
1. The fringe frequency f = |ZZ + virtual_detuning|
2. The signed residual ZZ ζ = f − virtual_detuning (sign recovered against the known virtual
   detuning, the same way LCH_Ramsey recovers the qubit detuning; valid while ωb > |ζ|)
3. The period 1 / f
4. The decay time constant tau = T2*

Both qubits of the pair are read out. The fit uses the measured qubit's signal (its excited-state
population with state discrimination, or its I quadrature otherwise); the joint two-qubit
populations P00/P01/P10/P11 are also recorded for visualization.

This is a characterization node: it maps the signed residual ZZ ζ (and the Ramsey period) vs coupler
bias, but performs no automated state writeback. The ZZ-off point is where ζ crosses zero on the
`zz_vs_coupler` figure (equivalently where the fitted period crosses the 1/virtual-detuning line on
`raw_data_with_period`); read it off and set the coupler bias manually.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_pair_qcq_zz_coupler_freq",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under customized/node/LCH_pair_qcq_zz_coupler_freq/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.num_shots = 50
    node.parameters.use_state_discrimination = True
    node.parameters.virtual_detuning_in_mhz = 0.5
    node.parameters.time_max_in_ns = 8000
    node.parameters.time_step_in_ns = 160
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits

    # Coupler bias amplitudes (volts) — tune the coupler frequency via the gated `const` flux pulse
    amplitudes = np.arange(node.parameters.amp_min, node.parameters.amp_max, node.parameters.amp_step)
    durations = (
        np.arange(node.parameters.time_min_in_ns, node.parameters.time_max_in_ns, node.parameters.time_step_in_ns) // 4
    )
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubit_pairs,
        amplitudes=amplitudes,
        durations=durations,
        detuning_hz=node.parameters.virtual_detuning_in_mhz * u.MHz,
        num_shots=node.parameters.num_shots,
        reset_type=node.parameters.reset_type,
        use_state_discrimination=node.parameters.use_state_discrimination,
        measure_qubit=node.parameters.measure_qubit,
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
    _, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (run half): execute on the QOP and store the raw dataset as "ds_raw"."""
    dataset = probe.acquire(
        node.machine,
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
        num_shots=node.parameters.num_shots,
        timeout=node.parameters.timeout,
        log=node.log,
    )
    measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
    dataset = dataset.assign_coords(measured_qubit_name=("qubit_pair", measured_qubit_names))
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    measured_qubits = []
    for qp in node.namespace["qubit_pairs"]:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    if "measured_qubit_name" not in node.results["ds_raw"].coords:
        measured_qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(
            measured_qubit_name=("qubit_pair", measured_qubit_names)
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset and the fitted results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # Store fit results in the format expected by the rest of the node
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(fit_results, log_callable=node.log)

    # Set outcomes based on fit success
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result.success else "failed")
        for qubit_pair_name, fit_result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot J_eff, decay time, raw oscillations, and fitted oscillations."""
    qubit_pairs = node.namespace["qubit_pairs"]
    ds_fit = node.results["ds_fit"]

    fig_zz = plot_zz_vs_coupler(ds_fit, qubit_pairs)
    fig_decay = plot_decay_rate_data(ds_fit, qubit_pairs, log_y=True)
    # Signal heatmap (bright/dark fringes) with the fitted Ramsey period (1/f) overlaid per
    # coupler-flux slice, so the fitted points can be checked against the fringe spacing.
    fig_raw = plot_raw_data(ds_fit, qubit_pairs)
    # Joint two-qubit populations (P00/P01/P10/P11) per pair, or both qubits' I/Q without
    # state discrimination — one figure per pair.
    state_figs = plot_joint_states(
        node.results["ds_raw"],
        qubit_pairs,
        use_state_discrimination=node.parameters.use_state_discrimination,
    )

    node.results["figures"] = {
        **({"zz_vs_coupler": fig_zz} if fig_zz is not None else {}),
        **({"decay_time": fig_decay} if fig_decay is not None else {}),
        "raw_data_with_period": fig_raw,
        **{f"joint_states_{pair}": fig for pair, fig in state_figs.items()},
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No automated state writeback — this node characterizes ZZ vs coupler bias.

    Read the ZZ-off point off the `zz_vs_coupler` figure (where |ζ| is minimised) and set the
    coupler bias manually.
    """
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results to the node storage."""
    node.save()


# %%
