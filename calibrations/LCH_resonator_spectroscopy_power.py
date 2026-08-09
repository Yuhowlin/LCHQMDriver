# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np

from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.core import tracked_updates

from scqo_qm.probes import resonator_spectroscopy_power as probe
from customized.node.LCH_resonator_spectroscopy_power import Parameters, analysis, update

# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY VERSUS READOUT POWER (LCH / scqat analysis)
Measure each resonator's |IQ| response across readout frequency and readout power to
locate the readout power just before the resonator frequency splitting, and the
resonator frequency at that power.

Unlike the official 02b node, the averaging loop runs innermost (as in
LCH_resonator_spectroscopy_flux), and the analysis is done by the scqat
ResonatorSpectroscopyPowerEstimator: it fits the resonator dip power-by-power
(single inverted Lorentzian per slice) to collapse the 2-D (power, detuning) map to a
1-D centre-frequency(power) trace, then picks the optimal readout power from where the
centre stops shifting.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.resonator_spectroscopy_power` (shared with scqo); the scqat
analysis adapter and update policy live in `customized.node.LCH_resonator_spectroscopy_power`.

Prerequisites:
    - Having calibrated the resonator frequency (node 02a / LCH_resonator_spectroscopy).
    - Having specified the desired flux point if relevant (qubit.z.flux_point).

State update:
    - The readout power: qubit.resonator.set_output_power()
    - The readout frequency at the optimal power: qubit.resonator.f_01 & qubit.resonator.RF_frequency
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_resonator_spectroscopy_power",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2", "q3"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): bump the readout power, create the sweep axes and the QUA program via the probe."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Update the readout power to match the desired range; reverted at the end of the node.
    node.namespace["tracked_resonators"] = []
    for qubit in qubits:
        with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.set_output_power(
                power_in_dbm=node.parameters.max_power_dbm,
                max_amplitude=node.parameters.max_amp,
            )
            node.namespace["tracked_resonators"].append(resonator)

    # The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
    amp_min = calculate_voltage_scaling_factor(node.parameters.max_power_dbm, node.parameters.min_power_dbm)
    amps = np.geomspace(amp_min, 1, node.parameters.num_power_points)
    power_dbm = np.linspace(
        node.parameters.min_power_dbm,
        node.parameters.max_power_dbm,
        node.parameters.num_power_points,
    )
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        dfs=dfs,
        amps=amps,
        power_dbm=power_dbm,
        num_shots=node.parameters.num_shots,
    )
    node.namespace["num_detuning_points"] = len(dfs)


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
        num_detuning_points=node.namespace["num_detuning_points"],
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
    """estimate: collapse each qubit's (power, detuning) map to a centre-vs-power
    trace with scqat and pick the optimal readout power (via the adapter)."""
    fit_results, sep_results, estimator = analysis.fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        n_sigma=node.parameters.outlier_n_sigma,
        derivative_crossing_threshold_in_hz_per_dbm=node.parameters.derivative_crossing_threshold_in_hz_per_dbm,
        derivative_smoothing_window_num_points=node.parameters.derivative_smoothing_window_num_points,
        moving_average_filter_window_num_points=node.parameters.moving_average_filter_window_num_points,
        buffer_from_crossing_threshold_in_dbm=node.parameters.buffer_from_crossing_threshold_in_dbm,
    )
    node.results["fit_results"] = fit_results
    node.namespace["sep_results"] = sep_results
    node.namespace["estimator"] = estimator

    for q_name, fit in fit_results.items():
        node.log(
            f"Results for qubit {q_name}: "
            f"optimal power: {fit['optimal_power']:.2f} dBm | "
            f"resonator frequency: {1e-9 * fit['resonator_frequency']:.3f} GHz "
            f"(shift {1e-6 * fit['frequency_shift']:.3f} MHz) | "
            f"{'SUCCESS!' if fit['success'] else 'FAIL!'}"
        )
    node.outcomes = {
        qubit_name: ("successful" if fit["success"] else "failed")
        for qubit_name, fit in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or not node.parameters.plot)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Redraw the scqat resonator-spectroscopy-vs-power figures from the stored fit results."""
    node.results["figures"] = analysis.figures(node.namespace["estimator"], node.namespace["sep_results"])
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """update: revert the temporary power bump, then write the optimal readout power
    and readout-frequency shift for each successful qubit (via the adapter)."""
    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            q_update = update.compute_update(node.results["fit_results"][q.name], node.parameters.max_amp)
            update.apply_update(q, q_update)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
