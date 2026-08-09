# %% {Imports}
import numpy as np

from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qubit_parametric_drive_freq_time as probe
from customized.node.LCH_qubit_parametric_drive_time import Parameters


# %% {Node initialisation}
description = """
        Parametric-drive qubit decoherence (rho_11 vs frequency and time).

        Prepares the qubit, applies a fixed-amplitude parametric (flux-line) drive
        while sweeping the drive frequency and duration, then reads out. With
        `tomography=False` (default) it prepares |1> and reads out the excited-state
        population directly (rho_11-only). With `tomography=True` it sweeps an extra
        X/Y/Z readout-basis axis for full single-qubit state tomography (set
        `prepare_state` to a superposition, e.g. "x90"/"-x90"). Either way scqat's
        ParametricDriveDecoherenceEstimator reconstructs rho_11(t), fits the
        non-Markovian amplitude-damping model per driving frequency, and reports
        gamma / lambda / Delta and the exceptional-point figure of merit
        8*lambda^2/gamma^2 vs frequency. Characterization only — no device-state
        writeback.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_qubit_parametric_drive_freq_time",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q2"]
    node.parameters.max_driving_time_ns = 420
    node.parameters.min_driving_time_ns = 20
    node.parameters.driving_time_step = 8
    node.parameters.max_frequency_mhz = 354.7
    node.parameters.min_frequency_mhz = 354.7
    node.parameters.frequency_points = 1
    node.parameters.drive_amp = 0.22
    node.parameters.amp_mode = "absolute"
    node.parameters.use_state_discrimination = True
    node.parameters.simulate = False
    node.parameters.num_shots = 400
    node.parameters.multiplexed = True
    # Set tomography = True (and a superposition prepare_state) for full X/Y/Z tomography.
    # node.parameters.tomography = True
    # node.parameters.prepare_state = "-x90"
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

    # Driving-time sweep (in clock cycles) and driving-frequency sweep (Hz).
    time_tick = np.arange(p.min_driving_time_ns // 4, p.max_driving_time_ns // 4, p.driving_time_step // 4)
    freqs = np.linspace(p.min_frequency_mhz * u.MHz, p.max_frequency_mhz * u.MHz, p.frequency_points)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        freqs=freqs,
        time_tick=time_tick,
        drive_amp=p.drive_amp,
        amp_mode=p.amp_mode,
        prepare_state=p.prepare_state,
        tomography=p.tomography,
        num_shots=p.num_shots,
        reset_type=p.reset_type,
        use_state_discrimination=p.use_state_discrimination,
        simulate=p.simulate,
        log=node.log,
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
    """estimate: per-driving_frequency non-Markovian decoherence fit via scqat's
    ParametricDriveDecoherenceEstimator.

    The estimator auto-detects the layout: with tomography it rebuilds the density
    matrix from the X/Y/Z basis readouts; without it, it takes the rho_11-only path.
    Either way it fits rho_11(t) at each driving frequency and returns gamma / lambda /
    Delta and the exceptional-point figure of merit 8*lambda^2/gamma^2. The estimator
    owns the plotting, so no separate plot step is needed."""
    from scqat.parsers import repetition_data
    from scqat.estimators import ParametricDriveDecoherenceEstimator

    ds = node.results["ds_raw"]
    analyze_kwargs = {}
    if not node.parameters.use_state_discrimination:
        ds = ds.rename({"I": "signal"})
        # Raw I quadrature is NOT a population — supply this qubit's readout-contrast
        # correction so rho_11 lands in [0,1]. (The estimator default is identity,
        # which is correct for the state-discriminated path below.)
        analyze_kwargs = dict(rho11_offset=0.045, rho11_scale=0.78)

    sep_data = repetition_data(ds, repetition_dim="qubit")
    node.results["fit_results"] = {}
    node.results["figures"] = {}
    estimator = ParametricDriveDecoherenceEstimator()
    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        # With state discrimination, `state` is already P(|1>) in [0,1], so the
        # estimator's identity default (rho11_offset=0, rho11_scale=1) is used as-is.
        results, figs = estimator.analyze(
            sq_data, output_dir=None, skip_figures=not node.parameters.plot, **analyze_kwargs
        )
        node.results["fit_results"][qubit_name] = estimator.extract_metadata(results)
        node.results["figures"][qubit_name] = figs


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """No-op: this is a characterization experiment, so nothing is written back
    to the device state."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
