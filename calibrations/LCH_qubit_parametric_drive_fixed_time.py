# %% {Imports}
import numpy as np

from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qubit_parametric_drive_fixed_time as probe
from customized.node.LCH_qubit_parametric_drive_fixed_time import Parameters


# %% {Node initialisation}
description = """
        Parametric-drive resonance map at fixed drive time.

        Prepares the qubit (x180), applies a parametric (flux-line) drive of fixed
        duration while sweeping the drive amplitude ratio and frequency, then reads
        out the excited-state population. scqat's ParametricDriveResonanceEstimator
        finds the resonance peak(s) on the 2-D amplitude x frequency map (a Lorentzian
        fit per amplitude slice, like qubit-spectroscopy-vs-flux) and reports the
        peak point-cloud (drive_amp, frequency, fwhm). Characterization only —
        no device-state writeback.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_qubit_parametric_drive_fixed_time",  # Name should be unique
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
    node.parameters.drive_amp_max = 0.1
    node.parameters.drive_amp_min = 0.3
    node.parameters.drive_amp_points = 21
    node.parameters.amp_mode = "absolute"
    node.parameters.max_frequency_mhz = 300
    node.parameters.min_frequency_mhz = 400
    node.parameters.frequency_points = 101
    node.parameters.use_state_discrimination = True
    node.parameters.simulate = False
    node.parameters.num_shots = 100
    node.parameters.multiplexed = True
    node.parameters.driving_time_in_ns = 200
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

    # Drive amplitude sweep (volts if amp_mode == "absolute", else a unitless prefactor).
    r_amps = np.linspace(p.drive_amp_min, p.drive_amp_max, p.drive_amp_points)
    freqs = np.linspace(p.min_frequency_mhz * u.MHz, p.max_frequency_mhz * u.MHz, p.frequency_points)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        r_amps=r_amps,
        freqs=freqs,
        amp_mode=p.amp_mode,
        driving_time_in_ns=p.driving_time_in_ns,
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
    """estimate: find the parametric-resonance peak(s) on the 2-D amplitude x
    frequency map via scqat's ParametricDriveResonanceEstimator.

    Each drive_amp slice is fit with a Lorentzian (delegating to the
    qubit-spectroscopy peak finder); the kept peaks form a point-cloud
    (drive_amp, frequency, fwhm) over the map. The estimator owns the
    plotting, so no separate plot step is needed."""
    from scqat.parsers import repetition_data
    from scqat.estimators import ParametricDriveResonanceEstimator

    ds = node.results["ds_raw"]
    if not node.parameters.use_state_discrimination:
        ds = ds.rename({"I": "signal"})

    sep_data = repetition_data(ds, repetition_dim="qubit")
    node.results["fit_results"] = {}
    node.results["figures"] = {}
    estimator = ParametricDriveResonanceEstimator()
    for sq_data in sep_data:
        qubit_name = sq_data["qubit"].values.item()
        results, figs = estimator.analyze(
            sq_data, output_dir=None, skip_figures=not node.parameters.plot
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
