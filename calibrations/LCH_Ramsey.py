# %% {Imports}
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits, get_idle_times_in_clock_cycles
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qubit_ramsey as probe
from customized.node.LCH_Ramsey import Parameters, analysis, update


# %% {Description}
description = """
        Ramsey with virtual detuning (x90 -> idle -> y90, the drive virtually
        detuned by `frequency_detuning_in_mhz`). Sweeps the idle time and fits the
        decaying fringe with scqat's RamseyEstimator, which selects between a
        single damped sine, a two-frequency beat (charge dispersion), and a pure
        relaxation decay (when the fringe frequency is ~0). The fitted frequency
        calibrates the qubit: it updates `q.f_01` and `q.xy.RF_frequency`, and for
        the beat case records `q.charge_dispersion` from the frequency split.

        This node is a thin qualibrate shell: the acquisition probe lives in
        `scqo_qm.probes.ramsey` (shared with scqo); the scqat analysis adapter
        and update policy live in `customized.node.LCH_Ramsey`.
"""

node = QualibrationNode[Parameters, Quam](name="LCH_Ramsey", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1","q2","q3"]
    node.parameters.frequency_detuning_in_mhz = 1
    node.parameters.num_shots = 200
    node.parameters.log_or_linear_sweep = "linear"
    node.parameters.wait_time_num_points = 100
    node.parameters.max_wait_time_in_ns = 4000
    node.parameters.multiplexed = False
    # node.parameters.simulate = True
    pass


## Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the core."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        idle_times_cycles=get_idle_times_in_clock_cycles(node.parameters),
        detuning_hz=node.parameters.frequency_detuning_in_mhz * u.MHz,
        num_shots=node.parameters.num_shots,
        reset_type=node.parameters.reset_type,
        use_state_discrimination=node.parameters.use_state_discrimination,
        simulate=node.parameters.simulate,
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
    """estimate: fit each qubit's probed Ramsey with scqat's RamseyEstimator (via the core)."""
    node.results["fit_results"], node.results["figures"] = analysis.fit(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """update: correct the qubit frequency from the fitted Ramsey fringe (via the core)."""
    detuning_hz = int(node.parameters.frequency_detuning_in_mhz * 1e6)
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            q_update = update.compute_update(node.results["fit_results"][q.name], detuning_hz)
            update.apply_update(q, q_update)

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
