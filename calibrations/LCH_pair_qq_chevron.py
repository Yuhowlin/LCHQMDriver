# %% {Imports}
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_pair_qq_chevron import Parameters, analysis, plotting
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import pair_qq_chevron as probe


# %% {Description}
description = """
        SINGLE-EXCITATION FLUX CHEVRON (customized)
A variant of 19_chevron_11_02 (two-qubit CZ chevron). The flux pulse still sweeps
amplitude x duration on the control qubit's z line and BOTH qubits are read out, but
only one qubit of the pair is excited with x180 (selected by drive_role, default the
control qubit) instead of preparing |11>.

Analysis is intentionally an empty estimator: no fit and no state writeback. The node
renders a 2D color map (control and target signals vs amplitude x duration) for visual
inspection.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.pair_qq_chevron`; the (no-op) estimate adapter and the plot live in
`customized.node.LCH_pair_qq_chevron`.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_pair_qq_chevron",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubit_pairs = ["q1-q2"]
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.num_shots = 10
    node.parameters.op_time_start = 10
    node.parameters.op_time_end = 20
    # node.parameters.drive_role = "control"
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program (with baked config) via the core."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    amplitudes = np.arange(node.parameters.amp_ratio_start, node.parameters.amp_ratio_end, node.parameters.amp_step)
    times_cycles = np.arange(node.parameters.op_time_start, node.parameters.op_time_end)
    (
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
        node.namespace["baked_config"],
    ) = probe.build_program(
        node.machine,
        qubit_pairs,
        amplitudes=amplitudes,
        times_cycles=times_cycles,
        num_shots=node.parameters.num_shots,
        reset_type=node.parameters.reset_type,
        use_state_discrimination=node.parameters.use_state_discrimination,
        drive_role=node.parameters.drive_role,
        simulate=node.parameters.simulate,
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program (using the baked config)."""
    qmm = node.machine.connect()
    config = node.namespace["baked_config"]
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (run half): execute on the QOP (with the baked config) and store the raw dataset as "ds_raw"."""
    node.results["ds_raw"] = probe.acquire(
        node.machine,
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
        num_shots=node.parameters.num_shots,
        timeout=node.parameters.timeout,
        log=node.log,
        config=node.namespace["baked_config"],
    )


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """estimate (no-op) + render the 2D color map for visualization."""
    node.results["fit_results"] = analysis.estimate(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.results["figures"] = plotting.plot_chevron_2d(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.outcomes = {
        qp.name: ("successful" if node.results["fit_results"][qp.name]["success"] else "failed")
        for qp in node.namespace["qubit_pairs"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Figure is produced in analyse_data."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
