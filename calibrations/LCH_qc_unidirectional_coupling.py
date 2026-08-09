# %% {Imports}
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_qc_unidirectional_coupling import Parameters, analysis, plotting
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import qc_unidirectional_coupling as probe


# %% {Description}
description = """
        UNIDIRECTIONAL COUPLING CIRCUIT (customized)
Exercises a unidirectional swap-chain reset scheme. Each shot prepares an excitation on the
chosen `excite_qubit` (x180) — selectable independently of the swap pairs — then applies the
circuit body R times: an ordered chain of SWAPs (one per pair in `swap_pairs`, applied in
list order, e.g. q1_q2 then q2_q3) followed by a RESET on an ancilla qubit
(`reset_qubit.macros[reset_operation]`). The number of rounds R is swept, and the selected
`qubits` (the measured set) are read out at the end, giving one population-vs-round curve per
measured qubit (R=0 is the no-round baseline).

This generalizes LCH_qc_swap_paramreset's single swap per round to an ordered list of swaps,
moving the excitation directionally along the chain before each reset.

The swap/reset macros are invoked bare (`.apply()` with no extra args), so the chosen
macros must be callable that way (their amplitudes/pulses baked into the QUAM macro
definition). Every pair in `swap_pairs` must carry a bare-callable `swap_operation` macro —
note `register_swap_macro.py` wires it onto `q1_q2` only, so register it on the other
pair(s) in the chain before running. Analysis is intentionally an empty estimator: no fit
and no state writeback; the node renders a 1D population-vs-round line per measured qubit
for visual inspection.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.qc_unidirectional_coupling`; the (no-op) estimate adapter and the plot
live in `customized.node.LCH_qc_unidirectional_coupling`.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_qc_unidirectional_coupling",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.qubits = ["q1", "q2", "q3"]
    node.parameters.excite_qubit = "q1"
    node.parameters.swap_pairs = ["q1_q2", "q2_q3"]
    node.parameters.swap_operation = "iswap"
    node.parameters.reset_qubit = "q2"
    node.parameters.reset_operation = "reset"
    node.parameters.min_rounds = 0
    node.parameters.max_rounds = 10
    node.parameters.rounds_step = 1
    node.parameters.simulate = False
    node.parameters.num_shots = 2048
    # multiplexed=True -> joint state populations (000, 001, ...); False -> per-qubit marginals.
    node.parameters.multiplexed = True
    # operation_gap_ns idles each pair's flux lines between gate operations so a preceding
    # parametric-reset flux can settle before the swap fires (0 = no gap).
    node.parameters.operation_gap_ns = 16
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    p = node.parameters
    # The MEASURED set read out at the end of the circuit.
    node.namespace["qubits"] = qubits = get_qubits(node)
    # The swap chain (ordered), the reset ancilla, and the excited qubit are named targets.
    swap_pairs = [node.machine.qubit_pairs[name] for name in p.swap_pairs]
    reset_qubit = node.machine.qubits[p.reset_qubit]
    # The qubit excited with x180 for state prep must be set explicitly (no default).
    if not p.excite_qubit:
        raise ValueError("excite_qubit must be set to the name of the qubit to excite (x180).")
    excite_qubit = node.machine.qubits[p.excite_qubit]
    # Round sweep (inclusive of max_rounds; 0 is the no-round baseline).
    rounds_array = np.arange(p.min_rounds, p.max_rounds + 1, p.rounds_step)

    (
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
    ) = probe.build_program(
        node.machine,
        list(qubits),
        swap_pairs,
        reset_qubit,
        excite_qubit=excite_qubit,
        swap_operation=p.swap_operation,
        reset_operation=p.reset_operation,
        rounds_array=rounds_array,
        num_shots=p.num_shots,
        reset_type=p.reset_type,
        use_state_discrimination=p.use_state_discrimination,
        operation_gap_ns=p.operation_gap_ns,
        simulate=p.simulate,
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
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
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """estimate (no-op) + render the 1D population-vs-round curves for visualization."""
    node.results["fit_results"] = analysis.estimate(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.results["figures"] = plotting.plot_rounds_1d(
        node.results["ds_raw"],
        node.namespace["qubits"],
        use_state_discrimination=node.parameters.use_state_discrimination,
        multiplexed=node.parameters.multiplexed,
    )
    node.outcomes = {
        q.name: ("successful" if node.results["fit_results"][q.name]["success"] else "failed")
        for q in node.namespace["qubits"]
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Figures are produced in analyse_data."""
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
