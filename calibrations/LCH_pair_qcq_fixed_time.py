# %% {Imports}
import numpy as np

from qualibrate import QualibrationNode
from quam_config import Quam
from customized.node.LCH_pair_qcq_fixed_time import Parameters, analysis, plotting
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import pair_qcq_fixed_time as probe


# %% {Description}
description = """
        FIXED-TIME QUBIT-FLUX x COUPLER-FLUX 2D SWEEP (single excitation, customized)
A fixed-time 2D variant of LCH_pair_qq_chevron. Instead of sweeping a flux amplitude x
duration, the pulse duration is FIXED and TWO flux amplitudes are swept, forming a 2D
color map: the COUPLER flux amplitude (x axis, on qp.coupler) and a QUBIT flux amplitude
(y axis, on the flux_role qubit's z line). Both flux pulses play simultaneously over the
same fixed window. One qubit of the pair is excited with x180 (selected by drive_role,
default the control qubit); both qubits are read out.

Each amplitude is read either as absolute volts (amp_mode="absolute", the emitted pulse
equals the swept value and |a/ref| < 2 is enforced) or as a unitless amplitude prefactor
(amp_mode="prefactor"). The coupler op defaults to "swap_01_10_square" and the qubit z op
to "const", with a shared tunable duration (flux_time; None uses each op's native length).

Analysis is intentionally an empty estimator: no fit and no state writeback. The node
renders a 2D color map (joint populations P00/P01/P10/P11) per qubit pair.

This node is a thin qualibrate shell: the acquisition probe lives in
`scqo_qm.probes.pair_qcq_fixed_time`; the (no-op) estimate adapter and the
plot live in `customized.node.LCH_pair_qcq_fixed_time`.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_pair_qcq_fixed_time",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.simulate = False
    node.parameters.num_shots = 200
    # FlatTopCosinePulse flux op (register first: python quam_config/register_flattop_cosine.py).
    # Leave flux_time = None so each op plays its native length (duration override is meant for
    # the constant/square waveform, not a shaped pulse).
    node.parameters.coupler_operation = "flattop_cosine"
    node.parameters.qubit_operation = "flattop_cosine"
    # node.parameters.flux_time = 100
    # node.parameters.flux_role = "control"
    # node.parameters.drive_role = "control"
    node.parameters.coupler_amp_start = -0.15
    node.parameters.coupler_amp_end = 0.15
    node.parameters.coupler_amp_step = 0.01
    node.parameters.qubit_amp_start = 0.145
    node.parameters.qubit_amp_end = 0.160
    node.parameters.qubit_amp_step = 0.001
    node.parameters.amp_mode = "absolute"
    # Debug isolation: play the swap through the iswap macro's .apply() (the qc_swap_reset
    # path) instead of the direct flux play, to test whether the macro reproduces the swap.
    # In macro mode the qubit (y) sweep scales the macro's ctrl flux and the coupler plays
    # bare at 0 (so the coupler x-sweep is ignored); look at the qubit_flux=0.139 row.
    # node.parameters.swap_via_macro = True
    node.parameters.swap_via_macro = False
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the two amplitude sweep axes and the QUA program via the core."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    coupler_amplitudes = np.arange(
        node.parameters.coupler_amp_start, node.parameters.coupler_amp_end, node.parameters.coupler_amp_step
    )
    qubit_amplitudes = np.arange(
        node.parameters.qubit_amp_start, node.parameters.qubit_amp_end, node.parameters.qubit_amp_step
    )
    (
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
    ) = probe.build_program(
        node.machine,
        qubit_pairs,
        coupler_amplitudes=coupler_amplitudes,
        qubit_amplitudes=qubit_amplitudes,
        coupler_operation=node.parameters.coupler_operation,
        qubit_operation=node.parameters.qubit_operation,
        flux_time=node.parameters.flux_time,
        flux_role=node.parameters.flux_role,
        amp_mode=node.parameters.amp_mode,
        num_shots=node.parameters.num_shots,
        reset_type=node.parameters.reset_type,
        use_state_discrimination=node.parameters.use_state_discrimination,
        drive_role=node.parameters.drive_role,
        swap_via_macro=node.parameters.swap_via_macro,
        swap_operation=node.parameters.swap_operation,
        simulate=node.parameters.simulate,
    )


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
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
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """estimate (no-op) + render the 2D color map for visualization."""
    node.results["fit_results"] = analysis.estimate(
        node.results["ds_raw"],
        use_state_discrimination=node.parameters.use_state_discrimination,
    )
    node.results["figures"] = plotting.plot_fixed_time_2d(
        node.results["ds_raw"],
        node.namespace["qubit_pairs"],
        use_state_discrimination=node.parameters.use_state_discrimination,
        swap_via_macro=node.parameters.swap_via_macro,
        amp_mode=node.parameters.amp_mode,
        qubit_operation=node.parameters.qubit_operation,
        coupler_operation=node.parameters.coupler_operation,
        flux_role=node.parameters.flux_role,
        swap_operation=node.parameters.swap_operation,
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
