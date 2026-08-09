# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import warnings

from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot

from scqo_qm.probes import resonator_spectroscopy_flux as probe
from customized.node.LCH_resonator_spectroscopy_flux import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    fit_flux_dependence,
    log_dispersion_results,
    plot_combined,
)


# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY VERSUS FLUX — SINGLE FLUX SOURCE (LCH / scqat analysis)
Maps the resonator response by sweeping a flux bias and the readout frequency, then extracts the
resonator centre frequency as a function of flux.

Unlike the official 02c node — which forces every measured qubit to drive its own z-line — this node
adds a `z_source` parameter naming the single flux source that drives the sweep while all qubits in
`qubits` are read out (e.g. to see how one flux line shifts other resonators):
  * a qubit name      -> that qubit's z-line drives the flux;
  * a qubit-pair name -> that pair's tunable coupler drives the flux;
  * None              -> behaviour identical to 02c (each qubit fluxes itself).

Analysis is done by the scqat resonator_spectroscopy_flux stage helpers (track_dips), fitting the resonator dip
flux-by-flux (single inverted Lorentzian per slice) to reduce the 2-D (flux, detuning) map to a 1-D
centre-frequency(flux) trace. For a qubit/None source a second-stage full-transmon dispersive fit
extracts the sweet-spot flux / idle offset / phi0 and writes them back. For a coupler
source that dispersive model does not apply, so it is skipped and no state is written back.

Prerequisites:
    - Having calibrated the resonator frequency (nodes 02a, 02b and/or 02c).
    - Having specified the desired flux point (qubit.z.flux_point).
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="LCH_resonator_spectroscopy_flux",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q2", "q3"]
    # node.parameters.z_source = "q5"  # a qubit name -> sweep only its flux while reading the listed resonators
    node.parameters.z_source = "q2_q3"  # a pair name -> sweep its coupler instead
    node.parameters.simulate = False
    # node.parameters.num_shots = 1
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """probe (build half): create the sweep axes and the QUA program via the probe."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    # Check if the qubits have a z-line attached
    if any([q.z is None for q in qubits]):
        warnings.warn("Found qubits without a flux line. Skipping")
    # Flux bias sweep in V
    dcs = np.linspace(
        node.parameters.min_flux_offset_in_v,
        node.parameters.max_flux_offset_in_v,
        node.parameters.num_flux_points,
    )
    # The frequency sweep around the resonator resonance frequency
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)
    node.namespace["qua_program"], node.namespace["sweep_axes"] = probe.build_program(
        node.machine,
        qubits,
        dcs=dcs,
        dfs=dfs,
        num_shots=node.parameters.num_shots,
        z_source=node.parameters.z_source,
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
    # The program's progress "n" stream counts the outer flux loop, so the progress-bar
    # total is the number of flux points, not the shot count.
    node.results["ds_raw"] = probe.acquire(
        node.machine,
        node.namespace["qua_program"],
        node.namespace["sweep_axes"],
        num_shots=node.parameters.num_flux_points,
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
    """Fit the resonator dip flux-by-flux with scqat to reduce the 2-D map to a
    1-D centre-frequency(flux) trace. The per-qubit (slice, results) pairs are kept
    in the namespace so plot_data can redraw the figures without refitting."""
    ds = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_raw"] = ds
    sep_results, fit_results = fit_raw_data(ds, node)
    node.namespace["sep_results"] = sep_results
    node.results["fit_results"] = fit_results

    # Log the per-slice (flux-by-flux) dip-fit summary
    log_fitted_results(node.results["fit_results"], log_callable=node.log)

    # When the coupler drives the flux, the dispersive (sweet-spot/phi0) model does
    # not describe the measured qubit's resonator, so skip it and base outcomes on
    # the flux-by-flux dip fit. (Qubit / None sources keep the dispersive stage.)
    is_coupler = node.parameters.z_source is not None and node.parameters.z_source in node.machine.qubit_pairs
    if is_coupler:
        node.namespace["dispersion_sep"] = {}
        node.results["dispersion_results"] = {}
        node.outcomes = {
            qubit_name: ("successful" if fr["success"] else "failed")
            for qubit_name, fr in node.results["fit_results"].items()
        }
        return

    # Second stage: fit the centre-frequency(flux) trace with the full-transmon
    # dispersive model (sweet-spot flux, dv_phi0, f_r0; g is conditional for now).
    dispersion_sep, dispersion_results = fit_flux_dependence(sep_results, node)
    node.namespace["dispersion_sep"] = dispersion_sep
    node.results["dispersion_results"] = dispersion_results
    log_dispersion_results(dispersion_results, log_callable=node.log)

    # The dispersive fit is the deliverable that gates the state update, so the
    # node outcome reflects its per-qubit success.
    node.outcomes = {
        qubit_name: ("successful" if disp["success"] else "failed")
        for qubit_name, disp in node.results["dispersion_results"].items()
    }


# %% {Plot_data}
# Map a flux element's configured operating point (flux_point) to its DC offset in
# volts. Works for both FluxLine (qubit z-line) and TunableCoupler.
_FLUX_POINT_OFFSET_ATTR = {
    "joint": "joint_offset",
    "independent": "independent_offset",
    "min": "min_offset",
    "off": "decouple_offset",
    "on": "interaction_offset",
    "arbitrary": "arbitrary_offset",
}


def current_flux_offset(elem):
    """Return the DC flux offset (V) that `elem` idles at per its flux_point, or None."""
    flux_point = getattr(elem, "flux_point", None)
    if flux_point == "zero":
        return 0.0
    attr = _FLUX_POINT_OFFSET_ATTR.get(flux_point)
    if attr is None:
        return None
    value = getattr(elem, attr, None)
    return None if value is None else float(value)


def resolve_flux_offsets(node: QualibrationNode[Parameters, Quam]):
    """Per-measured-qubit vertical-line offsets for the plots.

    z_source None -> each qubit's own z-line idle offset on its own plot.
    z_source set  -> the single source (qubit z-line or coupler) idle offset on every plot."""
    measured = node.namespace["qubits"]
    z_source_name = node.parameters.z_source
    if z_source_name is None:
        return {q.name: current_flux_offset(q.z) for q in measured if q.z is not None}
    if z_source_name in node.machine.qubits:
        source = node.machine.qubits[z_source_name].z
    elif z_source_name in node.machine.qubit_pairs:
        source = node.machine.qubit_pairs[z_source_name].coupler
    else:
        return {}
    offset = current_flux_offset(source)
    return {q.name: offset for q in measured}


@node.run_action(skip_if=node.parameters.simulate or not node.parameters.plot)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """One combined figure per qubit: the 2-D |IQ| raw map, the per-flux fitted
    resonator centres, the dispersive centre-frequency(flux) fit curve (qubit/None
    source), a vertical line at the flux source's operating offset, and a horizontal
    line at each resonator's current readout frequency."""
    z_source_name = node.parameters.z_source
    offset_label = "self idle offset" if z_source_name is None else f"{z_source_name} offset"
    readout_freqs = {
        q.name: float(q.resonator.RF_frequency)
        for q in node.namespace["qubits"]
        if q.resonator is not None
    }
    node.results["figures"] = plot_combined(
        node.namespace["sep_results"],
        node.namespace["dispersion_sep"],
        flux_offsets=resolve_flux_offsets(node),
        flux_offset_label=offset_label,
        readout_freqs=readout_freqs,
    )


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the robust dispersive-fit outputs to the QUAM state: the idle
    (sweet-spot) flux offset, the minimum-frequency flux point, the
    resonator readout frequency at that point, and the flux period (phi0).

    Only the degeneracy-independent quantities are written; g / f_q_max stay out
    of the state (they are conditional until a spectroscopy prior is supplied).

    When a single external qubit drives the sweep (z_source set to a qubit), only
    that qubit measures its own resonator-vs-its-own-flux; every other measured
    resonator is crosstalk from the source's flux line, so its z offsets / readout
    frequency must NOT be updated from this run. When z_source is a coupler the whole
    writeback is skipped (the dispersive fit is not run)."""
    z_src = node.parameters.z_source
    # A coupler flux source has no qubit-owned dispersive fit to write back; defer.
    if z_src is not None and z_src in node.machine.qubit_pairs:
        node.log(f"Skipping state update: z_source={z_src} is a coupler (writeback deferred)")
        return
    pass
    # with node.record_state_updates():
    #     for q in node.namespace["qubits"]:
    #         if q.z is None or node.outcomes[q.name] == "failed":
    #             continue
    #         # Skip crosstalk qubits: only the flux-source qubit's fit is a valid
    #         # self-flux calibration (when z_source is None, every qubit fluxes
    #         # itself, so none are skipped).
    #         if z_src is not None and q.name != z_src:
    #             node.log(f"Skipping state update for {q.name}: crosstalk under z_source={z_src}")
    #             continue

    #         disp = node.results["dispersion_results"][q.name]

    #         # Idle flux offset — the flux of maximum qubit (and resonator) frequency.
    #         if q.z.flux_point == "independent":
    #             q.z.independent_offset = disp["sweet_spot_flux"]
    #         else:
    #             q.z.joint_offset = disp["sweet_spot_flux"]
    #         # Minimum-frequency flux point (half a period from the sweet-spot flux).
    #         if node.parameters.update_flux_min:
    #             q.z.min_offset = disp["min_offset"]
    #         # Resonator readout frequency at the sweet spot (absolute).
    #         q.resonator.f_01 = disp["sweet_spot_res"]
    #         q.resonator.RF_frequency = disp["sweet_spot_res"]
    #         # Flux quantum in voltage / current.
    #         q.phi0_voltage = disp["dv_phi0"]
    #         q.phi0_current = disp["phi0_current"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
