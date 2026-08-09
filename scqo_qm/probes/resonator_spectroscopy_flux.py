"""Resonator-spectroscopy-vs-flux acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Sweep a flux bias and the readout frequency and read out each qubit's resonator; the resonator dip
is fitted flux-by-flux downstream to give a centre-frequency(flux) trace. Optional single flux
source (`z_source`): a qubit name (its z-line), a qubit-pair name (its tunable coupler), or None
(each measured qubit fluxes itself, == 02c).
"""

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import check_flux_bias_absolute


def build_program(
    machine,
    qubits,
    *,
    dcs,
    dfs,
    num_shots: int,
    z_source: Optional[str] = None,
):
    """Build the resonator-spectroscopy-vs-flux QUA program. Returns (program, sweep_axes).

    `dcs` is the flux-bias sweep (V), `dfs` the readout-detuning sweep (Hz); `num_shots` is the
    innermost averaging count. `qubits` is a BatchableList (see `_lib.select_qubits`). `z_source`
    (a qubit name, a qubit-pair name, or None) selects the single flux source.
    """
    u = unit(coerce_to_integer=True)
    num_qubits = len(qubits)

    # Resolve the single flux source (if any). It may be a qubit (its z-line) or
    # a qubit pair (its tunable coupler). Both FluxLine and TunableCoupler expose
    # set_dc_offset(dc) + settle(), so the sweep code below is identical for either.
    # When None, each measured qubit drives its own z-line (identical to 02c).
    if z_source is None:
        flux_driver = None
    elif z_source in machine.qubits:
        flux_driver = machine.qubits[z_source].z
    elif z_source in machine.qubit_pairs:
        flux_driver = machine.qubit_pairs[z_source].coupler
    else:
        raise ValueError(f"z_source={z_source!r} is neither a qubit nor a qubit-pair name")

    # ABSOLUTE frame: `set_dc_offset(dc)` REPLACES the standing bias, so the swept
    # values ARE the line voltage and no idle term is added. Past the port's full
    # scale the DAC clips and the simulator shows nothing.
    if flux_driver is not None:
        check_flux_bias_absolute(flux_driver, name=str(z_source), bias_v=dcs)
    else:
        for qubit in qubits:
            z = getattr(qubit, "z", None)
            if z is None:
                raise ValueError(
                    f"{qubit.name}: no flux line, but this probe biases every measured "
                    f"qubit's own z line - it cannot silently skip one and still report "
                    f"a flux axis in volts")
            check_flux_bias_absolute(z, name=qubit.name, bias_v=dcs)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "flux_bias": xr.DataArray(dcs, attrs={"long_name": "flux bias", "units": "V"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "readout frequency", "units": "Hz"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        dc = declare(fixed)  # QUA variable for the flux bias
        df = declare(int)  # QUA variable for the readout frequency detuning
        idx = declare(int)  # progress index over the outer flux loop

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            assign(idx, 0)
            with for_(*from_array(dc, dcs)):
                # Save the flux-point counter for the progress bar
                save(idx, n_st)
                assign(idx, idx + 1)
                # Apply the flux: either from the single source (qubit z-line or
                # coupler), or per-qubit (== 02c).
                if flux_driver is None:
                    for i, qubit in multiplexed_qubits.items():
                        qubit.z.set_dc_offset(dc)
                        qubit.z.settle()
                else:
                    flux_driver.set_dc_offset(dc)
                    flux_driver.settle()
                align()

                # Read out every measured qubit's resonator at this flux bias.
                for i, qubit in multiplexed_qubits.items():
                    rr = qubit.resonator
                    with for_(*from_array(df, dfs)):
                        # Update the resonator frequencies for resonator
                        rr.update_frequency(df + rr.intermediate_frequency)
                        # Average innermost: repeat the measurement num_shots times per point
                        with for_(n, 0, n < num_shots, n + 1):
                            # readout the resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            # wait for the resonator to deplete
                            rr.wait(rr.depletion_time * u.ns)
                            # save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Average the innermost num_shots shots, then buffer detuning then flux
                I_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(dfs)).buffer(len(dcs)).save(f"I{i + 1}")
                Q_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(dfs)).buffer(len(dcs)).save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    Here `num_shots` is only the progress-bar total; pass the number of flux points, since the
    program's saved "n" stream counts the outer flux loop, not shots.
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)
