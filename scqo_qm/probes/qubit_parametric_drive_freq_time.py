"""Parametric-drive decoherence (freq x time) acquisition probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Prepare the qubit, apply a fixed-amplitude parametric (flux-line) drive while sweeping the drive
frequency and duration, then read out. With `tomography=False` it reads out rho_11 directly; with
`tomography=True` it sweeps an extra X/Y/Z readout-basis axis for full single-qubit tomography. The
non-Markovian amplitude-damping model is fitted downstream.
"""

from typing import Callable, Optional

import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._amp_limits import MAX_AMP_SCALE
from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, declared_idle_offset_v

# MAX_AMP_SCALE is still needed here for the PREFACTOR branch, whose drive_amp IS
# an amplitude_scale and so never reaches the volts-based helpers.


def build_program(
    machine,
    qubits,
    *,
    freqs,
    time_tick,
    drive_amp: float,
    amp_mode: str,
    prepare_state: str,
    tomography: bool,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the parametric-drive decoherence (freq x time) QUA program. Returns (program, sweep_axes).

    `freqs` is the drive-frequency sweep (Hz), `time_tick` the drive-duration sweep in clock cycles
    (4 ns). `drive_amp` is fixed (volts if `amp_mode == "absolute"`, else a unitless prefactor).
    `qubits` is a BatchableList (see `_lib.select_qubits`). Raises ValueError if the emitted
    `amplitude_scale` would leave QUA's (-2, 2) fixed-point range.
    """
    num_qubits = len(qubits)
    time_ns = time_tick * 4  # in ns

    # In "absolute" mode drive_amp is a z-pulse excursion in volts on the standing bias
    # initialize_qpu applied (no flux_point argument here, so the declaration runs):
    # the shared helper does the `const` contract, QUA's amplitude_scale bound and the
    # idle + excursion sum, which nothing checked before.
    if amp_mode == "absolute":
        for qubit in qubits:
            check_flux_pulse_relative(
                qubit.z, name=qubit.name, idle_v=declared_idle_offset_v(qubit.z),
                amps_v=[drive_amp], operation="const")
    else:  # prefactor
        if abs(drive_amp) >= MAX_AMP_SCALE:
            raise ValueError(
                f"Prefactor drive_amp exceeds QUA's amplitude_scale range: "
                f"|a| = {abs(drive_amp):.3f} >= {MAX_AMP_SCALE}."
            )

    # X/Y/Z readout-basis axis — only swept when tomography is enabled.
    readout_basis_array = [0, 1, 2]

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "driving_frequency": xr.DataArray(freqs, attrs={"long_name": "driving frequency", "units": "Hz"}),
        "driving_time": xr.DataArray(time_ns, attrs={"long_name": "driving time", "units": "ns"}),
    }
    if tomography:
        sweep_axes["basis"] = xr.DataArray(
            readout_basis_array, attrs={"long_name": "basis for state tomography"}
        )

    with program() as prog:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        tt = declare(int)  # QUA variable for the driving time
        f_drive = declare(int)  # QUA variable for the driving frequency
        rbi = declare(int)  # readout-basis index (used only when tomography is on)

        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        def measure_shot(multiplexed_qubits, rbi_var):
            """One prepare -> parametric-drive -> (optional basis rotation) -> readout
            shot. When ``rbi_var`` is given, the readout basis is selected by it
            (X: -y90, Y: x90, Z: identity) for state tomography."""
            for i, qubit in multiplexed_qubits.items():
                qubit.reset(reset_type, simulate, log_callable=log)
                # Update the qubit frequency
                qubit.z.update_frequency(f_drive)

            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play(prepare_state)
            align()
            wait(200 // 4)
            for i, qubit in multiplexed_qubits.items():
                if i == 0:
                    qubit.z.reset_if_phase()
                    ref = float(qubit.z.operations["const"].amplitude)
                    amp_scale = drive_amp / ref if amp_mode == "absolute" else drive_amp
                    qubit.z.play("const", amplitude_scale=amp_scale, duration=tt)
            align()

            if rbi_var is not None:
                # Rotate into the measured basis before readout.
                for i, qubit in multiplexed_qubits.items():
                    with switch_(rbi_var):
                        with case_(0):
                            qubit.xy.play("-y90")
                        with case_(1):
                            qubit.xy.play("x90")
                        with case_(2):
                            qubit.xy.play("y90", amplitude_scale=0)
                align()

            for i, qubit in multiplexed_qubits.items():
                if use_state_discrimination:
                    qubit.readout_state(state[i])
                    save(state[i], state_st[i])
                else:
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            align()

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(f_drive, freqs)):
                    with for_(*from_array(tt, time_tick)):
                        if tomography:
                            with for_each_(rbi, readout_basis_array):
                                measure_shot(multiplexed_qubits, rbi)
                        else:
                            measure_shot(multiplexed_qubits, None)

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                if use_state_discrimination:
                    stream = state_st[i].buffer(len(readout_basis_array)) if tomography else state_st[i]
                    stream.buffer(len(time_tick)).buffer(len(freqs)).average().save(f"state{i + 1}")
                else:
                    i_stream = I_st[i].buffer(len(readout_basis_array)) if tomography else I_st[i]
                    q_stream = Q_st[i].buffer(len(readout_basis_array)) if tomography else Q_st[i]
                    i_stream.buffer(len(time_tick)).buffer(len(freqs)).average().save(f"I{i + 1}")
                    q_stream.buffer(len(time_tick)).buffer(len(freqs)).average().save(f"Q{i + 1}")

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
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset."""
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log)
