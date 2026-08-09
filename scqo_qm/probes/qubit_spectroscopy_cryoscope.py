"""Long-time (spectroscopy) cryoscope probe: vendor code only (qm/quam) — no qualibrate, no scqo.

Flux-line step response by qubit spectroscopy vs wait-time INTO a parked flux
pulse. Sequence per (detuning, wait): reset -> set the drive near the parked
qubit frequency (``df + IF + center_offset_hz``) -> play a long flux ``const``
pulse -> wait ``t`` into it -> play a long, weak ``saturation`` spectroscopy pulse
-> readout. Per wait-time the spectroscopy peak center gives the qubit frequency,
which maps to the delivered flux; the flux settling is fit downstream to a sum of
exponentials. This reaches the microsecond tails the Ramsey cryoscope cannot.

SPECTROSCOPY DRIVE: the tone is the square ``saturation`` op stretched to
``operation_len_ns`` (on the 4 ns grid), played at an amplitude that holds the
calibrated x180 ROTATION AREA (peak amplitude x length x envelope average/peak —
the envelope factor matters: see :func:`resolve_drive_scale`), so a longer pulse
is a proportionally weaker one and the tone stays a true pi pulse. The line is
then Fourier-limited (FWHM ~ 0.8 / operation_len_ns) with little power broadening
— a bare x180 (16 ns) gives a tens-of-MHz line that a few-MHz settling tail
cannot be tracked against. The flux ``const`` is held for the wait PLUS this
drive length plus a fixed tail, so it always fully brackets the drive.

PULSE CONTRACT: ``flux_amp_v`` is the parked flux amplitude in VOLTS measured
from the standing DC bias ``initialize_qpu`` applies — the z ``const`` rides on
that offset, so it is idle-relative. ``flux_point`` is passed EXPLICITLY because
the DAC emits ``idle + excursion`` and the number validated against the rail must
be the number that plays.

CENTER OFFSET: ``center_offset_hz`` is the arch-predicted parked detuning the
scqo experiment computes; the drive is shifted by it (NO LO shift — band-limited)
so the spectroscopy window stays centered on the peak. It is a plain frequency
shift added to ``update_frequency``.

Single qubit only: the sequence is built per qubit (one parked drive, one flux
line, one wait axis); run targets one at a time.
"""

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array

from scqo_qm.experiments._amp_limits import check_amp_scale_window
from scqo_qm.experiments._flux_limits import check_flux_pulse_relative, idle_offset_v
from scqo_qm.experiments._lib import acquire as _acquire

#: extra flux-pulse cycles held past the spectroscopy pulse so the drive sits
#: fully inside the parked flux (4 ns cycles).
_FLUX_TAIL_CYCLES = 25


def validate_inputs(qubits, flux_amp_v: float, flux_point: str) -> float:
    """Pure pre-flight checks (no QUA); return the flux ``const`` reference amplitude.

    Refuses, by name and before any hardware time: more than one target (the
    sequence is built per qubit); a missing flux line; and a parked flux whose
    ``idle + flux_amp_v`` clips the port or needs an ``amplitude_scale`` QUA cannot
    express (:func:`check_flux_pulse_relative`).
    """
    if len(qubits) != 1:
        names = list(qubits.get_names())
        raise ValueError(
            f"qubit_spectroscopy_cryoscope builds its spectroscopy sequence per "
            f"qubit and supports one target at a time, got {len(qubits)}: {names}. "
            f"Run them one at a time.")
    qubit = qubits[0]
    z = getattr(qubit, "z", None)
    if z is None:
        raise ValueError(
            f"{qubit.name}: no flux line, but the spectroscopy cryoscope parks a z "
            f"pulse to detune the qubit — it cannot run on a qubit with no z channel.")
    return check_flux_pulse_relative(
        z, name=f"{qubit.name} spectroscopy cryoscope flux pulse",
        idle_v=idle_offset_v(z, flux_point), amps_v=[float(flux_amp_v)])


def _area_ratio(pulse, default: float) -> float:
    """``average/peak`` of the pulse's I-quadrature envelope — the rotation area
    a pulse delivers per unit (peak amplitude x length). 1.0 for a square pulse,
    0.5 for the lab's (Drag)Cosine x180. Sampled from the pulse's own
    ``waveform_function()`` so a different envelope stays correct; ``default``
    covers objects that cannot render samples (bare test stubs).
    """
    try:
        wf = np.asarray(pulse.waveform_function())
    except Exception:
        return default
    env = np.abs(np.real(wf)) if np.iscomplexobj(wf) else np.abs(wf)
    peak = float(np.max(env))
    if not np.isfinite(peak) or peak == 0.0:
        return default
    return float(np.mean(env) / peak)


def resolve_drive_scale(qubit, *, operation: str, operation_len_ns: float,
                        operation_amp: float = 1.0) -> float:
    """Amplitude scale for the spectroscopy tone that holds the calibrated x180
    ROTATION AREA (peak amplitude x length x envelope average/peak) over
    ``operation_len_ns``, times an extra ``operation_amp`` multiplier.

    The envelope factor is load-bearing: the x180 is a cosine-envelope DRAG pulse
    that delivers only HALF its peak x length, so transplanting the bare product
    onto the square saturation op doubles the area — an exact 2*pi pulse, whose
    spectroscopy response is DARK on resonance with two bright shoulders at
    ``+/- sqrt(1.25)/operation_len_ns`` (seen on hardware as a split line). With
    the ratio in, the tone is a true pi pulse: full contrast on resonance and a
    Fourier-limited line (FWHM ~ 0.8 / operation_len_ns), and holding the area
    means a longer pulse is a proportionally weaker one (no power broadening).
    Pure (no QUA), so the arithmetic is unit-tested; refuses BY NAME a scale QUA's
    ``amp()`` cannot express (``|scale| >= 2``).
    """
    x180 = qubit.xy.operations["x180"]
    drive = qubit.xy.operations[operation]
    # sample-less stubs fall back to the lab shapes: cosine x180, square drive.
    area_scale = float(operation_amp) * (
        _area_ratio(x180, default=0.5) * x180.amplitude * x180.length
    ) / (_area_ratio(drive, default=1.0) * drive.amplitude * float(operation_len_ns))
    check_amp_scale_window(
        [area_scale], name=f"{qubit.name} spectroscopy cryoscope drive",
        knob="operation_amp")
    return area_scale


def build_program(
    machine,
    qubits,
    *,
    dfs,
    wait_cycles,
    flux_amp_v: float,
    center_offset_hz: float,
    num_shots: int,
    reset_type: str = "thermal",
    operation: str = "saturation",
    operation_len_ns: int = 400,
    operation_amp: float = 1.0,
    use_state_discrimination: bool = False,
    simulate: bool = False,
    flux_point: str = "joint",
    log: Optional[Callable] = None,
):
    """Build the long-time cryoscope QUA program. Returns ``(program, sweep_axes)``.

    ``dfs`` is the drive-detuning sweep (Hz), swept OUTER; ``wait_cycles`` the
    wait-into-the-flux sweep in clock cycles (4 ns), swept INNER (typically
    log-spaced). ``flux_amp_v`` is the idle-relative parked amplitude and
    ``center_offset_hz`` the arch-predicted parked detuning the drive is shifted
    by. ``operation`` (default ``saturation``) is stretched to ``operation_len_ns``
    and played at an amplitude holding the x180 pulse area, times the extra
    ``operation_amp`` multiplier. No baking — a plain stretched ``const`` plus the
    stretched spectroscopy pulse — so the shared ``_lib.acquire`` fetches it (the
    adapter returns ``(prog, axes)``).
    """
    amp_ref = validate_inputs(qubits, flux_amp_v, flux_point)
    qubit = qubits[0]
    dfs = np.round(np.asarray(dfs)).astype(int)
    wait_cycles = np.unique(np.asarray(wait_cycles).astype(int))
    const_scale = float(flux_amp_v) / amp_ref
    offset_hz = int(round(float(center_offset_hz)))
    # Long, weak spectroscopy tone: stretch the square `saturation` op to
    # operation_len_ns (the flux `const` below brackets it) and scale its amplitude
    # to hold the calibrated x180 pulse area (see resolve_drive_scale).
    op_len_cycles = int(operation_len_ns) // 4
    area_scale = resolve_drive_scale(
        qubit, operation=operation, operation_len_ns=operation_len_ns,
        operation_amp=operation_amp)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning_hz": xr.DataArray(
            dfs.astype(float), attrs={"long_name": "drive detuning", "units": "Hz"}),
        "wait_time_ns": xr.DataArray(
            (4 * wait_cycles).astype(float),
            attrs={"long_name": "wait into the flux pulse", "units": "ns"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        df = declare(int)
        t = declare(int)
        if use_state_discrimination:
            state = declare(int)
            state_st = declare_stream()

        machine.initialize_qpu(target=qubit, flux_point=flux_point)
        align()

        with for_(n, 0, n < num_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(df, dfs)):
                with for_each_(t, wait_cycles):
                    qubit.reset(reset_type, simulate, log_callable=log)
                    # park the drive at the arch-predicted detuned frequency + the swept df
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + offset_hz)
                    align()
                    # hold the flux on while the drive waits t into it, then probes
                    qubit.z.play("const", amplitude_scale=const_scale,
                                 duration=t + op_len_cycles + _FLUX_TAIL_CYCLES)
                    qubit.xy.wait(t)
                    qubit.xy.play(operation, amplitude_scale=area_scale,
                                  duration=op_len_cycles)
                    align()
                    if use_state_discrimination:
                        qubit.readout_state(state)
                        save(state, state_st)
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                        save(I[0], I_st[0])
                        save(Q[0], Q_st[0])

        with stream_processing():
            n_st.save("n")
            if use_state_discrimination:
                state_st.buffer(len(wait_cycles)).buffer(len(dfs)).average().save("state1")
            else:
                I_st[0].buffer(len(wait_cycles)).buffer(len(dfs)).average().save("I1")
                Q_st[0].buffer(len(wait_cycles)).buffer(len(dfs)).average().save("Q1")

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
