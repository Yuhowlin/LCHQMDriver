"""Parametric-drive resonance-map acquisition probe: vendor code only (qm/quam) —
no qualibrate, no scqo, no scqat.

Prepare the qubit (x180), then MODULATE its own flux (z) line with an RF tone —
the z element's oscillator is stepped to the swept drive frequency with
``update_frequency``, the IF phase is reset for shot-to-shot coherence, and the
z ``const`` operation is played at the swept absolute amplitude for a FIXED,
caller-given driving time — then read the qubit out. Where the modulation
frequency matches a sideband condition with a coupled component the excitation
parametrically transfers out of the qubit; the 2D population map over
(amplitude, frequency) draws the resonance line(s). Same flux-line technique as
:class:`scqo_qm.components.macros.parametric_reset_macro.ParametricReset`.

Revived from the retired ``LCH_qubit_parametric_drive_fixed_time`` probe (git
``cc04cec^``), modernized for scqo: EVERY selected qubit is driven (the node
drove only the first of the batch), the sweep is absolute volts only (the
rail + amplitude_scale + idle-sum guard is ``check_flux_pulse_relative``), and
the axes carry the canonical scqo names.

THE Z-LINE OSCILLATOR — why this probe carries its own config. quam's
``FluxLine`` ships ``intermediate_frequency=None``, so the generated config's z
elements carry NO oscillator and a program that ``update_frequency``'s them is
refused at compile. :func:`ensure_flux_oscillators` patches the GENERATED
CONFIG DICT (never the QUAM tree — a z IF leaked into the live tree would
modulate every later flux pulse in the session), and ``probe()`` returns the
3-tuple ``(prog, sweep_axes, acquire)`` with the patched config bound into the
acquire callable — the backend's shared fetch path regenerates a config and
would lose the patch. A tree that already declares a z IF is respected
(``setdefault``); the seed value is irrelevant, since every sweep iteration
sets the frequency itself.

``--preview`` note: the shell builds a standalone program, so preview renders
the script normally — but the script (and any gateway simulation) uses the
UNPATCHED config, so the waveform simulation may refuse the missing z
oscillator and degrade to script-only with a PreviewWarning.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.loops import from_array

from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import (
    check_flux_pulse_relative,
    declared_idle_offset_v,
)

#: settle gap (ns) between the pi pulse and the parametric tone, so the drive
#: pulse rings down before the flux modulation starts (the retired node's 200).
_PREP_SETTLE_NS = 200


def drive_time_cycles(drive_time_ns: int) -> int:
    """The fixed driving time as QUA clock cycles — refused by name off-grid.

    ``play(duration=...)`` counts 4 ns cycles and the shortest playable pulse
    is 16 ns, so a time that is not a positive multiple of 4 (or is shorter
    than 16 ns) would be silently truncated by integer division — refuse it
    instead. Pure: pinned by ``tests/test_parametric_drive_probe.py``."""
    t = int(drive_time_ns)
    if t < 16 or t % 4 != 0:
        raise ValueError(
            f"drive_time_ns must be a multiple of 4 ns and >= 16 ns on the QM "
            f"backend (play() counts 4 ns cycles; 16 ns is the shortest "
            f"pulse), got {drive_time_ns}.")
    return t // 4


def ensure_flux_oscillators(config: dict, elements, seed_hz: float) -> dict:
    """Give each named z ELEMENT of a generated config an oscillator.

    Adds ``intermediate_frequency: seed_hz`` to every named element that does
    not already declare one (``setdefault`` — a tree-declared IF wins), so the
    program's ``update_frequency`` compiles. Mutates and returns ``config``;
    the QUAM tree itself is never touched. Pure dict surgery: pinned by
    ``tests/test_parametric_drive_probe.py``."""
    for name in elements:
        if name not in config.get("elements", {}):
            raise ValueError(
                f"element {name!r} is not in the generated config; cannot "
                f"attach the parametric-drive oscillator")
        config["elements"][name].setdefault("intermediate_frequency", float(seed_hz))
    return config


def build_program(
    machine,
    qubits,
    *,
    amps_v,
    freqs_hz,
    drive_time_ns: int,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the parametric-drive (fixed-time) QUA program. Returns (program, sweep_axes).

    ``amps_v`` is the drive-amplitude sweep in ABSOLUTE volts — a z-pulse
    excursion riding on the standing bias ``initialize_qpu`` applied, validated
    per qubit by ``check_flux_pulse_relative`` (rail, amplitude_scale range and
    the idle + excursion sum). ``freqs_hz`` is the drive-frequency sweep in
    integer Hz (``update_frequency`` steps of 1 Hz). ``qubits`` is a
    BatchableList (see ``_lib.select_qubits``).
    """
    amps_v = np.asarray(amps_v, dtype=float)
    freqs_hz = np.asarray(freqs_hz).astype(int)
    cycles = drive_time_cycles(drive_time_ns)
    num_qubits = len(qubits)

    # Per-qubit volts -> amplitude_scale denominators; the guard refuses a
    # missing/zero/over-rail `const`, an inexpressible scale, and a window
    # that clips once it rides on the standing bias.
    refs = {}
    for qubit in qubits:
        if getattr(qubit, "z", None) is None:
            raise ValueError(
                f"{qubit.name} has no z line; nothing to play the parametric "
                f"drive on.")
        refs[qubit.name] = check_flux_pulse_relative(
            qubit.z, name=f"{qubit.name} parametric drive on {qubit.name}.z",
            idle_v=declared_idle_offset_v(qubit.z), amps_v=amps_v,
            operation="const")

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        # Outer loop -> y axis.
        "parametric_amp_v": xr.DataArray(
            amps_v, attrs={"long_name": "parametric drive amplitude", "units": "V"}),
        # Inner loop -> x axis.
        "parametric_freq_hz": xr.DataArray(
            freqs_hz, attrs={"long_name": "parametric drive frequency", "units": "Hz"}),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        a = declare(fixed)  # swept drive amplitude (volts)
        f_drive = declare(int)  # swept drive frequency (Hz, the z-line IF)
        if use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (the standing bias the
            # parametric excursion rides on).
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                # Amplitude loop (outer -> y axis)
                with for_(*from_array(a, amps_v)):
                    # Frequency loop (inner -> x axis)
                    with for_(*from_array(f_drive, freqs_hz)):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(reset_type, simulate, log_callable=log)
                            # Step the z oscillator to the swept drive frequency.
                            qubit.z.update_frequency(f_drive)

                        # State prep: excite every driven qubit to |1>.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                        align()
                        wait(_PREP_SETTLE_NS // 4)

                        # The parametric tone: swept volts on each qubit's own
                        # z line for the fixed driving time. reset_if_phase
                        # keeps the modulation phase shot-to-shot coherent.
                        for i, qubit in multiplexed_qubits.items():
                            qubit.z.reset_if_phase()
                            qubit.z.play(
                                "const",
                                amplitude_scale=a * (1.0 / refs[qubit.name]),
                                duration=cycles,
                            )
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

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Inner buffer = frequency (x), next buffer = amplitude (y).
                if use_state_discrimination:
                    state_st[i].buffer(len(freqs_hz)).buffer(len(amps_v)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(freqs_hz)).buffer(len(amps_v)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(freqs_hz)).buffer(len(amps_v)).average().save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float,
    log: Optional[Callable] = None,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Connect to the QOP, execute the program and fetch the raw xr.Dataset.

    Pass the oscillator-patched config from ``probe()`` as ``config``; the
    shared execute-and-fetch helper would otherwise regenerate a config whose z
    elements carry no oscillator and the compile would refuse
    ``update_frequency``.
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots,
                    timeout=timeout, log=log, config=config)


from functools import partial
from typing import Any

import numpy as np

from scqo import register
from scqo.experiments import QubitParametricDrive


@register
class QMQubitParametricDrive(QubitParametricDrive):
    """Build a multiplexed parametric-drive map QUA program on the QM OPX.

    Returns the 3-tuple probe shape ``(prog, sweep_axes, acquire)`` — not for
    heterogeneous streams (the fetch is the shared one) but to bind the
    oscillator-patched config into the acquire callable (module docstring)."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        # update_frequency steps in whole Hz; re-declare the canonical axis as
        # what actually plays (the >= MHz-scale windows make rounding lossless).
        freqs = np.round(np.asarray(
            self.sweep_axes["parametric_freq_hz"], dtype=float)).astype(int)
        self.sweep_axes["parametric_freq_hz"] = freqs.astype(float)

        prog, sweep_axes = build_program(
            machine,
            qubits,
            amps_v=np.asarray(self.sweep_axes["parametric_amp_v"], dtype=float),
            freqs_hz=freqs,
            drive_time_ns=int(self.params.drive_time_ns),
            num_shots=int(self.params.num_averages),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
        )
        config = ensure_flux_oscillators(
            machine.generate_config(),
            [qubit.z.name for qubit in qubits],
            float(freqs[0]),
        )
        return prog, sweep_axes, partial(acquire, config=config)
