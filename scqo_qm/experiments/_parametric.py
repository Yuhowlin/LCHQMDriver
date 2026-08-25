"""Shared parametric-drive machinery: vendor code only (qm/quam) — no qualibrate,
no scqo experiment logic, no scqat.

Both parametric-drive probes MODULATE a qubit's own flux (z) line with an RF
tone — ``qubit_parametric_drive_amp`` sweeps (amplitude, frequency) at a fixed
driving time, ``qubit_parametric_drive_time`` sweeps (frequency, driving time)
at a fixed amplitude. Everything they share lives here: the settle gap, the
ns->cycles guards, the generated-config oscillator patch, the config-forwarding
``acquire`` and the preview hook.

THE Z-LINE OSCILLATOR — why these probes carry their own config. quam's
``FluxLine`` ships ``intermediate_frequency=None``, so the generated config's z
elements carry NO oscillator and a program that ``update_frequency``'s them is
refused at compile. :func:`ensure_flux_oscillators` patches the GENERATED CONFIG
DICT (never the QUAM tree — a z IF leaked into the live tree would modulate every
later flux pulse in the session), and each ``probe()`` returns the 3-tuple
``(prog, sweep_axes, acquire)`` with the patched config bound into the acquire
callable — the backend's shared fetch path regenerates a config and would lose
the patch. A tree that already declares a z IF is respected (``setdefault``); the
seed value is irrelevant, since every sweep iteration sets the frequency itself.

``--preview`` renders fully: each shell builds a standalone program, and the
backend's ``patch_preview_config`` hook (supplied here by
:class:`FluxOscillatorPreviewMixin`) applies the same oscillator patch to the
preview's config, so the dumped script and the gateway waveform simulation
compile against what a real run would execute. (Confirmed live 2026-08-20:
without the patch the gateway refuses with "Can not change the intermediate
frequency of quantum Element q1.z because its' initial value was none".)
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import xarray as xr

from scqo_qm.experiments._lib import acquire as _acquire

#: settle gap (ns) between the pi pulse and the parametric tone, so the drive
#: pulse rings down before the flux modulation starts (the retired node's 200).
PREP_SETTLE_NS = 200


def drive_time_cycles(drive_time_ns) -> int:
    """One driving time as QUA clock cycles — refused by name off-grid.

    ``play(duration=...)`` counts 4 ns cycles and the shortest playable pulse
    is 16 ns, so a time that is not a positive multiple of 4 (or is shorter
    than 16 ns) would be silently truncated by integer division — refuse it
    instead. Pure: pinned by ``tests/test_parametric_drive_probes.py``."""
    t = int(drive_time_ns)
    if t < 16 or t % 4 != 0:
        raise ValueError(
            f"drive_time_ns must be a multiple of 4 ns and >= 16 ns on the QM "
            f"backend (play() counts 4 ns cycles; 16 ns is the shortest "
            f"pulse), got {drive_time_ns}.")
    return t // 4


def drive_time_cycles_array(times_ns) -> np.ndarray:
    """A whole SWEPT driving-time axis as QUA clock cycles.

    Every point goes through :func:`drive_time_cycles`, so one off-grid value
    refuses the run by name before any QUA is built — the neutral axis is
    already built on the 4 ns grid (scqo's ``time_axis_ns(..., grid_ns=4)``),
    so a violation here means the axis was tampered with, not rounded.
    Deliberately does NOT de-duplicate: an exact-grid axis has no collisions to
    absorb, and quietly shortening a sweep axis would desynchronize it from the
    stream buffers."""
    times = np.asarray(times_ns).ravel()
    if times.size == 0:
        raise ValueError("the drive-time axis is empty; nothing to sweep.")
    return np.array([drive_time_cycles(t) for t in times], dtype=int)


def ensure_flux_oscillators(config: dict, elements, seed_hz: float) -> dict:
    """Give each named z ELEMENT of a generated config an oscillator.

    Adds ``intermediate_frequency: seed_hz`` to every named element that does
    not already declare one (``setdefault`` — a tree-declared IF wins), so the
    program's ``update_frequency`` compiles. Mutates and returns ``config``;
    the QUAM tree itself is never touched. Pure dict surgery: pinned by
    ``tests/test_parametric_drive_probes.py``."""
    for name in elements:
        if name not in config.get("elements", {}):
            raise ValueError(
                f"element {name!r} is not in the generated config; cannot "
                f"attach the parametric-drive oscillator")
        config["elements"][name].setdefault("intermediate_frequency", float(seed_hz))
    return config


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


class FluxOscillatorPreviewMixin:
    """The backend's ``patch_preview_config`` hook for both parametric shells.

    The generic hook in ``QMBackend.preview`` is duck-typed and name-agnostic:
    a shell whose program only compiles against an AMENDED config exposes this
    method, and the dumped script plus the gateway simulation then see the same
    config the real run would execute against (module docstring).

    Mixed in BEFORE the scqo experiment class so it wins the MRO.
    """

    #: the sweep axis holding the drive frequency, whose first point seeds the
    #: patch. Both shells name it the same; kept a hook rather than a literal so
    #: a future sibling with a different axis name only overrides one line.
    parametric_freq_axis: str = "parametric_freq_hz"

    def patch_preview_config(self, config: dict) -> dict:
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine  # type: ignore[attr-defined]
        qubits = select_qubits(machine, self.params.targets,  # type: ignore[attr-defined]
                               multiplexed=True)
        seed = float(np.round(np.asarray(
            self.sweep_axes[self.parametric_freq_axis], dtype=float)[0]))  # type: ignore[attr-defined]
        return ensure_flux_oscillators(
            config, [qubit.z.name for qubit in qubits], seed)
