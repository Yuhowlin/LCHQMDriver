"""Single-excitation flux-chevron acquisition probe: vendor code only (qm/quam) -
no qualibrate, no scqo, no scqat.

Adapted from `calibrations/19_chevron_11_02.py` (the two-qubit CZ chevron). The
flux pulse still sweeps amplitude x duration on the control qubit's z line and
both qubits are still read out, but only **one** qubit of the pair is excited with
`x180` (selected by `drive_role`, default the control qubit) instead of preparing
|11>. There is no fit/state-writeback downstream; the node renders a 2D color map.

With state discrimination both qubits are read out 2-level and the saved data is the
joint two-qubit populations P00, P01, P10, P11 (first digit = control, second =
target) as variables `state_gg/state_ge/state_eg/state_ee` -- not the independent
single-qubit averages. Without state discrimination the raw I/Q of each qubit is saved.

The sub-4ns flux-pulse granularity uses the baking tool (`qualang_tools.bakery`):
short segments (1..16 ns) are baked into the config, and longer pulses combine a
baked tail with a dynamically stretched (multiple-of-4ns) `play`.

Amplitude sweep (`amp_mode`), mirroring `pair_swap_flux_map`:
  - "prefactor" (default, what the qualibrate node uses): the sweep values are a
    unitless pre-factor on a base amplitude computed from QUAM as the |11>-|02>
    resonance point of each pair.
  - "absolute": the sweep values ARE the emitted flux-pulse amplitudes in volts.
    The QUAM |11>-|02> formula is not consulted at all (it is meaningless when the
    caller names volts, and a bring-up tree may not carry the fields it reads).

`flux_role` selects which qubit's z line carries the flux pulse; it defaults to
the control qubit, which is what this probe used to hardwire.

QM single-excitation swap chevron for scqo — supplies ``probe()``.

Parameters, the record-only map summary and the (absent) writeback are
inherited from ``scqo.experiments.PairSwapChevron``; the joint-population
reduction comes from :class:`JointPopulationMixin`. scqo sweeps
``(flux_amp_v, swap_time_ns)`` in absolute volts and whole nanoseconds; the
LCHQM probe sweeps a QUA amplitude scale x a 1 ns-granular pulse duration
(baked below 4 ns), so this adapter drives the probe in its ``amp_mode
="absolute"`` mode and maps the neutral high/low roles onto the vendor's
control/target.

Unlike every other QM adapter here, ``probe()`` ACQUIRES and returns a ready
``xr.Dataset``: the program only runs against the probe's own baked config
(which carries the ``flux_pulse1..16`` operations), and the backend's shared
fetch path would regenerate a config without them.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np
import xarray as xr
from qm.qua import *

from qualang_tools.bakery import baking
from qualang_tools.loops import from_array

from scqo_qm.experiments._amp_limits import MAX_AMP_SCALE
from scqo_qm.experiments._lib import acquire as _acquire
from scqo_qm.experiments._flux_limits import (
    check_flux_pulse_relative,
    dac_rail_v,
    declared_idle_offset_v,
    flux_reference_amplitude,
    rail_remedy,
)


def _flux_qubit(qp, flux_role: str):
    """Return the qubit of the pair whose z line carries the flux pulse."""
    return qp.qubit_target if flux_role == "target" else qp.qubit_control


def resolve_amplitudes(qubit_pairs, amplitudes, *, amp_mode: str, flux_role: str):
    """Resolve the amplitude sweep into what the QUA program actually needs.

    Returns `(qua_amps, base_levels, denoms)`:
      - `qua_amps` is the array the single shared QUA loop iterates (the value the
        baked pulse's `amp_array` scales by);
      - `base_levels[pair]` is the level the 1..16 ns segments are baked at;
      - `denoms[pair]` is the stored amplitude of the z line's `const` operation,
        which the >16 ns `play` branch divides by.

    Both QUA branches emit `base_level * qua_amp` volts -- the baked branch scales
    the baked waveform directly, the play branch plays `const` at
    `(base_level/denom) * qua_amp`. That equality is the invariant every mode has
    to preserve, and it is what `tests/test_pair_swap_probes.py` pins.

    In "prefactor" mode `base_level` is the per-pair QUAM |11>-|02> amplitude and
    `qua_amps` is the sweep verbatim -- the same numbers this probe has always
    played, so the qualibrate node is unchanged (what is new there is that the
    degenerate cases refuse by name instead of by ZeroDivisionError). In
    "absolute" mode the sweep is volts: one
    reference `amp_ref = max|amplitudes|` becomes the baked level for every pair
    and `qua_amps = amplitudes / amp_ref`, so the emitted volts equal the swept
    value on both branches. The reference must be a SCALAR because
    `for_(*from_array(a, ...))` is one loop shared by every multiplexed pair;
    deriving it from the sweep also makes the stored waveform peak equal the
    sweep's own maximum, so the DAC-rail guard below becomes the physically
    meaningful one ("you asked to emit more than the rail can").

    Pure: no QUA, no config, no baking -- callable from a test with stub pairs.
    """
    if amp_mode not in ("absolute", "prefactor"):
        raise ValueError(f"amp_mode must be 'absolute' or 'prefactor', got {amp_mode!r}")
    if flux_role not in ("control", "target"):
        raise ValueError(f"flux_role must be 'control' or 'target', got {flux_role!r}")

    amplitudes = np.asarray(amplitudes, dtype=float)
    if amplitudes.size == 0:
        raise ValueError("amplitudes is empty; nothing to sweep.")

    # The z line carrying the pulse must exist and own the `const` operation the
    # >16 ns branch stretches. `flux_reference_amplitude` enforces the rail/2
    # convention on it (the rail being PER PORT, direct 0.5 V vs amplified 2.5 V).
    denoms = {}
    rails = {}
    for qp in qubit_pairs:
        fq = _flux_qubit(qp, flux_role)
        if fq.z is None:
            raise ValueError(
                f"{flux_role} qubit of {qp.name} ({fq.name}) has no z line; cannot play a flux pulse."
            )
        if "const" not in fq.z.operations:
            raise ValueError(
                f"z line of {fq.name} has no operation 'const'; available: {list(fq.z.operations)}"
            )
        denoms[qp.name] = flux_reference_amplitude(fq.z, name=fq.name, operation="const")
        rails[qp.name] = dac_rail_v(fq.z)

    if amp_mode == "absolute":
        amp_ref = float(np.max(np.abs(amplitudes)))
        if amp_ref == 0.0:
            raise ValueError(
                "an all-zero absolute amplitude sweep has no reference to bake against; "
                "widen the window or use amp_mode='prefactor'."
            )
        for qp in qubit_pairs:
            z = _flux_qubit(qp, flux_role).z
            check_flux_pulse_relative(
                z, name=f"{qp.name} absolute flux sweep on {_flux_qubit(qp, flux_role).name}.z",
                idle_v=declared_idle_offset_v(z), amps_v=amplitudes, operation="const")
        base_levels = {qp.name: amp_ref for qp in qubit_pairs}
        qua_amps = amplitudes / amp_ref
    else:
        # The flux-pulse base amplitude that brings |11> into resonance with |02> for each pair.
        base_levels = {}
        for qp in qubit_pairs:
            detuning = (qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency
                        - qp.qubit_target.anharmonicity)
            quad = float(qp.qubit_control.freq_vs_flux_01_quad_term or 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                level = float(np.sqrt(-detuning / quad)) if quad else float("nan")
            if not np.isfinite(level) or level <= 0.0:
                # The pre-factor sweep is defined RELATIVE to this level, so a
                # chip whose quad term is unmeasured (0/None) or whose detuning
                # has the wrong sign has nothing for it to be a factor OF. This
                # used to be a bare ZeroDivisionError (quad 0), a TypeError
                # (quad None) or a baked NaN waveform (wrong sign) -- and it is
                # the state 7 of the 9 live chipA pairs are actually in.
                raise ValueError(
                    f"{qp.name}: the |11>-|02> base amplitude is not a usable level "
                    f"({level}). It is sqrt(-detuning / freq_vs_flux_01_quad_term) with "
                    f"detuning = {detuning:.4g} Hz and "
                    f"freq_vs_flux_01_quad_term = {quad:.4g} — measure the control qubit's "
                    f"flux arch (qubit_spectroscopy_flux_pulse) to fill the quad term in, or "
                    f"sweep volts directly with amp_mode='absolute', which does not use it."
                )
            base_levels[qp.name] = level
        qua_amps = amplitudes
        for qp in qubit_pairs:
            level = base_levels[qp.name]
            if abs(level) >= rails[qp.name]:
                # WARN, not raise: this level is computed from the chip's own
                # |11>-|02> detuning, and a real chip can legitimately land above
                # the rail. Raising here would break the qualibrate node on a
                # config that has always "worked" -- silently clipped.
                z = _flux_qubit(qp, flux_role).z
                warnings.warn(
                    f"{qp.name}: the |11>-|02> base amplitude is {level} V, at or past "
                    f"the port's {rails[qp.name]} V full scale. The baked waveform peak is "
                    f"clipped on hardware and the simulator hides it -- sweep in volts "
                    f"(amp_mode='absolute') below the rail instead. "
                    + rail_remedy(z, name=f"{qp.name} base amplitude",
                                  needed_v=abs(level), rail=rails[qp.name]),
                    RuntimeWarning, stacklevel=2)

    # One baked op set per flux ELEMENT (baking files its pulses and waveforms
    # under the z element's own name -- `<element>_baked_pulse_<i>`), so two
    # pairs sharing a flux qubit at DIFFERENT base levels would silently
    # overwrite each other's waveforms in the shared config.
    per_element = {}
    for qp in qubit_pairs:
        element = _flux_qubit(qp, flux_role).z.name
        seen = per_element.setdefault(element, (qp.name, base_levels[qp.name]))
        if seen[0] != qp.name and seen[1] != base_levels[qp.name]:
            raise ValueError(
                f"pairs {seen[0]} and {qp.name} share the flux element {element!r} but need "
                f"different baked base levels ({seen[1]} V vs {base_levels[qp.name]} V); the "
                f"baked flux_pulse ops would overwrite each other. Run these pairs separately."
            )

    max_qua = float(np.max(np.abs(qua_amps)))
    if max_qua >= MAX_AMP_SCALE:
        raise ValueError(
            f"amplitude sweep exceeds QUA's amplitude_scale range: max |a| = {max_qua:.3f} "
            f">= {MAX_AMP_SCALE} (the baked pulse is scaled by this value directly)."
        )
    for qp in qubit_pairs:
        play_scale = abs(base_levels[qp.name] / denoms[qp.name]) * max_qua
        if play_scale >= MAX_AMP_SCALE:
            raise ValueError(
                f"flux sweep for {qp.name} exceeds QUA's amplitude_scale range on the >16 ns "
                f"branch: max |(base/const)*a| = {play_scale:.3f} >= {MAX_AMP_SCALE} "
                f"(base = {base_levels[qp.name]} V, const = {denoms[qp.name]} V). "
                f"Reduce the amplitude range or raise the 'const' op amplitude."
            )
    return qua_amps, base_levels, denoms


def baked_waveform(qubit, baked_config, base_level: float = 0.5, max_samples: int = 16):
    """Create truncated baked waveforms for the chevron flux pulse.

    Generates a list of baking objects, each containing an incrementally longer flux pulse
    (1..max_samples samples) at the specified base_level. Each baked pulse is registered
    as an operation named "flux_pulse{i}" on the provided qubit z line.

    Copied from `calibration_utils.chevron_cz.parameters.baked_waveform` so this probe
    stays free of qualibrate imports.

    Returns a list of baking objects; index i corresponds to a pulse of i+1 samples.
    """
    pulse_segments = []
    waveform = [base_level] * max_samples
    for i in range(1, max_samples + 1):
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments


def build_program(
    machine,
    qubit_pairs,
    *,
    amplitudes,
    times_cycles,
    num_shots: int,
    reset_type: str,
    use_state_discrimination: bool,
    amp_mode: str = "prefactor",
    flux_role: str = "control",
    drive_role: str = "control",
    simulate: bool = False,
):
    """Build the single-excitation flux-chevron QUA program.

    Returns (program, sweep_axes, baked_config). The returned `baked_config` carries
    the baked flux-pulse operations and MUST be the config used to execute (pass it to
    `acquire(..., config=baked_config)`); a freshly generated config would lack them.

    `amplitudes` is the flux-pulse amplitude sweep -- a unitless pre-factor centred on
    1.0 when `amp_mode="prefactor"` (the default), or absolute volts when
    `amp_mode="absolute"` (see the module docstring and `resolve_amplitudes`).
    `times_cycles` is the pulse-duration sweep in ns; `qubit_pairs` is a BatchableList
    of qubit pairs (`qubit_pairs.batch()` / `.get_names()`). `flux_role` selects which
    qubit of each pair carries the flux pulse and `drive_role` which receives the x180
    ("control" or "target"); the two are independent.
    """
    num_qubit_pairs = len(qubit_pairs)

    qua_amps, base_levels, denoms = resolve_amplitudes(
        qubit_pairs, amplitudes, amp_mode=amp_mode, flux_role=flux_role
    )

    sweep_axes = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        # the axis carries what the CALLER swept, not the QUA scale factor
        "amplitude": xr.DataArray(
            np.asarray(amplitudes, dtype=float),
            attrs={"long_name": "amplitudes of the flux pulse",
                   "units": "V" if amp_mode == "absolute" else "prefactor"},
        ),
        "time": xr.DataArray(times_cycles, attrs={"long_name": "pulse duration", "units": "ns"}),
    }

    baked_config = machine.generate_config()

    # Pre-compute the baked short segments (1..16 samples) for each flux qubit in the pairs.
    baked_signals = {
        _flux_qubit(qp, flux_role).name: baked_waveform(
            _flux_qubit(qp, flux_role), baked_config,
            base_level=base_levels[qp.name], max_samples=16
        )
        for qp in qubit_pairs
    }

    with program() as prog:
        t = declare(int)  # QUA variable for the flux pulse segment index
        a = declare(fixed)
        t_left_ns = declare(int)
        t_cycles = declare(int)
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = machine.declare_qua_variables()
        if use_state_discrimination:
            # Per-shot single-qubit outcomes (both qubits read out 2-level -> {0, 1}).
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            # Joint two-qubit indicators (first digit = control, second = target);
            # averaged over shots they give the P00/P01/P10/P11 populations.
            ind_gg = declare(int)  # 00
            ind_ge = declare(int)  # 01
            ind_eg = declare(int)  # 10
            ind_ee = declare(int)  # 11
            state_gg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ge_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_eg_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_ee_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                machine.initialize_qpu(target=qp.qubit_control)
                machine.initialize_qpu(target=qp.qubit_target)
            align()
            # Averaging loop
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                # Pulse amplitude loop (the QUA scale factor; see resolve_amplitudes)
                with for_(*from_array(a, qua_amps)):
                    ################################################################################################
                    # The duration argument in the play command can only produce pulses with duration multiple of  #
                    # 4ns. To overcome this limitation we use the baking tool from the qualang-tools package to    #
                    # generate pulses with 1ns granularity. To avoid creating custom waveforms for each iteration  #
                    # we combine baked pulses with dynamically stretched (multiple of 4ns) pulses.                 #
                    ################################################################################################
                    with for_(*from_array(t, times_cycles)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            fq = _flux_qubit(qp, flux_role)
                            # Qubit initialization
                            qp.qubit_control.reset(reset_type, simulate)
                            qp.qubit_target.reset(reset_type, simulate)
                            align()
                            # Excite only one qubit of the pair (single-excitation chevron).
                            if drive_role == "target":
                                qp.qubit_target.xy.play("x180")
                            else:
                                qp.qubit_control.xy.play("x180")

                            align()

                            # For the first 16ns we play baked pulses exclusively. Loop the time index until 16.
                            with if_(t <= 16):
                                with switch_(t):
                                    # Switch case to select the baked pulse with duration t ns
                                    for j in range(1, 17):
                                        with case_(j):
                                            baked_signals[fq.name][j - 1].run(
                                                amp_array=[(fq.z.name, a)]
                                            )

                            # For pulse durations above 16ns we combine baking with regular play statements.
                            with else_():
                                # We calculate the closest lower multiple of 4 of the time index
                                assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                                # Calculate the duration to add to pulse multiple of 4.
                                assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 to multiply by 4
                                # Switch case with the 4 possible sequences:
                                with switch_(t_left_ns):
                                    # Play only the pulse multiple of 4
                                    with case_(0):
                                        align()
                                        p = base_levels[qp.name]
                                        denom = denoms[qp.name]
                                        scale = (p / denom) * a
                                        fq.z.play(
                                            "const",
                                            duration=t_cycles,
                                            amplitude_scale=scale,
                                        )
                                    # Play the pulse multiple of 4 followed by the baked pulse of the missing duration
                                    for j in range(1, 4):
                                        with case_(j):
                                            align()
                                            p = base_levels[qp.name]
                                            denom = denoms[qp.name]
                                            scale = (p / denom) * a
                                            with strict_timing_():
                                                fq.z.play(
                                                    "const",
                                                    duration=t_cycles,
                                                    amplitude_scale=scale,
                                                )
                                                baked_signals[fq.name][j - 1].run(
                                                    amp_array=[(fq.z.name, a)]
                                                )
                            align()

                            if use_state_discrimination:
                                qp.qubit_control.readout_state(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                # Joint-state indicators from the two binary outcomes:
                                #   ee(11)=c*t, eg(10)=c-ee, ge(01)=t-ee, gg(00)=1-c-t+ee
                                assign(ind_ee, state_c[ii] * state_t[ii])
                                assign(ind_eg, state_c[ii] - ind_ee)
                                assign(ind_ge, state_t[ii] - ind_ee)
                                assign(ind_gg, 1 - state_c[ii] - state_t[ii] + ind_ee)
                                save(ind_gg, state_gg_st[ii])
                                save(ind_ge, state_ge_st[ii])
                                save(ind_eg, state_eg_st[ii])
                                save(ind_ee, state_ee_st[ii])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if use_state_discrimination:
                    # Averaging the 0/1 indicators over shots yields the joint populations.
                    state_gg_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_gg{i}")
                    state_ge_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_ge{i}")
                    state_eg_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_eg{i}")
                    state_ee_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_ee{i}")
                else:
                    I_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_target{i}")

    return prog, sweep_axes, baked_config


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

    Pass the baked config returned by `build_program` as `config`; the shared
    execute-and-fetch helper would otherwise regenerate a config without the baked ops.
    """
    return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)


from typing import Any

import numpy as np
import xarray as xr
from scqo import register
from scqo.experiments import PairSwapChevron

from ._pair_roles import JointPopulationMixin


@register
class QMPairSwapChevron(JointPopulationMixin, PairSwapChevron):
    """Build, run and fetch the multiplexed swap chevron on the QM OPX."""

    # preview opt-out (backend.SELF_ACQUIRING_ATTR): truthy reason = refuse
    probe_self_acquires = ("it bakes a per-call config for the sub-17 ns "
                           "branch and fetches against it inside probe()")

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from ._vendor import role_side, vendor_pair_name
        from scqo_qm.experiments._lib import select_qubit_pairs

        machine = self.backend.machine  # type: ignore[attr-defined]
        # targets are ROSTER composite names; the probe helper selects by QUAM
        # pair key (QM names its pairs after the coupler), so translate first —
        # order preserved, which is what the axis relabel below relies on.
        vendor_names = [vendor_pair_name(self, p) for p in self.params.targets]
        pairs = select_qubit_pairs(machine, vendor_names, multiplexed=True)

        # Resolved BEFORE any QUA is built, so a roster/params mismatch refuses
        # without costing instrument time. `high` fixes the reduce_raw
        # orientation; the two selectors are independent of each other.
        self._high_side = role_side(self, "high", field="targets")
        drive_role = role_side(self, self.params.drive_side, field="drive_side")
        flux_role = role_side(self, self.params.flux_side, field="flux_side",
                              needs_flux=True)

        # The probe's `times_cycles` argument is ns despite its name (it plays
        # 1..16 ns baked segments below the 4 ns clock), and it must be an
        # integer array — from_array loops it into an int QUA variable.
        times_ns = np.round(self.sweep_axes["swap_time_ns"]).astype(int)
        prog, axes, baked_config = build_program(
            machine,
            pairs,
            amplitudes=self.sweep_axes["flux_amp_v"],
            times_cycles=times_ns,
            num_shots=self.params.num_averages,
            reset_type=check_reset_method(self),
            use_state_discrimination=True,
            amp_mode="absolute",
            flux_role=flux_role,
            drive_role=drive_role,
        )
        # The canonical time axis is the probe's REAL grid: re-declare it so
        # sizes and values match the raw data exactly.
        self.sweep_axes["swap_time_ns"] = axes["time"].values.astype(float)
        sweep_axes = {
            # The probe labels its target axis with VENDOR pair keys; scqo's
            # dataset (and estimate()) key on the ROSTER composite names the
            # operator asked for. Same order by construction (vendor_names was
            # built from targets), so relabel rather than rename downstream.
            "qubit_pair": xr.DataArray(list(self.params.targets)),
            "flux_amp_v": axes["amplitude"],
            "swap_time_ns": axes["time"],
        }
        # Acquire here: the baked config is per-call and cannot be reached
        # through the backend's (program, axes, module) shape.
        shots = getattr(self.params, "num_averages", None) or getattr(
            self.params, "num_shots", 1)
        return acquire(
            machine, prog, sweep_axes,
            num_shots=int(shots), timeout=self.backend._timeout,
            config=baked_config,
        )
