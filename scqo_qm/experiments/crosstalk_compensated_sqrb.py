"""Crosstalk Compensated Simultaneous SQRB acquisition probe (QM/QUAM).

Supplies QUA program generation and data acquisition for CrosstalkCompensatedSQRB.
Compares isolated SQRB, simultaneous SQRB, and active crosstalk-compensated simultaneous SQRB.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.loops import from_array

from scqo import register
from scqo.experiments.crosstalk_compensated_sqrb import (
    CrosstalkCompensatedSQRB,
    compute_theoretical_phase_rate,
)
from scqo_qm.experiments._lib import acquire as _acquire

# Recovery gates lookup: inv_gates[i] is the Clifford gate index that resets state i to 0
# Recovery gates lookup: inv_gates[i] is the Clifford gate index that resets state i to 0
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

# 3-subslot decomposition for all 24 Clifford gates: (slot0, slot1, slot2)
# Each entry is None (idle wait) or (op_name, pulse_kind, axis_phase_turns)
CLIFFORD_3SUBSLOTS = [
    # Gate 0: I
    (None, None, None),
    # Gate 1: x180
    (("x180", "x180", 0.0), None, None),
    # Gate 2: y180
    (("y180", "x180", 0.25), None, None),
    # Gate 3: y180, x180
    (("y180", "x180", 0.25), ("x180", "x180", 0.0), None),
    # Gate 4: x90, y90
    (("x90", "x90", 0.0), ("y90", "x90", 0.25), None),
    # Gate 5: x90, -y90
    (("x90", "x90", 0.0), ("-y90", "x90", 0.75), None),
    # Gate 6: -x90, y90
    (("-x90", "x90", 0.5), ("y90", "x90", 0.25), None),
    # Gate 7: -x90, -y90
    (("-x90", "x90", 0.5), ("-y90", "x90", 0.75), None),
    # Gate 8: y90, x90
    (("y90", "x90", 0.25), ("x90", "x90", 0.0), None),
    # Gate 9: y90, -x90
    (("y90", "x90", 0.25), ("-x90", "x90", 0.5), None),
    # Gate 10: -y90, x90
    (("-y90", "x90", 0.75), ("x90", "x90", 0.0), None),
    # Gate 11: -y90, -x90
    (("-y90", "x90", 0.75), ("-x90", "x90", 0.5), None),
    # Gate 12: x90
    (("x90", "x90", 0.0), None, None),
    # Gate 13: -x90
    (("-x90", "x90", 0.5), None, None),
    # Gate 14: y90
    (("y90", "x90", 0.25), None, None),
    # Gate 15: -y90
    (("-y90", "x90", 0.75), None, None),
    # Gate 16: -x90, y90, x90
    (("-x90", "x90", 0.5), ("y90", "x90", 0.25), ("x90", "x90", 0.0)),
    # Gate 17: -x90, -y90, x90
    (("-x90", "x90", 0.5), ("-y90", "x90", 0.75), ("x90", "x90", 0.0)),
    # Gate 18: x180, y90
    (("x180", "x180", 0.0), ("y90", "x90", 0.25), None),
    # Gate 19: x180, -y90
    (("x180", "x180", 0.0), ("-y90", "x90", 0.75), None),
    # Gate 20: y180, x90
    (("y180", "x180", 0.25), ("x90", "x90", 0.0), None),
    # Gate 21: y180, -x90
    (("y180", "x180", 0.25), ("-x90", "x90", 0.5), None),
    # Gate 22: x90, y90, x90
    (("x90", "x90", 0.0), ("y90", "x90", 0.25), ("x90", "x90", 0.0)),
    # Gate 23: -x90, y90, -x90
    (("-x90", "x90", 0.5), ("y90", "x90", 0.25), ("-x90", "x90", 0.5)),
]


# Every op a compensation track can play in benchmark mode (mirrored from the aggressor)
CANCEL_MIRROR_OPS: tuple[str, ...] = tuple(
    sorted({slot[0] for subslots in CLIFFORD_3SUBSLOTS for slot in subslots if slot is not None})
)


def play_fixed_clifford(gate_idx, qubit, pulse_cycles: int):
    """Play single Clifford gate decomposed into 3 fixed-duration subslots."""
    with switch_(gate_idx, unsafe=True):
        for idx, subslots in enumerate(CLIFFORD_3SUBSLOTS):
            with case_(idx):
                for slot in subslots:
                    if slot is None:
                        qubit.xy.wait(pulse_cycles)
                    else:
                        qubit.xy.play(slot[0])


def play_fixed_compensation_clifford(
    drive_gate_idx,
    cancel_elem_name: str,
    amp_val,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
):
    """Play compensation pulses for the 3 subslots mirroring the drive qubit gate."""
    theta = -2.0 * np.pi * float(sub_phase_rate_turns)
    a = float(amp_val)
    v00 = float(a * np.cos(theta))
    v01 = float(-a * np.sin(theta))
    v10 = float(a * np.sin(theta))
    v11 = float(a * np.cos(theta))

    with switch_(drive_gate_idx, unsafe=True):
        for idx, subslots in enumerate(CLIFFORD_3SUBSLOTS):
            with case_(idx):
                slots = [s for s in subslots if s is not None]
                n_pulses = len(slots)
                if n_pulses == 3:
                    # 3-pulse gates: advance frame by 2*sub_p after pulse 1 and use
                    # matrix amp rotation for pulse 2 to keep subslot 2 free of frame rotations.
                    play(slots[0][0] * amp(a), cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                    play(slots[1][0] * amp(a), cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        frame_rotation_2pi(2.0 * sub_phase_rate_turns, cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        play(slots[2][0] * amp(v00, v01, v10, v11), cancel_elem_name)
                    else:
                        play(slots[2][0] * amp(a), cancel_elem_name)
                else:
                    # 0, 1, or 2 pulse gates: idle subslots rotate frame before wait
                    for slot in subslots:
                        if slot is None:
                            if sub_phase_rate_turns != 0.0:
                                frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                            wait(pulse_cycles, cancel_elem_name)
                        else:
                            play(slot[0] * amp(a), cancel_elem_name)
                            if sub_phase_rate_turns != 0.0:
                                frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)


def _play_cancel_subslot(
    slot_idx: int,
    subslots_src,
    cancel_elem_name: str,
    amp_val: float,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
):
    """Play one subslot of a compensation track mirroring the aggressor's gate.

    Idle subslots advance the frame before the wait, pulsed subslots after the play. For
    3-pulse gates the frame advances by 2*sub_p after pulse 2 and pulse 3 is rotated back by
    sub_p through the amp matrix, keeping subslot 3 free of frame rotations.
    """
    slots_src = [s for s in subslots_src if s is not None]
    a = float(amp_val)
    if len(slots_src) == 3 and slot_idx == 1:
        play(slots_src[1][0] * amp(a), cancel_elem_name)
        if sub_phase_rate_turns != 0.0:
            frame_rotation_2pi(2.0 * sub_phase_rate_turns, cancel_elem_name)
        return
    if len(slots_src) == 3 and slot_idx == 2:
        if sub_phase_rate_turns != 0.0:
            theta = -2.0 * np.pi * float(sub_phase_rate_turns)
            v00 = float(a * np.cos(theta))
            v01 = float(-a * np.sin(theta))
            v10 = float(a * np.sin(theta))
            v11 = float(a * np.cos(theta))
            play(slots_src[2][0] * amp(v00, v01, v10, v11), cancel_elem_name)
        else:
            play(slots_src[2][0] * amp(a), cancel_elem_name)
        return

    slot = subslots_src[slot_idx]
    if slot is None:
        if sub_phase_rate_turns != 0.0:
            frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
        wait(pulse_cycles, cancel_elem_name)
    else:
        play(slot[0] * amp(a), cancel_elem_name)
        if sub_phase_rate_turns != 0.0:
            frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)


def _stark_scale(slot) -> float:
    """AC-Stark-like Z per pulse relative to a 180-type pulse (quadratic in amplitude)."""
    return 1.0 if slot[1] == "x180" else 0.25


def play_simultaneous_clifford_subslots(
    gate_p: int,
    gate_d: int,
    *,
    cond: str,
    probe_qubit: Any,
    drive_qubit: Any,
    cancel_elem_name: str,
    amp_val: float,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
    reverse_cancel_elem_name: str | None = None,
    reverse_amp_val: float = 0.0,
    reverse_sub_phase_rate_turns: float = 0.0,
    stark_turns_probe: float = 0.0,
    stark_turns_drive: float = 0.0,
):
    """Play single Clifford gate decomposed into 3 fixed-duration subslots synchronously across channels.

    Subslots are played simultaneously for probe_qubit, drive_qubit and the cancel element(s)
    to ensure perfect pulse alignment and zero inter-gate timing gaps under strict_timing_().

    cond: 'isolated' (probe only), 'isolated_drive' (drive only, bidirectional isolated pass),
    'simultaneous' (both) or 'compensated' (both + cancellation). Under 'compensated',
    cancel_elem_name mirrors the drive gate (drive -> probe crosstalk) and, when given,
    reverse_cancel_elem_name mirrors the probe gate (probe -> drive crosstalk).

    Stark correction (compensated only): after every drive pulse the probe's frame turns by
    stark_turns_probe (x 1/4 for a 90-type pulse: the shift goes as amplitude^2), and after
    every probe pulse the drive's frame turns by stark_turns_drive. Turning a qubit's frame
    also turns the phase of the crosstalk ITS pulses cause, so the cancel element mirroring
    that qubit turns with it.
    """
    subslots_p = CLIFFORD_3SUBSLOTS[gate_p]
    subslots_d = CLIFFORD_3SUBSLOTS[gate_d]
    play_probe = cond != "isolated_drive"
    play_drive = cond in ("simultaneous", "compensated", "isolated_drive")
    compensate = cond == "compensated"

    for slot_idx in range(3):
        if play_probe:
            if subslots_p[slot_idx] is None:
                probe_qubit.xy.wait(pulse_cycles)
            else:
                probe_qubit.xy.play(subslots_p[slot_idx][0])

        if play_drive:
            if subslots_d[slot_idx] is None:
                drive_qubit.xy.wait(pulse_cycles)
            else:
                drive_qubit.xy.play(subslots_d[slot_idx][0])

        if compensate:
            _play_cancel_subslot(
                slot_idx, subslots_d, cancel_elem_name, amp_val, sub_phase_rate_turns, pulse_cycles
            )
            if reverse_cancel_elem_name is not None:
                _play_cancel_subslot(
                    slot_idx,
                    subslots_p,
                    reverse_cancel_elem_name,
                    reverse_amp_val,
                    reverse_sub_phase_rate_turns,
                    pulse_cycles,
                )
            if stark_turns_probe and subslots_d[slot_idx] is not None:
                t = float(stark_turns_probe) * _stark_scale(subslots_d[slot_idx])
                frame_rotation_2pi(t, probe_qubit.xy.name)
                if reverse_cancel_elem_name is not None:
                    frame_rotation_2pi(t, reverse_cancel_elem_name)
            if stark_turns_drive and subslots_p[slot_idx] is not None:
                t = float(stark_turns_drive) * _stark_scale(subslots_p[slot_idx])
                frame_rotation_2pi(t, drive_qubit.xy.name)
                frame_rotation_2pi(t, cancel_elem_name)


CLIFFORD_PULSES: list[list[str]] = [
    [],                                      # 0: I
    ["x180"],                                # 1: X
    ["y180"],                                # 2: Y
    ["y180", "x180"],                        # 3: Y, X
    ["x90", "y90"],                          # 4: X/2, Y/2
    ["x90", "-y90"],                         # 5: X/2, -Y/2
    ["-x90", "y90"],                         # 6: -X/2, Y/2
    ["-x90", "-y90"],                        # 7: -X/2, -Y/2
    ["y90", "x90"],                          # 8: Y/2, X/2
    ["y90", "-x90"],                         # 9: Y/2, -X/2
    ["-y90", "x90"],                         # 10: -Y/2, X/2
    ["-y90", "-x90"],                        # 11: -Y/2, -X/2
    ["x90"],                                 # 12: X/2
    ["-x90"],                                # 13: -X/2
    ["y90"],                                 # 14: Y/2
    ["-y90"],                                # 15: -Y/2
    ["-x90", "y90", "x90"],                  # 16: -X/2, Y/2, X/2
    ["-x90", "-y90", "x90"],                 # 17: -X/2, -Y/2, X/2
    ["x180", "y90"],                         # 18: X, Y/2
    ["x180", "-y90"],                        # 19: X, -Y/2
    ["y180", "x90"],                         # 20: Y, X/2
    ["y180", "-x90"],                        # 21: Y, -X/2
    ["x90", "y90", "x90"],                   # 22: X/2, Y/2, X/2
    ["-x90", "y90", "-x90"],                 # 23: -X/2, Y/2, -X/2
]


def generate_clifford_sequences(
    depth: int,
    num_sequences: int,
    seed: int | None = None,
) -> list[list[int]]:
    """Precompute random Clifford sequences and their recovery gate at compile time."""
    rng = np.random.default_rng(seed)
    sequences = []
    for _ in range(num_sequences):
        gates = rng.integers(0, 24, size=depth).tolist()
        curr = 0
        for g in gates:
            curr = int(c1_table[curr, g])
        inv_gate = int(inv_gates[curr])
        sequences.append(gates + [inv_gate])
    return sequences


def play_compact_clifford(gate_idx, qubit, pulse_cycles: int):
    """Play single Clifford gate compactly matching qubit_sqrb without subslot padding."""
    with switch_(gate_idx, unsafe=True):
        for idx, ops in enumerate(CLIFFORD_PULSES):
            with case_(idx):
                if len(ops) == 0:
                    qubit.xy.wait(pulse_cycles)
                else:
                    for op in ops:
                        qubit.xy.play(op)


def play_compact_compensation_clifford(
    drive_gate_idx,
    cancel_elem_name: str,
    amp_val,
    sub_phase_rate_turns: float,
    pulse_cycles: int,
):
    """Play compensation pulses compactly mirroring the drive qubit gate without subslot padding."""
    with switch_(drive_gate_idx, unsafe=True):
        for idx, ops in enumerate(CLIFFORD_PULSES):
            with case_(idx):
                if len(ops) == 0:
                    wait(pulse_cycles, cancel_elem_name)
                    if sub_phase_rate_turns != 0.0:
                        frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)
                else:
                    for op in ops:
                        play(op * amp(amp_val), cancel_elem_name)
                        if sub_phase_rate_turns != 0.0:
                            frame_rotation_2pi(sub_phase_rate_turns, cancel_elem_name)


def fpga_clifford_sequence(seed: int, seq_index: int, depth: int) -> list[int]:
    """sequence_mode='fpga' sequence: `depth` random Cliffords from its own stream (seed, seq_index).

    Every shorter depth d plays the first d gates (recovery from fpga_recovery_gates), so one
    array serves all depths.
    """
    gates = np.random.default_rng([int(seed), int(seq_index)]).integers(0, 24, size=int(depth)).tolist()
    return [int(g) for g in gates]


def fpga_recovery_gates(gates: list[int], depths) -> list[int]:
    """Recovery Clifford after the first d gates, for each d in depths."""
    wanted = {int(d) for d in depths}
    out: dict[int, int] = {}
    curr = 0
    if 0 in wanted:
        out[0] = int(inv_gates[0])
    for i, g in enumerate(gates, 1):
        curr = int(c1_table[curr, g])
        if i in wanted:
            out[i] = int(inv_gates[curr])
    return [out[int(d)] for d in depths]


N_CLIFFORDS = len(CLIFFORD_3SUBSLOTS)


def _declare_fpga_sequences(
    depth_arr,
    seq_arr,
    *,
    num_sequences: int,
    seed: int | None,
    unroll: int,
    arrays: tuple[str, ...] = ("p", "d"),
) -> dict:
    """QUA arrays for sequence_mode='fpga' (call inside the program).

    Sequence s draws d_max gates from its own stream (seed, s) (probe) and (seed + 1000, s)
    (drive); depth d plays the first d and the recovery gate of that prefix. These are not the
    sequences the unrolled mode draws. Flat layout: gates[i * d_max + k], inv[i * n_d + j] for
    the i-th sequence of this program. `arrays` picks which to declare: 'p' (probe), 'd'
    (drive), 'c' (merged index gate_p * 24 + gate_d for a single switch_ over both qubits).
    """
    d_max = int(np.max(depth_arr))
    seed_p = seed if seed is not None else 42
    seed_d = (seed + 1000) if seed is not None else 1042
    out: dict[str, Any] = {"d_max": d_max, "n_d": len(depth_arr), "n_s": len(seq_arr), "unroll": max(1, int(unroll))}
    lists: dict[str, tuple[list[int], list[int]]] = {}
    for key, sd in (("p", seed_p), ("d", seed_d)):
        gates, inv = [], []
        for s_idx in seq_arr:
            seq = fpga_clifford_sequence(sd, int(s_idx), d_max)
            gates.extend(seq)
            inv.extend(fpga_recovery_gates(seq, depth_arr))
        lists[key] = (gates, inv)
    lists["c"] = tuple(
        [a * N_CLIFFORDS + b for a, b in zip(lists["p"][i], lists["d"][i])] for i in range(2)
    )
    for key in arrays:
        out[f"gates_{key}"] = declare(int, value=lists[key][0])
        out[f"inv_{key}"] = declare(int, value=lists[key][1])
        out[f"g_inv_{key}"] = declare(int)
    out["depths"] = declare(int, value=[int(d) for d in depth_arr])
    out["seq_ids"] = declare(int, value=[int(s) for s in seq_arr])
    for name in ("s", "j", "k", "d", "base", "k_end"):
        out[name] = declare(int)
    # one loop variable per element for dispatch='parallel' (each element loops on its own)
    out["k_el"] = [declare(int) for _ in range(4)]
    return out


def _declare_fpga_rng(depth_arr, seq_arr, *, seed: int | None, unroll: int, arrays: tuple[str, ...]) -> dict:
    """QUA variables for sequence_mode='fpga' with sequences drawn ON the FPGA (call inside the program).

    As qubit_sqrb: a QUA Random(seed) draws d_max gates per qubit for each sequence, the Cayley
    table tracks the state, and after[i] holds the recovery gate after the first i+1 gates, so
    depth d plays gates[:d] then after[d-1]. Only one sequence lives in memory at a time, so the
    number of sequences per program is unlimited. sequence_idx s of the program is the s-th draw.
    """
    depths = [int(d) for d in depth_arr]
    if min(depths) < 1:
        raise ValueError("fpga_sequence_source='fpga_rng' needs depths >= 1")
    seq_ids = [int(x) for x in seq_arr]
    if seq_ids != list(range(seq_ids[0], seq_ids[0] + len(seq_ids))):
        raise ValueError("fpga_sequence_source='fpga_rng' plays consecutive sequence indices")
    d_max = max(depths)
    out: dict[str, Any] = {
        "rng": True, "d_max": d_max, "n_d": len(depths), "n_s": len(seq_ids), "first_seq": seq_ids[0],
        "unroll": max(1, int(unroll)),
    }
    out["cayley"] = declare(int, value=c1_table.flatten().tolist())
    out["inv_list"] = declare(int, value=inv_gates)
    for key in ("p", "d"):
        out[f"gates_{key}"] = declare(int, size=d_max)
        out[f"after_{key}"] = declare(int, size=d_max)
        out[f"cur_{key}"] = declare(int)
        out[f"g_inv_{key}"] = declare(int)
    if "c" in arrays:
        out["gates_c"] = declare(int, size=d_max)
        out["g_inv_c"] = declare(int)
    out["step"] = declare(int)
    out["i_gen"] = declare(int)
    out["rand"] = Random(seed=seed if seed is not None else 42)
    out["depths"] = declare(int, value=depths)
    for name in ("s", "j", "k", "d", "base", "k_end"):
        out[name] = declare(int)
    out["k_el"] = [declare(int) for _ in range(4)]
    return out


def _generate_fpga_sequence(fpga: dict):
    """Draw the next probe and drive sequences (d_max gates each) and their recovery gates."""
    i_gen, step, cayley, inv_list = fpga["i_gen"], fpga["step"], fpga["cayley"], fpga["inv_list"]
    for key in ("p", "d"):
        assign(fpga[f"cur_{key}"], 0)
    with for_(i_gen, 0, i_gen < fpga["d_max"], i_gen + 1):
        for key in ("p", "d"):
            cur = fpga[f"cur_{key}"]
            assign(step, fpga["rand"].rand_int(N_CLIFFORDS))
            assign(fpga[f"gates_{key}"][i_gen], step)
            assign(cur, cayley[cur * N_CLIFFORDS + step])
            assign(fpga[f"after_{key}"][i_gen], inv_list[cur])
        if "gates_c" in fpga:
            assign(fpga["gates_c"][i_gen], fpga["gates_p"][i_gen] * N_CLIFFORDS + fpga["gates_d"][i_gen])


def _play_composite(gate, element: str, scale: float | None = None):
    """Play composite Clifford `gate` (QUA int expression or Python int) on element."""
    if isinstance(gate, (int, np.integer)):
        op = f"{COMPOSITE_PREFIX}{int(gate)}"
        play(op if scale is None else op * amp(scale), element)
        return
    with switch_(gate, unsafe=True):
        for g in range(len(CLIFFORD_3SUBSLOTS)):
            with case_(g):
                op = f"{COMPOSITE_PREFIX}{g}"
                play(op if scale is None else op * amp(scale), element)


def _fpga_elements(cond, *, probe_qubit, drive_qubit, cancel_elem, reverse_cancel_elem, cancel_amp, drive_cancel_amp):
    """(element, source qubit 'p'/'d', amp scale or None) for every element a condition plays."""
    out = []
    if cond != "isolated_drive":
        out.append((probe_qubit.xy.name, "p", None))
    if cond in ("simultaneous", "compensated", "isolated_drive"):
        out.append((drive_qubit.xy.name, "d", None))
    if cond == "compensated":
        out.append((cancel_elem, "d", cancel_amp))
        if reverse_cancel_elem is not None:
            out.append((reverse_cancel_elem, "p", drive_cancel_amp))
    return out


def _fpga_uses_merged(cond: str, dispatch: str) -> bool:
    """Merged dispatch only pays off when both qubits play (one switch_ instead of 2-4)."""
    return dispatch == "merged" and cond in ("simultaneous", "compensated")


def _play_fpga_merged(gc, elements):
    """One switch_ over gate_p * 24 + gate_d; every case plays all elements of the condition.

    Experimental, not the default: written on the guess that the ~52 ns per-Clifford gap came
    from one switch_ per element, but the QM simulator shows the same gap with a single
    switch_ (isolated pass, 2026-10-01), so this is not its cause. Each switch is 576 cases x
    up to 4 plays - heavy to compile on the shared gateway; never measured.
    """
    with switch_(gc, unsafe=True):
        for a in range(N_CLIFFORDS):
            for b in range(N_CLIFFORDS):
                with case_(a * N_CLIFFORDS + b):
                    for element, src, scale in elements:
                        op = f"{COMPOSITE_PREFIX}{a if src == 'p' else b}"
                        play(op if scale is None else op * amp(scale), element)


def _play_fpga_clifford(gp, gd, *, cond, probe_qubit, drive_qubit, cancel_elem, reverse_cancel_elem, cancel_amp, drive_cancel_amp):
    """One Clifford on every element the condition uses (gp/gd: probe/drive gate)."""
    if cond != "isolated_drive":
        _play_composite(gp, probe_qubit.xy.name)
    if cond in ("simultaneous", "compensated", "isolated_drive"):
        _play_composite(gd, drive_qubit.xy.name)
    if cond == "compensated":
        _play_composite(gd, cancel_elem, cancel_amp)
        if reverse_cancel_elem is not None:
            _play_composite(gp, reverse_cancel_elem, drive_cancel_amp)


def _play_benchmark_fpga(
    fpga: dict,
    passes,
    *,
    probe_qubit,
    drive_qubit,
    read_qubits,
    cancel_elem,
    reverse_cancel_elem,
    xy_elems,
    cancel_amp,
    drive_cancel_amp,
    init_phase,
    drive_init_phase,
    phase_reset,
    readout_scheme,
    num_shots,
    reset_type,
    simulate,
    use_state_discrimination,
    I, Q, I_st, Q_st, n, m, m_st, state=None, state_st=None,
    dispatch: str = "parallel",
):
    """sequence_mode='fpga' body of one condition: QUA loops over sequence, depth and shot.

    Stream order matches the unrolled mode (shots innermost, then depth, then sequence).
    """
    s, j, k, d = fpga["s"], fpga["j"], fpga["k"], fpga["d"]
    base, k_end = fpga["base"], fpga["k_end"]
    d_max, n_d, unroll = fpga["d_max"], fpga["n_d"], fpga["unroll"]
    keys = [key for key in ("p", "d", "c") if f"gates_{key}" in fpga]
    kw = dict(
        probe_qubit=probe_qubit,
        drive_qubit=drive_qubit,
        cancel_elem=cancel_elem,
        reverse_cancel_elem=reverse_cancel_elem,
        cancel_amp=cancel_amp,
        drive_cancel_amp=drive_cancel_amp,
    )
    rng = bool(fpga.get("rng"))
    with for_(s, 0, s < fpga["n_s"], s + 1):
        if rng:
            assign(m, s + fpga["first_seq"])
            save(m, m_st)
            _generate_fpga_sequence(fpga)
            assign(base, 0)
        else:
            assign(m, fpga["seq_ids"][s])
            save(m, m_st)
            # first gate of this sequence in the flat arrays; set here, outside the timed block
            assign(base, s * d_max)
        with for_(j, 0, j < n_d, j + 1):
            assign(d, fpga["depths"][j])
            for key in keys:
                if not rng:
                    assign(fpga[f"g_inv_{key}"], fpga[f"inv_{key}"][s * n_d + j])
                elif key == "c":
                    assign(fpga["g_inv_c"], fpga["after_p"][d - 1] * N_CLIFFORDS + fpga["after_d"][d - 1])
                else:
                    assign(fpga[f"g_inv_{key}"], fpga[f"after_{key}"][d - 1])
            with for_(n, 0, n < num_shots, n + 1):
                for play_cond, read_idx in passes:
                    probe_qubit.reset(reset_type, simulate)
                    drive_qubit.reset(reset_type, simulate)
                    align(*xy_elems)
                    for el in xy_elems:
                        reset_frame(el)
                    for el in xy_elems:
                        reset_if_phase(el)
                    if phase_reset == "global":
                        reset_global_phase()
                    frame_rotation_2pi(init_phase / (2 * np.pi), cancel_elem)
                    if reverse_cancel_elem is not None:
                        frame_rotation_2pi(drive_init_phase / (2 * np.pi), reverse_cancel_elem)
                    align(*xy_elems)

                    if dispatch == "parallel":
                        # One FPGA loop per element, each indexing the gate array with its own
                        # loop variable: every loop is a single-element thread, so the loops run
                        # side by side from the align above and Clifford i starts at the same
                        # time on every element. On chipQM hardware (timestamp_stream,
                        # 2026-10-01) this plays 48 ns Cliffords back to back on all 4
                        # elements; the same 4 switches in ONE shared loop took 160 ns each.
                        for idx, (element, src, scale) in enumerate(_fpga_elements(play_cond, **kw)):
                            kk = fpga["k_el"][idx]
                            gates = fpga[f"gates_{src}"]
                            with for_(kk, base, kk < base + d, kk + 1):
                                _play_composite(gates[kk], element, scale)
                            _play_composite(fpga[f"g_inv_{src}"], element, scale)
                        clifford = recovery = None
                    elif _fpga_uses_merged(play_cond, dispatch):
                        elements = _fpga_elements(play_cond, **kw)
                        gates_c = fpga["gates_c"]
                        clifford = lambda i: _play_fpga_merged(gates_c[i], elements)  # noqa: E731
                        recovery = lambda: _play_fpga_merged(fpga["g_inv_c"], elements)  # noqa: E731
                    else:
                        gates_p, gates_d = fpga.get("gates_p"), fpga.get("gates_d")
                        clifford = lambda i: _play_fpga_clifford(  # noqa: E731
                            gates_p[i] if gates_p is not None else None,
                            gates_d[i] if gates_d is not None else None,
                            cond=play_cond, **kw,
                        )
                        recovery = lambda: _play_fpga_clifford(  # noqa: E731
                            fpga.get("g_inv_p"), fpga.get("g_inv_d"), cond=play_cond, **kw
                        )

                    if clifford is None:
                        pass  # dispatch='parallel' played above
                    elif unroll == 1:
                        # Index the arrays with the loop variable itself: on chipQM hardware
                        # (timestamp_stream, 2026-10-01) this plays 48 ns Cliffords back to back,
                        # while a computed index (gates[k + u]) costs ~50-70 ns per Clifford.
                        with for_(k, base, k < base + d, k + 1):
                            clifford(k)
                    else:
                        # unrolled-by-`unroll` loop over the sequence, then the remainder
                        assign(k_end, base + d - (unroll - 1))
                        with for_(k, base, k < k_end, k + unroll):
                            for u in range(unroll):
                                clifford(k if u == 0 else k + u)
                        with for_(k, k, k < base + d, k + 1):
                            clifford(k)
                    if recovery is not None:
                        recovery()

                    align()
                    for jj, i in enumerate(read_idx):
                        if jj > 0 and readout_scheme == "sequential":
                            align()
                        if use_state_discrimination:
                            read_qubits[i].readout_state(state[i])
                            save(state[i], state_st[i])
                        else:
                            read_qubits[i].resonator.measure("readout", qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
                    align()


def build_program(
    machine,
    probe_qubit,
    drive_qubit,
    *,
    mode: str,
    target_gate: str = "x180",
    depths: list[int] | None = None,
    num_sequences: int = 30,
    sequence_indices: list[int] | np.ndarray | None = None,
    num_shots: int = 1000,
    cancel_amp: float = 0.0,
    init_phase: float = 0.0,
    phase_rate_turns: float | None = None,
    sub_phase_rate_turns: float | None = None,
    cancel_amps: list[float] | None = None,
    init_phases: list[float] | None = None,
    cal_stage_repetitions: list[int] | int | None = None,
    cal_repetitions: list[int] | int | None = None,
    cal_depth: int = 20,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    seed: int | None = None,
    simulate: bool = False,
    conditions: list[str] | None = None,
    strict_timing: bool = True,
    bidirectional: bool = False,
    drive_cancel_amp: float = 0.0,
    drive_init_phase: float = 0.0,
    reverse_sub_phase_rate_turns: float | None = None,
    reverse_track: bool | None = None,
    read_both: bool | None = None,
    readout_scheme: str = "separate_passes",
    read_target: str | None = None,
    stark_frame_rad: float = 0.0,
    drive_stark_frame_rad: float = 0.0,
    phase_reset: str = "global",
    cancel_tone: str = "victim",
    sequence_mode: str = "unrolled",
    fpga_unroll: int = 1,
    fpga_dispatch: str = "parallel",
    fpga_sequence_source: str = "fpga_rng",
):
    """Build the QUA program for Crosstalk Compensated Simultaneous SQRB.

    bidirectional (benchmark only): a second cancel element on the drive qubit's line mirrors
    the probe's gates, both qubits are read out (streams I1/I2), and the isolated condition
    runs probe-alone then drive-alone within each shot so every qubit has its own isolated RB.

    phase_reset: 'global' = reset_if_phase on every xy element + reset_global_phase();
    'if_only' drops reset_global_phase(), which on chipQM is what makes an element's phase
    depend on which resonators the program reads (the multiplexed-readout phase slip).

    cancel_tone: 'aggressor' = the cancel elements output at the aggressor's frequency
    (config from _patch_cancel_element(tone='aggressor')): benchmark plays them with no frame
    steps; calibrate turns them with the drive's frame-freezing steps.

    sequence_mode (benchmark): 'unrolled' plays every Clifford as its own QUA statements;
    'fpga' stores the sequences in QUA arrays and plays one composite 3-subslot op per
    Clifford and element (config from _patch_composite_cliffords) from an FPGA loop, so the
    program size does not grow with depth. The loop is not strict-timed: each Clifford takes
    48 ns + switch_ dispatch on every element alike, which only cancel_tone='aggressor' tolerates.
    """
    if phase_reset not in ("global", "if_only"):
        raise ValueError(f"unknown phase_reset {phase_reset!r}")
    if sequence_mode not in ("unrolled", "fpga"):
        raise ValueError(f"unknown sequence_mode {sequence_mode!r}")
    if fpga_sequence_source not in ("fpga_rng", "host"):
        raise ValueError(f"unknown fpga_sequence_source {fpga_sequence_source!r}")
    if fpga_dispatch not in ("parallel", "per_element", "merged"):
        raise ValueError(f"unknown fpga_dispatch {fpga_dispatch!r}")
    aggressor_tone = cancel_tone == "aggressor"
    depth_arr = np.asarray(depths, dtype=int) if depths is not None else np.array([4])
    if sequence_indices is not None:
        seq_arr = np.asarray(sequence_indices, dtype=int)
    else:
        seq_arr = np.arange(num_sequences, dtype=int)
    cancel_elem = f"{probe_qubit.xy.name}_cancel"
    bidirectional = bool(bidirectional) and mode == "benchmark"
    # Diagnostics: each half of the bidirectional program can be switched off on its own
    use_reverse = bidirectional and (reverse_track is None or bool(reverse_track))
    read_drive = bidirectional and (read_both is None or bool(read_both))
    reverse_cancel_elem = f"{drive_qubit.xy.name}_cancel" if use_reverse else None
    read_qubits = [probe_qubit, drive_qubit] if read_drive else [probe_qubit]
    # read_target: this program reads ONE qubit ("probe" or "drive"); acquisition runs one
    # program per qubit and merges them (separate_passes without doubling the program)
    if read_drive and read_target in ("probe", "drive"):
        read_qubits = [probe_qubit] if read_target == "probe" else [drive_qubit]

    if cal_stage_repetitions is not None:
        cal_reps = [int(cal_stage_repetitions)] if isinstance(cal_stage_repetitions, (int, np.integer)) else [int(r) for r in cal_stage_repetitions]
    elif cal_repetitions is not None:
        cal_reps = [int(cal_repetitions)] if isinstance(cal_repetitions, (int, np.integer)) else [int(r) for r in cal_repetitions]
    else:
        cal_reps = [int(cal_depth)]

    max_circuit_depth = int(depth_arr.max()) if mode == "benchmark" else int(cal_reps[0])

    try:
        x180_len = probe_qubit.xy.operations["x180"].length
        pulse_cycles = max(x180_len // 4, 1)
    except Exception:
        pulse_cycles = 4

    if sub_phase_rate_turns is not None:
        sub_p = float(sub_phase_rate_turns)
    elif phase_rate_turns is not None:
        sub_p = float(phase_rate_turns)
    else:
        sub_p = 0.0
    # Reverse direction sees the opposite detuning (f_p - f_d)
    rev_sub_p = float(reverse_sub_phase_rate_turns) if reverse_sub_phase_rate_turns is not None else -sub_p
    # Calibrate freezes the crosstalk axis by turning the drive's frame -sub_p per pulse; an
    # aggressor-tone cancel element follows the drive's frame. In benchmark an aggressor-tone
    # cancel element is already phase-locked to the crosstalk: no frame steps.
    cal_cancel_step = -sub_p if aggressor_tone else 0.0
    if aggressor_tone:
        sub_p_cancel, rev_sub_p_cancel = 0.0, 0.0
    else:
        sub_p_cancel, rev_sub_p_cancel = sub_p, rev_sub_p
    use_fpga = sequence_mode == "fpga" and mode == "benchmark"
    if use_fpga:
        if "compensated" in (conditions or ["compensated"]) and not aggressor_tone:
            raise ValueError(
                "sequence_mode='fpga' needs cancel_tone='aggressor' for the compensated condition: "
                "the FPGA loop leaves dispatch gaps between Cliffords that per-subslot frame steps "
                "of a victim-tone cancel element cannot follow"
            )
        if stark_frame_rad or drive_stark_frame_rad:
            raise ValueError("sequence_mode='fpga' does not support stark_frame_rad")

    cond_name_map = {"isolated": 0, "simultaneous": 1, "compensated": 2}
    if conditions is not None:
        benchmark_conditions = [c for c in conditions if c in cond_name_map]
    else:
        benchmark_conditions = ["isolated", "simultaneous", "compensated"]
    cond_indices = [cond_name_map[c] for c in benchmark_conditions]

    if mode == "calibrate":
        sweep_axes = {
            "qubit": xr.DataArray([probe_qubit.name]),
            "cancel_amp": xr.DataArray(np.asarray(cancel_amps, dtype=float)),
            "init_phase": xr.DataArray(np.asarray(init_phases, dtype=float)),
        }
    else:
        sweep_axes = {
            "qubit": xr.DataArray([q.name for q in read_qubits]),
            "condition": xr.DataArray(np.asarray(benchmark_conditions, dtype=str)),
            "sequence_idx": xr.DataArray(seq_arr),
            "depth": xr.DataArray(depth_arr),
        }

    with program() as prog:
        if len(read_qubits) > 1:
            I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables(num_IQ_pairs=len(read_qubits))
        else:
            I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        if use_state_discrimination:
            state = [declare(int) for _ in read_qubits]
            state_st = [declare_stream() for _ in read_qubits]

        m = declare(int)
        m_st = declare_stream()

        amp_fixed = declare(fixed)
        init_phase_fixed = declare(fixed)

        if use_fpga:
            # arrays the played conditions need: per-qubit for single-qubit passes (and
            # per_element dispatch), merged for two-qubit passes under merged dispatch
            played = set()
            for c in benchmark_conditions:
                played.add(c)
                if c == "isolated" and read_drive and read_target in (None, "drive"):
                    played.add("isolated_drive")
            if read_drive and read_target == "drive":
                played.discard("isolated")
            arrays = set()
            for c in played:
                if _fpga_uses_merged(c, fpga_dispatch):
                    arrays.add("c")
                else:
                    if c != "isolated_drive":
                        arrays.add("p")
                    if c != "isolated":
                        arrays.add("d")
            if fpga_sequence_source == "fpga_rng":
                fpga = _declare_fpga_rng(
                    depth_arr, seq_arr, seed=seed, unroll=fpga_unroll,
                    arrays=tuple(k for k in ("p", "d", "c") if k in arrays),
                )
            else:
                fpga = _declare_fpga_sequences(
                    depth_arr, seq_arr, num_sequences=num_sequences, seed=seed, unroll=fpga_unroll,
                    arrays=tuple(k for k in ("p", "d", "c") if k in arrays),
                )
        elif mode == "benchmark":
            total_seqs = max(int(num_sequences), int(seq_arr.max()) + 1)
            seqs_p = {
                int(d): generate_clifford_sequences(int(d), total_seqs, seed=seed if seed is not None else 42)
                for d in depth_arr
            }
            seqs_d = {
                int(d): generate_clifford_sequences(
                    int(d), total_seqs, seed=(seed + 1000) if seed is not None else 1042
                )
                for d in depth_arr
            }

        machine.initialize_qpu(target=probe_qubit)
        machine.initialize_qpu(target=drive_qubit)
        align()

        if mode == "calibrate":
            # Calibrate mode: sweep cancel_amp, init_phase, and average over cal_reps with probe idle in |0>
            with for_each_(amp_fixed, [float(a) for a in cancel_amps]):
                with for_each_(init_phase_fixed, [float(p) / (2 * np.pi) for p in init_phases]):
                    for r_val in cal_reps:
                        r_int = int(r_val)
                        with for_(n, 0, n < num_shots, n + 1):
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            if strict_timing:
                                with strict_timing_():
                                    align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                                    reset_frame(probe_qubit.xy.name)
                                    reset_frame(drive_qubit.xy.name)
                                    reset_frame(cancel_elem)
                                    reset_if_phase(probe_qubit.xy.name)
                                    reset_if_phase(drive_qubit.xy.name)
                                    reset_if_phase(cancel_elem)
                                    if phase_reset == "global":
                                        reset_global_phase()
                                    frame_rotation_2pi(init_phase_fixed, cancel_elem)

                                    for _ in range(r_int):
                                        drive_qubit.xy.play(target_gate)
                                        play(target_gate * amp(amp_fixed), cancel_elem)
                                        if sub_p != 0.0:
                                            frame_rotation_2pi(-sub_p, drive_qubit.xy.name)
                                        if cal_cancel_step != 0.0:
                                            frame_rotation_2pi(cal_cancel_step, cancel_elem)
                            else:
                                align(probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem)
                                reset_frame(probe_qubit.xy.name)
                                reset_frame(drive_qubit.xy.name)
                                reset_frame(cancel_elem)
                                reset_if_phase(probe_qubit.xy.name)
                                reset_if_phase(drive_qubit.xy.name)
                                reset_if_phase(cancel_elem)
                                if phase_reset == "global":
                                    reset_global_phase()
                                frame_rotation_2pi(init_phase_fixed, cancel_elem)

                                for _ in range(r_int):
                                    drive_qubit.xy.play(target_gate)
                                    play(target_gate * amp(amp_fixed), cancel_elem)
                                    if sub_p != 0.0:
                                        frame_rotation_2pi(-sub_p, drive_qubit.xy.name)
                                    if cal_cancel_step != 0.0:
                                        frame_rotation_2pi(cal_cancel_step, cancel_elem)

                            align()
                            if use_state_discrimination:
                                probe_qubit.readout_state(state[0])
                                save(state[0], state_st[0])
                            else:
                                probe_qubit.resonator.measure("readout", qua_vars=(I[0], Q[0]))
                                save(I[0], I_st[0])
                                save(Q[0], Q_st[0])
                            align()

        else:
            # Benchmark mode: sweep depths across selected conditions with precomputed sequences
            assign(init_phase_fixed, float(init_phase / (2 * np.pi)))
            xy_elems = [probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem]
            if reverse_cancel_elem is not None:
                xy_elems.append(reverse_cancel_elem)
            timing_block = strict_timing_ if strict_timing else nullcontext

            for cond in benchmark_conditions:
                # (played condition, indices into read_qubits measured after it). Bidirectional
                # isolated runs probe-alone then drive-alone in the same shot, one qubit read each.
                if read_drive and read_target in ("probe", "drive"):
                    if cond == "isolated":
                        passes = [("isolated" if read_target == "probe" else "isolated_drive", [0])]
                    else:
                        passes = [(cond, [0])]
                elif read_drive and cond == "isolated":
                    passes = [("isolated", [0]), ("isolated_drive", [1])]
                elif read_drive and readout_scheme == "separate_passes":
                    # one qubit read per pass: the same sequence is played once per qubit
                    passes = [(cond, [0]), (cond, [1])]
                else:
                    passes = [(cond, list(range(len(read_qubits))))]

                if use_fpga:
                    _play_benchmark_fpga(
                        fpga,
                        passes,
                        probe_qubit=probe_qubit,
                        drive_qubit=drive_qubit,
                        read_qubits=read_qubits,
                        cancel_elem=cancel_elem,
                        reverse_cancel_elem=reverse_cancel_elem,
                        xy_elems=xy_elems,
                        cancel_amp=float(cancel_amp),
                        drive_cancel_amp=float(drive_cancel_amp),
                        init_phase=float(init_phase),
                        drive_init_phase=float(drive_init_phase),
                        phase_reset=phase_reset,
                        readout_scheme=readout_scheme,
                        num_shots=num_shots,
                        reset_type=reset_type,
                        simulate=simulate,
                        use_state_discrimination=use_state_discrimination,
                        I=I, Q=Q, I_st=I_st, Q_st=Q_st, n=n, m=m, m_st=m_st,
                        state=state if use_state_discrimination else None,
                        state_st=state_st if use_state_discrimination else None,
                        dispatch=fpga_dispatch,
                    )
                    continue

                for m_idx in seq_arr:
                    m_int = int(m_idx)
                    assign(m, m_int)
                    save(m, m_st)

                    for d_val in depth_arr:
                        d_int = int(d_val)
                        gate_list_p = seqs_p[d_int][m_int]
                        gate_list_d = seqs_d[d_int][m_int]

                        with for_(n, 0, n < num_shots, n + 1):
                            for play_cond, read_idx in passes:
                                probe_qubit.reset(reset_type, simulate)
                                drive_qubit.reset(reset_type, simulate)
                                with timing_block():
                                    align(*xy_elems)
                                    for el in xy_elems:
                                        reset_frame(el)
                                    for el in xy_elems:
                                        reset_if_phase(el)
                                    if phase_reset == "global":
                                        reset_global_phase()
                                    frame_rotation_2pi(init_phase_fixed, cancel_elem)
                                    if reverse_cancel_elem is not None:
                                        frame_rotation_2pi(
                                            float(drive_init_phase / (2 * np.pi)), reverse_cancel_elem
                                        )

                                    for k in range(d_int + 1):
                                        play_simultaneous_clifford_subslots(
                                            gate_list_p[k],
                                            gate_list_d[k],
                                            cond=play_cond,
                                            probe_qubit=probe_qubit,
                                            drive_qubit=drive_qubit,
                                            cancel_elem_name=cancel_elem,
                                            amp_val=float(cancel_amp),
                                            sub_phase_rate_turns=sub_p_cancel,
                                            pulse_cycles=pulse_cycles,
                                            reverse_cancel_elem_name=reverse_cancel_elem,
                                            reverse_amp_val=float(drive_cancel_amp),
                                            reverse_sub_phase_rate_turns=rev_sub_p_cancel,
                                            stark_turns_probe=float(stark_frame_rad) / (2 * np.pi),
                                            stark_turns_drive=(
                                                float(drive_stark_frame_rad) / (2 * np.pi) if bidirectional else 0.0
                                            ),
                                        )

                                align()
                                # 'simultaneous': multiplexed; 'sequential': one after the other
                                for j, i in enumerate(read_idx):
                                    if j > 0 and readout_scheme == "sequential":
                                        align()
                                    if use_state_discrimination:
                                        read_qubits[i].readout_state(state[i])
                                        save(state[i], state_st[i])
                                    else:
                                        read_qubits[i].resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                        save(I[i], I_st[i])
                                        save(Q[i], Q_st[i])
                                align()

        with stream_processing():
            if mode == "calibrate":
                if use_state_discrimination:
                    state_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(cal_reps)).map(FUNCTIONS.average()).buffer(len(init_phases)).buffer(len(cancel_amps)).save("state1")
                else:
                    I_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(cal_reps)).map(FUNCTIONS.average()).buffer(len(init_phases)).buffer(len(cancel_amps)).save("I1")
                    Q_st[0].buffer(num_shots).map(FUNCTIONS.average()).buffer(len(cal_reps)).map(FUNCTIONS.average()).buffer(len(init_phases)).buffer(len(cancel_amps)).save("Q1")
            else:
                m_st.save("n")
                num_conds = len(cond_indices)
                num_seqs = len(seq_arr)
                num_depths = len(depth_arr)
                for i in range(len(read_qubits)):
                    if use_state_discrimination:
                        state_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_seqs).buffer(num_conds).save(f"state{i + 1}")
                    else:
                        I_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_seqs).buffer(num_conds).save(f"I{i + 1}")
                        Q_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_seqs).buffer(num_conds).save(f"Q{i + 1}")

    return prog, sweep_axes


def acquire(
    machine,
    prog,
    sweep_axes,
    *,
    num_shots: int,
    timeout: float = 120.0,
    log: bool = False,
    config: Optional[dict] = None,
) -> xr.Dataset:
    """Acquire CrosstalkCompensatedSQRB dataset."""
    return _acquire(
        machine,
        prog,
        sweep_axes,
        num_shots=num_shots,
        timeout=timeout,
        log=log,
        config=config,
    )


# Transient QOP failures seen on the lab cluster mid-run ("Internal error ... Compilation
# failed", RPC/stream drops). A long benchmark is dozens of chunks; one of these must not
# throw away every chunk already measured.
_TRANSIENT_QOP_ERRORS = ("Internal error", "Compilation failed", "Unexpected error in RPC", "Connection lost")


def _acquire_chunk_with_retry(machine, prog, sweep_axes, *, attempts: int = 3, pause_s: float = 30.0, **kw):
    """_acquire one chunk, retrying the same program on a transient QOP error."""
    import time

    for attempt in range(1, attempts + 1):
        try:
            return _acquire(machine, prog, sweep_axes, **kw)
        except Exception as err:  # noqa: BLE001 - classified by message below
            msg = f"{type(err).__name__}: {err}"
            if attempt == attempts or not any(t in msg for t in _TRANSIENT_QOP_ERRORS):
                raise
            print(f"# [chunk retry {attempt}/{attempts - 1}] transient QOP error, retrying in {pause_s:.0f} s: {msg[:160]}")
            time.sleep(pause_s)


def _patch_cancel_element(
    config: dict,
    victim_qubit: Any,
    aggressor_qubit: Any,
    target_gate: str = "x180",
    tone: str = "victim",
) -> dict:
    """Add a cancellation element on the victim's line with an independent hardware core.

    The element is a copy of the victim's xy element (victim port, IF and frame) whose
    Clifford operations all point at the AGGRESSOR's pulses, so the compensation mirrors the
    crosstalk waveform itself: every gate the aggressor plays is cancelled with its own
    shape and relative amplitude, and cancel_amp is one scale for all of them.

    tone='victim' keeps the victim's IF (the crosstalk phase is then followed by frame steps of
    sub_p per 16 ns subslot). tone='aggressor' moves the IF so the element outputs at the
    aggressor's RF frequency on the victim's line: a scaled copy of the leaked field itself,
    phase-locked to it in absolute time, so it needs no frame steps and tolerates timing gaps.

    On OPX1000 architectures, elements bound to the same core cannot execute pulses
    concurrently (they are serialized). Assigning cancel_elem an unused core on the
    same FEM allows cancel_elem and victim_qubit.xy to output simultaneously. Called once per
    direction; a second call sees the first cancel element's core as used.
    """
    cancel_elem_name = f"{victim_qubit.xy.name}_cancel"
    if cancel_elem_name in config.get("elements", {}):
        return config

    base_elem = config["elements"].get(victim_qubit.xy.name)
    if base_elem is None:
        alt = victim_qubit.xy.name.replace("_", ".") if "_" in victim_qubit.xy.name else victim_qubit.xy.name.replace(".", "_")
        base_elem = config["elements"].get(alt, {})
    elem_dict = copy.deepcopy(base_elem)

    # Every op the compensation can play inherits the aggressor's pulse definition
    aggressor_name = aggressor_qubit.name if hasattr(aggressor_qubit, "name") else str(aggressor_qubit)
    aggressor_elem_name = aggressor_qubit.xy.name if hasattr(aggressor_qubit, "xy") else f"{aggressor_name}.xy"
    aggressor_elem = config["elements"].get(aggressor_elem_name)
    if aggressor_elem is None:
        alt_aggr = aggressor_elem_name.replace("_", ".") if "_" in aggressor_elem_name else aggressor_elem_name.replace(".", "_")
        aggressor_elem = config["elements"].get(alt_aggr, {})
    aggressor_ops = (aggressor_elem or {}).get("operations", {})
    for op in (*CANCEL_MIRROR_OPS, target_gate):
        if op in aggressor_ops:
            elem_dict.setdefault("operations", {})[op] = copy.deepcopy(aggressor_ops[op])

    # Assign an independent hardware core on the same FEM
    fem_port = None
    if "MWInput" in elem_dict and isinstance(elem_dict["MWInput"], dict):
        fem_port = elem_dict["MWInput"].get("port")
    elif "mixInputs" in elem_dict and isinstance(elem_dict["mixInputs"], dict):
        fem_port = elem_dict["mixInputs"].get("I")

    if fem_port and len(fem_port) >= 2:
        fem_id = fem_port[:2]
        used_cores = set()
        for el in config["elements"].values():
            if not isinstance(el, dict):
                continue
            core = el.get("core") or el.get("thread")
            el_port = None
            if "MWInput" in el and isinstance(el["MWInput"], dict):
                el_port = el["MWInput"].get("port")
            elif "mixInputs" in el and isinstance(el["mixInputs"], dict):
                el_port = el["mixInputs"].get("I")
            if core and el_port and len(el_port) >= 2 and el_port[:2] == fem_id:
                used_cores.add(core)

        # Standard core candidates on OPX1000 FEMs: 'a' through 'h'
        core_candidates = ["a", "b", "c", "d", "e", "f", "g", "h"]
        available_cores = [c for c in core_candidates if c not in used_cores]
        if available_cores:
            new_core = available_cores[0]
        else:
            new_core = "f" if elem_dict.get("core") != "f" else "g"

        if "core" in elem_dict:
            elem_dict["core"] = new_core
        if "thread" in elem_dict:
            elem_dict["thread"] = new_core

    if tone == "aggressor":
        shift = float(aggressor_qubit.xy.RF_frequency) - float(victim_qubit.xy.RF_frequency)
        new_if = float(elem_dict["intermediate_frequency"]) + shift
        if abs(new_if) > 400e6:
            raise ValueError(
                f"cancel_tone='aggressor' needs IF {new_if / 1e6:.1f} MHz on {victim_qubit.xy.name}'s "
                "line, outside the MW-FEM +-400 MHz band"
            )
        elem_dict["intermediate_frequency"] = new_if
    elif tone != "victim":
        raise ValueError(f"unknown cancel tone {tone!r}")

    config["elements"][cancel_elem_name] = elem_dict
    return config


COMPOSITE_PREFIX = "clif"


def _op_iq(config: dict, element: str, op: str) -> np.ndarray:
    """Complex I+jQ samples of `op` on `element` in a generated config."""
    pulse = config["pulses"][config["elements"][element]["operations"][op]]
    parts = []
    for key in ("I", "Q"):
        wf = config["waveforms"][pulse["waveforms"][key]]
        if wf["type"] == "arbitrary":
            parts.append(np.asarray(wf["samples"], dtype=float))
        else:
            parts.append(np.full(int(pulse["length"]), float(wf["sample"])))
    return parts[0] + 1j * parts[1]


def _patch_composite_cliffords(config: dict, element: str, source_element: str | None = None) -> dict:
    """Add ops clif0..clif23 to `element`: each Clifford's 3 subslots as ONE waveform.

    Subslot j holds source_element's pulse for CLIFFORD_3SUBSLOTS[g][j] (zeros when idle), so
    a Clifford is a single 3-pulse-long play - what lets an FPGA loop pick it with one switch_.
    A cancel element passes the aggressor's xy element as source (played * amp(cancel_amp)).
    """
    source_element = source_element or element
    pulse_len = len(_op_iq(config, source_element, "x180"))
    for g, subslots in enumerate(CLIFFORD_3SUBSLOTS):
        parts = [
            np.zeros(pulse_len, complex) if slot is None else _op_iq(config, source_element, slot[0])
            for slot in subslots
        ]
        if any(len(p) != pulse_len for p in parts):
            raise ValueError(f"{source_element}: Clifford subslot pulses differ in length")
        z = np.concatenate(parts)
        name = f"{element}.{COMPOSITE_PREFIX}{g}"
        config["waveforms"][f"{name}.wf.I"] = {"type": "arbitrary", "samples": z.real.tolist()}
        config["waveforms"][f"{name}.wf.Q"] = {"type": "arbitrary", "samples": z.imag.tolist()}
        config["pulses"][f"{name}.pulse"] = {
            "operation": "control",
            "length": int(len(z)),
            "waveforms": {"I": f"{name}.wf.I", "Q": f"{name}.wf.Q"},
        }
        config["elements"][element]["operations"][f"{COMPOSITE_PREFIX}{g}"] = f"{name}.pulse"
    return config


@register
class QMCrosstalkCompensatedSQRB(CrosstalkCompensatedSQRB):
    """Build Crosstalk Compensated SQRB QUA program on QM OPX."""

    def _select_pair(self):
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        probe_name = self.params.probe_qubit
        drive_name = self.params.drive_qubit
        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        return q_dict.get(probe_name, qubits[0]), q_dict.get(drive_name, qubits[1])

    def _phase_rates_turns(self, probe_qubit, drive_qubit) -> tuple[float, float]:
        """Per-subslot phase rates in turns, wrapped to (-0.5, 0.5]: (drive->probe, probe->drive)."""
        # Detect atomic pulse duration from calibrated probe x180 length
        try:
            x180_op = probe_qubit.xy.operations["x180"]
            pulse_duration_ns = float(getattr(x180_op, "length", self.params.clifford_duration_ns))
        except Exception:
            pulse_duration_ns = float(self.params.clifford_duration_ns)

        # Determine theoretical subslot phase rate using full RF frequencies and pulse duration
        if self.params.phase_rate is None:
            f_d = getattr(drive_qubit.xy, "RF_frequency", None) or getattr(drive_qubit, "f_01", None) or getattr(drive_qubit.xy, "intermediate_frequency", 50e6)
            f_p = getattr(probe_qubit.xy, "RF_frequency", None) or getattr(probe_qubit, "f_01", None) or getattr(probe_qubit.xy, "intermediate_frequency", 50e6)
            sub_phase_rate_rad = compute_theoretical_phase_rate(
                float(f_d), float(f_p), pulse_duration_ns
            )
        else:
            sub_phase_rate_rad = float(self.params.phase_rate)

        # Reverse direction sees the opposite detuning (f_p - f_d)
        if self.params.drive_phase_rate is None:
            rev_phase_rate_rad = -sub_phase_rate_rad
        else:
            rev_phase_rate_rad = float(self.params.drive_phase_rate)

        def _wrap(rad: float) -> float:
            turns = float((rad / (2 * np.pi)) % 1.0)
            return turns - 1.0 if turns > 0.5 else turns

        return _wrap(sub_phase_rate_rad), _wrap(rev_phase_rate_rad)

    def _bidirectional_kwargs(self, rev_sub_phase_rate_turns: float) -> dict[str, Any]:
        return {
            "bidirectional": bool(self.params.bidirectional),
            "drive_cancel_amp": float(self.params.drive_cancel_amp),
            "drive_init_phase": float(self.params.drive_init_phase),
            "reverse_sub_phase_rate_turns": rev_sub_phase_rate_turns,
            "reverse_track": self.params.reverse_track,
            "read_both": self.params.read_both,
            "readout_scheme": self.params.readout_scheme,
            "stark_frame_rad": float(self.params.stark_frame_rad),
            "drive_stark_frame_rad": float(self.params.drive_stark_frame_rad),
            "phase_reset": self.params.phase_reset,
            "cancel_tone": self.params.cancel_tone,
            "sequence_mode": self.params.sequence_mode,
            "fpga_unroll": int(self.params.fpga_unroll),
            "fpga_dispatch": self.params.fpga_dispatch,
            "fpga_sequence_source": self.params.fpga_sequence_source,
        }

    def _patch_config(self, config: dict, probe_qubit, drive_qubit) -> dict:
        target_gate = getattr(self.params, "target_gate", "x180")
        tone = self.params.cancel_tone
        pairs = [(probe_qubit, drive_qubit)]
        if self.params.uses_reverse_track():
            pairs.append((drive_qubit, probe_qubit))
        # Cancel elements take the next free FEM core in creation order and the phase reference
        # is per core: create them in victim-name order so the bidirectional tomography
        # (same rule) calibrates each element on the core it runs on here.
        for victim, aggressor in sorted(pairs, key=lambda p: p[0].xy.name):
            config = _patch_cancel_element(config, victim, aggressor, target_gate, tone=tone)
        if self.params.sequence_mode == "fpga" and self.params.mode == "benchmark":
            config = _patch_composite_cliffords(config, probe_qubit.xy.name)
            config = _patch_composite_cliffords(config, drive_qubit.xy.name)
            config = _patch_composite_cliffords(config, f"{probe_qubit.xy.name}_cancel", drive_qubit.xy.name)
            if self.params.uses_reverse_track():
                config = _patch_composite_cliffords(config, f"{drive_qubit.xy.name}_cancel", probe_qubit.xy.name)
        return config

    def patch_preview_config(self, config: dict) -> dict:
        """Patch the cancellation element(s) into the generated config for preview."""
        config = copy.deepcopy(config)
        probe_qubit, drive_qubit = self._select_pair()
        return self._patch_config(config, probe_qubit, drive_qubit)

    def probe(self) -> Any:
        from ._reset import check_reset_method

        machine = self.backend.machine
        probe_qubit, drive_qubit = self._select_pair()
        sub_phase_rate_turns, rev_sub_phase_rate_turns = self._phase_rates_turns(probe_qubit, drive_qubit)

        target_gate = getattr(self.params, "target_gate", "x180")
        config = machine.generate_config()
        config = self._patch_config(config, probe_qubit, drive_qubit)

        stage_reps = list(self.params.get_cal_stage_repetitions()) if self.params.mode == "calibrate" else [20]
        initial_cal_reps = [stage_reps[0]] if self.params.mode == "calibrate" else [int(getattr(self.params, "cal_depth", 20))]

        all_depths = list(self.params.get_depths())
        batch_size = max(1, getattr(self.params, "benchmark_batch_size", 1))
        initial_depths = all_depths[:batch_size] if self.params.mode == "benchmark" else all_depths

        prog, sweep_axes = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            target_gate=target_gate,
            depths=initial_depths,
            conditions=self.params.conditions if self.params.mode == "benchmark" else None,
            num_sequences=int(self.params.num_random_sequences),
            num_shots=int(self.params.num_averages),
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            phase_rate_turns=sub_phase_rate_turns,
            sub_phase_rate_turns=sub_phase_rate_turns,
            cancel_amps=list(self.params.get_cancel_amps()),
            init_phases=list(self.params.get_init_phases()),
            cal_stage_repetitions=initial_cal_reps,
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            seed=self.params.seed,
            simulate=getattr(self.backend, "simulate", False),
            strict_timing=bool(getattr(self.params, "strict_timing", True)),
            **self._bidirectional_kwargs(rev_sub_phase_rate_turns),
        )

        if self.params.mode == "calibrate":
            return (
                prog,
                sweep_axes,
                lambda m, p, s, **kw: self._acquire_calibrate(
                    m,
                    p,
                    s,
                    config=config,
                    probe_qubit=probe_qubit,
                    drive_qubit=drive_qubit,
                    reset_type=check_reset_method(self),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    **kw,
                ),
            )

        return (
            prog,
            sweep_axes,
            lambda m, p, s, **kw: self._acquire_benchmark(
                m,
                p,
                s,
                config=config,
                probe_qubit=probe_qubit,
                drive_qubit=drive_qubit,
                reset_type=check_reset_method(self),
                sub_phase_rate_turns=sub_phase_rate_turns,
                rev_sub_phase_rate_turns=rev_sub_phase_rate_turns,
                **kw,
            ),
        )

    def preview_program(self):
        """Render a single representative program for preview without memory bloat."""
        from ._reset import check_reset_method

        machine = self.backend.machine
        probe_qubit, drive_qubit = self._select_pair()
        sub_phase_rate_turns, rev_sub_phase_rate_turns = self._phase_rates_turns(probe_qubit, drive_qubit)

        target_gate = getattr(self.params, "target_gate", "x180")
        if self.params.mode == "benchmark":
            param_conds = self.params.conditions or ["compensated"]
            preview_conds = ["compensated"] if "compensated" in param_conds else [param_conds[0]]
        else:
            preview_conds = None
        preview_depths = [4] if self.params.mode == "benchmark" else [20]
        preview_num_seq = 1 if self.params.mode == "benchmark" else int(self.params.num_random_sequences)
        preview_cal_reps = [20] if self.params.mode == "calibrate" else None

        prog, _ = build_program(
            machine,
            probe_qubit,
            drive_qubit,
            mode=self.params.mode,
            target_gate=target_gate,
            conditions=preview_conds,
            depths=preview_depths,
            num_sequences=preview_num_seq,
            cal_stage_repetitions=preview_cal_reps,
            num_shots=1,
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            sub_phase_rate_turns=sub_phase_rate_turns,
            cancel_amps=[float(self.params.cancel_amp)],
            init_phases=[float(self.params.init_phase)],
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            simulate=getattr(self.backend, "simulate", False),
            strict_timing=bool(getattr(self.params, "strict_timing", True)),
            **self._bidirectional_kwargs(rev_sub_phase_rate_turns),
        )
        return prog

    def _acquire_benchmark(
        self,
        machine,
        prog,
        sweep_axes,
        *,
        num_shots: int,
        timeout: float = 120.0,
        log: bool = False,
        config: Optional[dict] = None,
        probe_qubit=None,
        drive_qubit=None,
        reset_type: str = "thermal",
        sub_phase_rate_turns: float = 0.0,
        rev_sub_phase_rate_turns: float | None = None,
    ) -> xr.Dataset:
        """Execute fine-grained batched benchmark mode on Quantum Machine.

        Chunks the workload across condition, depth, and sequence dimensions to ensure
        fast compilation and stay well within OPX memory and timeout limits.
        """
        cond_map = ["isolated", "simultaneous", "compensated"]
        param_conds = getattr(self.params, "conditions", None)
        benchmark_conditions = [c for c in param_conds if c in cond_map] if param_conds else cond_map

        all_depths = list(self.params.get_depths())
        batch_size = max(1, getattr(self.params, "benchmark_batch_size", 1))
        depth_batches = [all_depths[i : i + batch_size] for i in range(0, len(all_depths), batch_size)]
        fpga = self.params.sequence_mode == "fpga"
        if fpga:
            # FPGA loop: program size does not grow with depth, every depth goes in one program
            depth_batches = [all_depths]

        total_sequences = int(self.params.num_random_sequences)
        max_gates = int(getattr(self.params, "max_gates_per_batch", 120))
        fixed_seq_batch_size = getattr(self.params, "benchmark_sequence_batch_size", None)
        target_gate = getattr(self.params, "target_gate", "x180")

        read_targets = (
            ["probe", "drive"]
            if self.params.reads_both() and self.params.readout_scheme == "separate_passes"
            else [None]
        )
        if fpga and self.params.fpga_sequence_source == "fpga_rng":
            # Sequences are drawn on the FPGA one at a time: no array-memory limit, so every
            # condition, sequence and depth goes into ONE program (one per read target).
            per_target = []
            for rt in read_targets:
                cur_prog, cur_axes = build_program(
                    machine,
                    probe_qubit,
                    drive_qubit,
                    mode="benchmark",
                    target_gate=target_gate,
                    conditions=benchmark_conditions,
                    depths=all_depths,
                    num_sequences=total_sequences,
                    num_shots=num_shots,
                    cancel_amp=float(self.params.cancel_amp),
                    init_phase=float(self.params.init_phase),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    reset_type=reset_type,
                    use_state_discrimination=bool(self.params.use_state_discrimination),
                    seed=self.params.seed,
                    simulate=getattr(self.backend, "simulate", False),
                    strict_timing=bool(getattr(self.params, "strict_timing", True)),
                    **self._bidirectional_kwargs(
                        -sub_phase_rate_turns if rev_sub_phase_rate_turns is None else rev_sub_phase_rate_turns
                    ),
                    read_target=rt,
                )
                per_target.append(
                    _acquire_chunk_with_retry(
                        machine, cur_prog, cur_axes, num_shots=num_shots, timeout=timeout, log=log, config=config
                    )
                )
                print(
                    f"# [Benchmark fpga_rng] {len(benchmark_conditions)} condition(s) x {total_sequences} "
                    f"sequences x {len(all_depths)} depths in one program"
                    + (f" (read {rt})" if rt else "") + " completed."
                )
            return per_target[0] if len(per_target) == 1 else xr.concat(per_target, dim="qubit")

        total_steps = len(benchmark_conditions) * len(depth_batches)
        current_step = 0
        cond_datasets: list[xr.Dataset] = []

        for c_idx, cond in enumerate(benchmark_conditions, 1):
            depth_datasets: list[xr.Dataset] = []
            for d_idx, d_batch in enumerate(depth_batches, 1):
                current_step += 1
                max_d = max(int(d) for d in d_batch)
                # Bidirectional doubles the program per sequence: the isolated condition runs
                # two passes, compensated plays two cancel tracks, and separate_passes readout
                # plays every sequence once per qubit.
                gates_per_seq = (max_d + 1) * (2 if self.params.bidirectional else 1)

                if fixed_seq_batch_size is not None and int(fixed_seq_batch_size) > 0:
                    seq_chunk_size = int(fixed_seq_batch_size)
                elif fpga:
                    # bounded by the QUA array memory: two gate arrays of max depth per sequence
                    seq_chunk_size = max(
                        1, min(total_sequences, int(self.params.fpga_max_array_gates) // (2 * max_d))
                    )
                else:
                    seq_chunk_size = max(1, min(total_sequences, max_gates // gates_per_seq))

                num_chunks = int(np.ceil(total_sequences / seq_chunk_size))
                seq_chunks = [
                    list(chunk)
                    for chunk in np.array_split(np.arange(total_sequences, dtype=int), num_chunks)
                ]

                read_targets = (
                    ["probe", "drive"]
                    if self.params.reads_both() and self.params.readout_scheme == "separate_passes"
                    else [None]
                )
                seq_datasets: dict = {rt: [] for rt in read_targets}
                for sc_idx, s_chunk in enumerate(seq_chunks, 1):
                    for rt in read_targets:
                        cur_prog, cur_axes = build_program(
                            machine,
                            probe_qubit,
                            drive_qubit,
                            mode="benchmark",
                            target_gate=target_gate,
                            conditions=[cond],
                            depths=d_batch,
                            num_sequences=total_sequences,
                            sequence_indices=s_chunk,
                            num_shots=num_shots,
                            cancel_amp=float(self.params.cancel_amp),
                            init_phase=float(self.params.init_phase),
                            sub_phase_rate_turns=sub_phase_rate_turns,
                            reset_type=reset_type,
                            use_state_discrimination=bool(self.params.use_state_discrimination),
                            seed=self.params.seed,
                            simulate=getattr(self.backend, "simulate", False),
                            strict_timing=bool(getattr(self.params, "strict_timing", True)),
                            **self._bidirectional_kwargs(
                                -sub_phase_rate_turns if rev_sub_phase_rate_turns is None else rev_sub_phase_rate_turns
                            ),
                            read_target=rt,
                        )

                        ds_chunk = _acquire_chunk_with_retry(
                            machine,
                            cur_prog,
                            cur_axes,
                            num_shots=num_shots,
                            timeout=timeout,
                            log=log,
                            config=config,
                        )
                        seq_datasets[rt].append(ds_chunk)

                per_target = [xr.concat(seq_datasets[rt], dim="sequence_idx") for rt in read_targets]
                ds_depth = per_target[0] if len(per_target) == 1 else xr.concat(per_target, dim="qubit")
                depth_datasets.append(ds_depth)
                print(
                    f"# [Benchmark {current_step}/{total_steps}] "
                    f"Cond: '{cond}' ({c_idx}/{len(benchmark_conditions)}), "
                    f"Depth: {d_batch} ({d_idx}/{len(depth_batches)}), "
                    f"Sequences: {total_sequences} in {len(seq_chunks)} chunk(s) completed."
                )

            ds_cond = xr.concat(depth_datasets, dim="depth")
            cond_datasets.append(ds_cond)

        total_ds = xr.concat(cond_datasets, dim="condition")
        return total_ds

    def _acquire_calibrate(
        self,
        machine,
        prog,
        sweep_axes,
        *,
        num_shots: int,
        timeout: float = 120.0,
        log: bool = False,
        config: Optional[dict] = None,
        probe_qubit=None,
        drive_qubit=None,
        reset_type: str = "thermal",
        sub_phase_rate_turns: float = 0.0,
    ) -> xr.Dataset:
        """Execute iterative multi-stage zoom-in calibration on Quantum Machine."""
        stage_reps = list(self.params.get_cal_stage_repetitions())
        num_stages = len(stage_reps)

        if num_stages <= 1:
            return _acquire(
                machine,
                prog,
                sweep_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )

        cur_min_amp = float(self.params.min_cancel_amp)
        cur_max_amp = float(self.params.max_cancel_amp)
        cur_min_phase = float(self.params.min_init_phase)
        cur_max_phase = float(self.params.max_init_phase)
        num_amps = int(self.params.num_cancel_amps)
        num_phases = int(self.params.num_init_phases)
        zoom = float(self.params.zoom_factor)

        last_ds: xr.Dataset | None = None
        stage_history: list[dict[str, Any]] = []
        target_gate = getattr(self.params, "target_gate", "x180")

        for stage_idx, r in enumerate(stage_reps):
            amps_stage = list(np.linspace(cur_min_amp, cur_max_amp, num_amps))
            phases_stage = list(np.linspace(cur_min_phase, cur_max_phase, num_phases))

            if stage_idx == 0:
                cur_prog = prog
                cur_axes = sweep_axes
            else:
                cur_prog, cur_axes = build_program(
                    machine,
                    probe_qubit,
                    drive_qubit,
                    mode="calibrate",
                    target_gate=target_gate,
                    cal_stage_repetitions=[r],
                    num_shots=num_shots,
                    cancel_amp=float(self.params.cancel_amp),
                    init_phase=float(self.params.init_phase),
                    sub_phase_rate_turns=sub_phase_rate_turns,
                    cancel_amps=amps_stage,
                    init_phases=phases_stage,
                    reset_type=reset_type,
                    use_state_discrimination=bool(self.params.use_state_discrimination),
                    simulate=getattr(self.backend, "simulate", False),
                    strict_timing=bool(getattr(self.params, "strict_timing", True)),
                    phase_reset=self.params.phase_reset,
                    cancel_tone=self.params.cancel_tone,
                )

            ds_stage = _acquire(
                machine,
                cur_prog,
                cur_axes,
                num_shots=num_shots,
                timeout=timeout,
                log=log,
                config=config,
            )
            last_ds = ds_stage

            var_name = "state" if "state" in ds_stage.data_vars else ("population" if "population" in ds_stage.data_vars else "I")
            p_vals = np.squeeze(ds_stage[var_name].values)
            min_idx = np.unravel_index(np.argmin(p_vals), p_vals.shape)
            best_amp = float(ds_stage.coords["cancel_amp"].values[min_idx[0]])
            best_phase = float(ds_stage.coords["init_phase"].values[min_idx[1]])
            min_p = float(p_vals[min_idx])

            stage_info = {
                "stage": stage_idx + 1,
                "repetitions": int(r),
                "cancel_amps": [float(a) for a in amps_stage],
                "init_phases": [float(p) for p in phases_stage],
                "p_vals": p_vals.tolist(),
                "best_cancel_amp": best_amp,
                "best_init_phase": best_phase,
                "min_signal": min_p,
            }
            stage_history.append(stage_info)
            print(
                f"# [Calibrate Stage {stage_idx + 1}/{num_stages}] Reps={r}: "
                f"Best cancel_amp={best_amp:.4f}, init_phase={best_phase:.4f} rad, min response={min_p:.4f}"
            )

            # Zoom in search window for next stage
            if stage_idx < num_stages - 1:
                span_amp = (cur_max_amp - cur_min_amp) / zoom
                span_phase = (cur_max_phase - cur_min_phase) / zoom
                cur_min_amp = max(0.0, best_amp - span_amp / 2.0)
                cur_max_amp = best_amp + span_amp / 2.0
                cur_min_phase = best_phase - span_phase / 2.0
                cur_max_phase = best_phase + span_phase / 2.0

        if last_ds is not None:
            import json
            last_ds.attrs["stage_history"] = json.dumps(stage_history)
            return last_ds
        return _acquire(machine, prog, sweep_axes, num_shots=num_shots, timeout=timeout, log=log, config=config)

