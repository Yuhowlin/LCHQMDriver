"""Crosstalk Compensated Simultaneous SQRB acquisition probe (QM/QUAM).

QUA program generation and acquisition for CrosstalkCompensatedSQRB: isolated, simultaneous and
compensated single-qubit RB of a victim (probe) qubit next to an aggressor (drive) qubit.

How it runs (each choice was measured on chipQM hardware, 2026-10-01):

* Every Clifford is ONE 48 ns composite op (its 3 subslots of 16 ns concatenated, idle subslots
  as zeros), so a whole Clifford is a single play.
* Sequences live in QUA arrays and every element plays them from its OWN FPGA loop indexed by
  its own loop variable. Those loops run in parallel from a common align and play 48 ns
  Cliffords back to back (timestamp_stream). One shared loop with a switch_ per element took
  160 ns per Clifford; a computed index (gates[k + 1]) ~116 ns.
* The cancel element sits on the victim's line at the AGGRESSOR's frequency and plays a scaled,
  phase-shifted copy of the aggressor's pulses - a copy of the leaked field itself, so it needs
  no per-pulse frame steps and also removes the leak's AC-Stark shift.
* Every shot resets frames and IF phases plus reset_global_phase() (without it the phase
  between lines wanders shot to shot). Reading a qubit shifts its xy phase by one 4 ns IF cycle
  relative to other FEM cores, so cancel parameters are calibrated by
  crosstalk_cancellation_tomography in the same context (bidirectional + read_drive).
* Programs stay short (sequences_per_program): within one ~1 min program the cross-line error
  grew steadily.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table

from scqo import register
from scqo.experiments.crosstalk_compensated_sqrb import CrosstalkCompensatedSQRB
from scqo_qm.experiments._lib import acquire as _acquire

# Recovery gates lookup: inv_gates[i] is the Clifford gate index that resets state i to 0
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]

# 3-subslot decomposition for all 24 Clifford gates: (slot0, slot1, slot2)
# Each entry is None (idle) or (op_name, pulse_kind, axis_phase_turns)
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

N_CLIFFORDS = len(CLIFFORD_3SUBSLOTS)

# Every op a cancel element copies from its aggressor
CANCEL_MIRROR_OPS: tuple[str, ...] = tuple(
    sorted({slot[0] for subslots in CLIFFORD_3SUBSLOTS for slot in subslots if slot is not None})
)

COMPOSITE_PREFIX = "clif"

# QUA data memory holds ~16k gate entries per program (8 sequences x depth 1024 x 2 qubits
# compiled, 10 did not): bounds the sequences per program together with sequences_per_program.
MAX_ARRAY_GATES = 16000

CONDITIONS = ("isolated", "simultaneous", "compensated")


# --------------------------------------------------------------------------- sequences


def clifford_sequence(seed: int, seq_index: int, depth: int) -> list[int]:
    """`depth` random Cliffords drawn from the stream (seed, seq_index).

    Every shorter depth d plays the first d gates (recovery from recovery_gates), so one array
    per sequence serves all depths.
    """
    gates = np.random.default_rng([int(seed), int(seq_index)]).integers(0, N_CLIFFORDS, size=int(depth)).tolist()
    return [int(g) for g in gates]


def recovery_gates(gates: list[int], depths) -> list[int]:
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


def _declare_sequences(depth_arr, seq_arr, *, seed: int | None) -> dict:
    """QUA arrays and loop variables for one program (call inside the program).

    Probe sequences come from seed (default 42), drive sequences from seed + 1000 (default 1042).
    Flat layout: gates[i * d_max + k] and inv[i * n_d + j] for the i-th sequence of the program.
    """
    d_max = int(np.max(depth_arr))
    seed_p = seed if seed is not None else 42
    seed_d = (seed + 1000) if seed is not None else 1042
    out: dict[str, Any] = {"d_max": d_max, "n_d": len(depth_arr), "n_s": len(seq_arr)}
    for key, sd in (("p", seed_p), ("d", seed_d)):
        gates, inv = [], []
        for s_idx in seq_arr:
            seq = clifford_sequence(sd, int(s_idx), d_max)
            gates.extend(seq)
            inv.extend(recovery_gates(seq, depth_arr))
        out[f"gates_{key}"] = declare(int, value=gates)
        out[f"inv_{key}"] = declare(int, value=inv)
        out[f"g_inv_{key}"] = declare(int)
    out["depths"] = declare(int, value=[int(d) for d in depth_arr])
    out["seq_ids"] = declare(int, value=[int(s) for s in seq_arr])
    for name in ("s", "j", "d", "base", "k_stop"):
        out[name] = declare(int)
    out["k_el"] = [declare(int) for _ in range(4)]  # one loop variable per element
    return out


# --------------------------------------------------------------------------- playback


def _play_composite(gate, element: str, scale: float | None = None):
    """Play composite Clifford `gate` (QUA int expression) on element, optionally scaled."""
    with switch_(gate, unsafe=True):
        for g in range(N_CLIFFORDS):
            with case_(g):
                op = f"{COMPOSITE_PREFIX}{g}"
                play(op if scale is None else op * amp(scale), element)


def _played_elements(cond, *, probe_qubit, drive_qubit, cancel_elem, reverse_cancel_elem, cancel_amp, drive_cancel_amp):
    """(element, gate source 'p'/'d', amp scale or None) for every element a pass plays."""
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


def _passes(cond: str, bidirectional: bool) -> list[tuple[str, list[int]]]:
    """(played condition, indices of read qubits) per pass of one shot.

    Bidirectional isolated runs probe-alone then drive-alone, each qubit read after its own pass;
    simultaneous / compensated play once and read both qubits together.
    """
    if not bidirectional:
        return [(cond, [0])]
    if cond == "isolated":
        return [("isolated", [0]), ("isolated_drive", [1])]
    return [(cond, [0, 1])]


def build_program(
    machine,
    probe_qubit,
    drive_qubit,
    *,
    depths: list[int],
    num_shots: int,
    num_sequences: int = 1,
    sequence_indices: list[int] | np.ndarray | None = None,
    conditions: list[str] | None = None,
    cancel_amp: float = 0.0,
    init_phase: float = 0.0,
    bidirectional: bool = False,
    drive_cancel_amp: float = 0.0,
    drive_init_phase: float = 0.0,
    reset_type: str = "thermal",
    use_state_discrimination: bool = False,
    seed: int | None = None,
    simulate: bool = False,
    debug_timestamps: bool = False,
):
    """Build the crosstalk-compensated SQRB program (config: QMCrosstalkCompensatedSQRB._patch_config).

    Loops: condition (Python) > sequence > depth > shot > pass. Streams I{n}/Q{n} (or state{n})
    per read qubit, averaged over shots and shaped (condition, sequence, depth). With
    debug_timestamps every element's loop is bracketed by zero-amplitude markers with
    timestamp_stream (ts0_/ts1_<pass>_<element>, 4 ns cycles).
    """
    depth_arr = np.asarray(depths, dtype=int)
    if depth_arr.min() < 1:
        raise ValueError("depths must be >= 1")
    seq_arr = np.asarray(sequence_indices if sequence_indices is not None else np.arange(num_sequences), dtype=int)
    benchmark_conditions = [c for c in (conditions or CONDITIONS) if c in CONDITIONS]
    bidirectional = bool(bidirectional)

    cancel_elem = f"{probe_qubit.xy.name}_cancel"
    reverse_cancel_elem = f"{drive_qubit.xy.name}_cancel" if bidirectional else None
    xy_elems = [probe_qubit.xy.name, drive_qubit.xy.name, cancel_elem] + ([reverse_cancel_elem] if bidirectional else [])
    read_qubits = [probe_qubit, drive_qubit] if bidirectional else [probe_qubit]
    elem_kw = dict(
        probe_qubit=probe_qubit, drive_qubit=drive_qubit, cancel_elem=cancel_elem,
        reverse_cancel_elem=reverse_cancel_elem, cancel_amp=float(cancel_amp),
        drive_cancel_amp=float(drive_cancel_amp),
    )

    sweep_axes = {
        "qubit": xr.DataArray([q.name for q in read_qubits]),
        "condition": xr.DataArray(np.asarray(benchmark_conditions, dtype=str)),
        "sequence_idx": xr.DataArray(seq_arr),
        "depth": xr.DataArray(depth_arr),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables(num_IQ_pairs=len(read_qubits))
        if use_state_discrimination:
            state = [declare(int) for _ in read_qubits]
            state_st = [declare_stream() for _ in read_qubits]
        m = declare(int)
        m_st = declare_stream()
        seqs = _declare_sequences(depth_arr, seq_arr, seed=seed)
        s, j, d, base, k_stop = seqs["s"], seqs["j"], seqs["d"], seqs["base"], seqs["k_stop"]

        machine.initialize_qpu(target=probe_qubit)
        machine.initialize_qpu(target=drive_qubit)
        align()

        for cond in benchmark_conditions:
            passes = _passes(cond, bidirectional)
            with for_(s, 0, s < seqs["n_s"], s + 1):
                assign(m, seqs["seq_ids"][s])
                save(m, m_st)
                assign(base, s * seqs["d_max"])  # this sequence's first gate in the flat arrays
                with for_(j, 0, j < seqs["n_d"], j + 1):
                    assign(d, seqs["depths"][j])
                    # loop bound once per depth: `kk < base + d` costs an extra cycle per Clifford
                    assign(k_stop, base + d)
                    for key in ("p", "d"):
                        assign(seqs[f"g_inv_{key}"], seqs[f"inv_{key}"][s * seqs["n_d"] + j])
                    with for_(n, 0, n < num_shots, n + 1):
                        for play_cond, read_idx in passes:
                            probe_qubit.reset(reset_type, simulate)
                            drive_qubit.reset(reset_type, simulate)
                            align(*xy_elems)
                            for el in xy_elems:
                                reset_frame(el)
                            for el in xy_elems:
                                reset_if_phase(el)
                            reset_global_phase()
                            frame_rotation_2pi(float(init_phase) / (2 * np.pi), cancel_elem)
                            if reverse_cancel_elem is not None:
                                frame_rotation_2pi(float(drive_init_phase) / (2 * np.pi), reverse_cancel_elem)
                            align(*xy_elems)

                            elements = _played_elements(play_cond, **elem_kw)
                            if debug_timestamps:
                                for element, _, _ in elements:
                                    play("x180" * amp(0.0), element, timestamp_stream=f"ts0_{play_cond}_{element}")
                            # one loop per element, each on its own loop variable: the loops run
                            # side by side, Clifford i starts at the same time on every element
                            for idx, (element, src, scale) in enumerate(elements):
                                kk = seqs["k_el"][idx]
                                gates = seqs[f"gates_{src}"]
                                with for_(kk, base, kk < k_stop, kk + 1):
                                    _play_composite(gates[kk], element, scale)
                                _play_composite(seqs[f"g_inv_{src}"], element, scale)
                                if debug_timestamps:
                                    play("x180" * amp(0.0), element, timestamp_stream=f"ts1_{play_cond}_{element}")

                            align()
                            for i in read_idx:
                                if use_state_discrimination:
                                    read_qubits[i].readout_state(state[i])
                                    save(state[i], state_st[i])
                                else:
                                    read_qubits[i].resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                    save(I[i], I_st[i])
                                    save(Q[i], Q_st[i])
                            align()

        with stream_processing():
            m_st.save("n")
            n_c, n_s, n_d = len(benchmark_conditions), len(seq_arr), len(depth_arr)
            for i in range(len(read_qubits)):
                if use_state_discrimination:
                    state_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(n_d).buffer(n_s).buffer(n_c).save(f"state{i + 1}")
                else:
                    I_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(n_d).buffer(n_s).buffer(n_c).save(f"I{i + 1}")
                    Q_st[i].buffer(num_shots).map(FUNCTIONS.average()).buffer(n_d).buffer(n_s).buffer(n_c).save(f"Q{i + 1}")

    return prog, sweep_axes


# --------------------------------------------------------------------------- acquisition


# Transient QOP failures seen on the lab cluster mid-run ("Internal error ... Compilation
# failed", RPC/stream drops). A benchmark is many programs; one of these must not throw away
# every program already measured.
_TRANSIENT_QOP_ERRORS = ("Internal error", "Compilation failed", "Unexpected error in RPC", "Connection lost")


def _acquire_chunk_with_retry(machine, prog, sweep_axes, *, attempts: int = 3, pause_s: float = 30.0, **kw):
    """_acquire one program, retrying the same program on a transient QOP error."""
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


def sequence_chunks(num_sequences: int, max_depth: int, per_program: int) -> list[list[int]]:
    """Split the sequences into programs: at most per_program each, and within the array memory."""
    by_memory = max(1, MAX_ARRAY_GATES // (2 * int(max_depth)))
    size = max(1, min(int(num_sequences), int(per_program), by_memory))
    return [list(range(i, min(i + size, num_sequences))) for i in range(0, num_sequences, size)]


# --------------------------------------------------------------------------- config


def _patch_cancel_element(config: dict, victim_qubit: Any, aggressor_qubit: Any) -> dict:
    """Add `<victim xy>_cancel`: the victim's line, the aggressor's frequency and pulses, a free core.

    The element is a copy of the victim's xy element (same port) whose ops point at the
    AGGRESSOR's pulses and whose IF is moved by (f_aggressor - f_victim), so it outputs a copy
    of the aggressor's drive at the aggressor's frequency on the victim's line. Elements on one
    core cannot play at the same time, so it takes the next unused core on the FEM; the phase
    reference is per core, which is why callers create cancel elements in a fixed order.
    """
    cancel_elem_name = f"{victim_qubit.xy.name}_cancel"
    if cancel_elem_name in config.get("elements", {}):
        return config

    elem_dict = copy.deepcopy(config["elements"][victim_qubit.xy.name])
    aggressor_ops = config["elements"].get(aggressor_qubit.xy.name, {}).get("operations", {})
    for op in CANCEL_MIRROR_OPS:
        if op in aggressor_ops:
            elem_dict.setdefault("operations", {})[op] = copy.deepcopy(aggressor_ops[op])

    # Assign an independent hardware core on the same FEM
    fem_port = None
    if isinstance(elem_dict.get("MWInput"), dict):
        fem_port = elem_dict["MWInput"].get("port")
    elif isinstance(elem_dict.get("mixInputs"), dict):
        fem_port = elem_dict["mixInputs"].get("I")
    if fem_port and len(fem_port) >= 2:
        fem_id = fem_port[:2]
        used_cores = set()
        for el in config["elements"].values():
            if not isinstance(el, dict):
                continue
            core = el.get("core") or el.get("thread")
            el_port = None
            if isinstance(el.get("MWInput"), dict):
                el_port = el["MWInput"].get("port")
            elif isinstance(el.get("mixInputs"), dict):
                el_port = el["mixInputs"].get("I")
            if core and el_port and len(el_port) >= 2 and el_port[:2] == fem_id:
                used_cores.add(core)
        available = [c for c in "abcdefgh" if c not in used_cores]
        new_core = available[0] if available else ("f" if elem_dict.get("core") != "f" else "g")
        if "core" in elem_dict:
            elem_dict["core"] = new_core
        if "thread" in elem_dict:
            elem_dict["thread"] = new_core

    shift = float(aggressor_qubit.xy.RF_frequency) - float(victim_qubit.xy.RF_frequency)
    new_if = float(elem_dict["intermediate_frequency"]) + shift
    if abs(new_if) > 400e6:
        raise ValueError(
            f"the cancel element needs IF {new_if / 1e6:.1f} MHz on {victim_qubit.xy.name}'s line, "
            "outside the MW-FEM +-400 MHz band"
        )
    elem_dict["intermediate_frequency"] = new_if
    config["elements"][cancel_elem_name] = elem_dict
    return config


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

    Subslot j holds source_element's pulse for CLIFFORD_3SUBSLOTS[g][j] (zeros when idle). A
    cancel element passes the aggressor's xy element as source (played * amp(cancel_amp)).
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


def patch_config(config: dict, probe_qubit, drive_qubit, *, bidirectional: bool) -> dict:
    """Cancel element(s) and composite Clifford ops for build_program.

    Cancel elements are created in victim-name order (the phase reference is per FEM core, and
    crosstalk_cancellation_tomography uses the same order), so each lands on the core it was
    calibrated on.
    """
    pairs = [(probe_qubit, drive_qubit)] + ([(drive_qubit, probe_qubit)] if bidirectional else [])
    for victim, aggressor in sorted(pairs, key=lambda p: p[0].xy.name):
        config = _patch_cancel_element(config, victim, aggressor)
    config = _patch_composite_cliffords(config, probe_qubit.xy.name)
    config = _patch_composite_cliffords(config, drive_qubit.xy.name)
    config = _patch_composite_cliffords(config, f"{probe_qubit.xy.name}_cancel", drive_qubit.xy.name)
    if bidirectional:
        config = _patch_composite_cliffords(config, f"{drive_qubit.xy.name}_cancel", probe_qubit.xy.name)
    return config


# --------------------------------------------------------------------------- experiment


@register
class QMCrosstalkCompensatedSQRB(CrosstalkCompensatedSQRB):
    """Crosstalk Compensated SQRB on QM OPX1000."""

    def _select_pair(self):
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        probe_name, drive_name = self.params.probe_qubit, self.params.drive_qubit
        qubits = select_qubits(machine, [probe_name, drive_name], multiplexed=True)
        q_dict = {q.name: q for q in qubits}
        return q_dict.get(probe_name, qubits[0]), q_dict.get(drive_name, qubits[1])

    def _patch_config(self, config: dict, probe_qubit, drive_qubit) -> dict:
        return patch_config(config, probe_qubit, drive_qubit, bidirectional=bool(self.params.bidirectional))

    def patch_preview_config(self, config: dict) -> dict:
        """Patch the cancellation element(s) and composite ops into the generated config for preview."""
        config = copy.deepcopy(config)
        probe_qubit, drive_qubit = self._select_pair()
        return self._patch_config(config, probe_qubit, drive_qubit)

    def _program_kwargs(self) -> dict[str, Any]:
        from ._reset import check_reset_method

        return dict(
            cancel_amp=float(self.params.cancel_amp),
            init_phase=float(self.params.init_phase),
            bidirectional=bool(self.params.bidirectional),
            drive_cancel_amp=float(self.params.drive_cancel_amp),
            drive_init_phase=float(self.params.drive_init_phase),
            reset_type=check_reset_method(self),
            use_state_discrimination=bool(self.params.use_state_discrimination),
            seed=self.params.seed,
            simulate=getattr(self.backend, "simulate", False),
        )

    def probe(self) -> Any:
        machine = self.backend.machine
        probe_qubit, drive_qubit = self._select_pair()
        config = self._patch_config(machine.generate_config(), probe_qubit, drive_qubit)

        conditions = self.params.get_conditions()
        depths = [int(x) for x in self.params.get_depths()]
        chunks = sequence_chunks(int(self.params.num_random_sequences), max(depths), int(self.params.sequences_per_program))
        # the runner gets the first program; _acquire_benchmark builds and runs all of them
        prog, sweep_axes = build_program(
            machine, probe_qubit, drive_qubit, depths=depths, conditions=[conditions[0]],
            sequence_indices=chunks[0], num_shots=int(self.params.num_averages), **self._program_kwargs(),
        )
        return (
            prog,
            sweep_axes,
            lambda m, p, s, **kw: self._acquire_benchmark(
                m, config=config, probe_qubit=probe_qubit, drive_qubit=drive_qubit, **kw
            ),
        )

    def preview_program(self):
        """One representative program (compensated if benchmarked, one sequence, depth 4)."""
        machine = self.backend.machine
        probe_qubit, drive_qubit = self._select_pair()
        conditions = self.params.get_conditions()
        cond = "compensated" if "compensated" in conditions else conditions[0]
        prog, _ = build_program(
            machine, probe_qubit, drive_qubit, depths=[4], conditions=[cond], sequence_indices=[0],
            num_shots=1, **self._program_kwargs(),
        )
        return prog

    def _acquire_benchmark(
        self,
        machine,
        *,
        num_shots: int,
        timeout: float = 120.0,
        log: bool = False,
        config: Optional[dict] = None,
        probe_qubit=None,
        drive_qubit=None,
    ) -> xr.Dataset:
        """Run every condition, in programs of at most sequences_per_program sequences x every depth."""
        conditions = self.params.get_conditions()
        depths = [int(x) for x in self.params.get_depths()]
        total = int(self.params.num_random_sequences)
        chunks = sequence_chunks(total, max(depths), int(self.params.sequences_per_program))
        cond_datasets = []
        for c_idx, cond in enumerate(conditions, 1):
            chunk_sets = []
            for chunk in chunks:
                prog, axes = build_program(
                    machine, probe_qubit, drive_qubit, depths=depths, conditions=[cond],
                    num_sequences=total, sequence_indices=chunk, num_shots=num_shots, **self._program_kwargs(),
                )
                chunk_sets.append(_acquire_chunk_with_retry(
                    machine, prog, axes, num_shots=num_shots, timeout=timeout, log=log, config=config
                ))
            cond_datasets.append(xr.concat(chunk_sets, dim="sequence_idx"))
            print(
                f"# [Benchmark {c_idx}/{len(conditions)}] condition '{cond}': {total} sequences x "
                f"{len(depths)} depths in {len(chunks)} program(s) completed."
            )
        return xr.concat(cond_datasets, dim="condition")
