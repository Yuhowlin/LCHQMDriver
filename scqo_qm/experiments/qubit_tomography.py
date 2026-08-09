"""Qubit Tomography acquisition probe: vendor code only (qm/quam).

QM qubit tomography for scqo - supplies only ``probe()``.

Parameters, fit, and reporting are inherited from ``scqo.experiments.QubitTomography``.
"""

from __future__ import annotations

from typing import Callable, Optional, Dict, List, Any
import numpy as np
import xarray as xr
from qm.qua import *

from scqo_qm.experiments._lib import acquire as _acquire


def play_init_state(qubit, state_str: str):
    """Play state preparation pulses."""
    st = str(state_str).strip().lower()
    if st in ("0", "g"):
        pass
    elif st in ("1", "e"):
        play("x180", qubit.xy.name)
    elif st in ("+", "+x"):
        play("y90", qubit.xy.name)
    elif st in ("-", "-x"):
        play("-y90", qubit.xy.name)
    elif st in ("+i", "+y"):
        play("-x90", qubit.xy.name)
    elif st in ("-i", "-y"):
        play("x90", qubit.xy.name)


def get_op_cycles(qubit, op_name: str = "x180", default_cycles: int = 10) -> int:
    """Get QUAM operation duration in QUA clock cycles (1 cycle = 4 ns)."""
    try:
        if hasattr(qubit, "xy") and hasattr(qubit.xy, "operations") and op_name in qubit.xy.operations:
            length_ns = qubit.xy.operations[op_name].length
            return max(1, int(length_ns) // 4)
    except Exception:
        pass
    return default_cycles


def play_target_gate(qubit, gate_str: str):
    """Play target gate pulse once."""
    gt = str(gate_str).strip().upper()
    if gt in ("I", "ID"):
        cycles = get_op_cycles(qubit, "x180")
        wait(cycles, qubit.xy.name)
    elif gt in ("X", "X180"):
        play("x180", qubit.xy.name)
    elif gt in ("X90", "X/2"):
        play("x90", qubit.xy.name)
    elif gt in ("Y", "Y180"):
        play("y180", qubit.xy.name)
    elif gt in ("Y90", "Y/2"):
        play("y90", qubit.xy.name)


def play_basis_rotation(qubit, basis_str: str):
    """Play basis measurement rotation pulse."""
    if basis_str == "z":
        pass
    elif basis_str == "x":
        play("-y90", qubit.xy.name)
    elif basis_str == "y":
        play("x90", qubit.xy.name)


def build_program(
    machine,
    qubits,
    *,
    qubit_configs: Dict[str, Dict[str, str]],
    gate_counts: List[int],
    num_shots: int,
    num_training_shots: int = 2000,
    symmetrized_readout: bool = True,
    reset_type: str = "thermal",
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the Qubit Tomography QUA program.

    A qubit whose config sets ``noise_mode: True`` is a spectator noise
    source: it still plays its target gates in phase 2 (that is the point —
    driving it injects crosstalk while a neighbour runs tomography) but skips
    init, basis rotation and measurement; its streams carry dummy zeros so
    the dataset shape stays uniform and the estimator marks it success=0.
    """
    num_qubits = len(qubits)
    qubit_names = qubits.get_names()
    bases = ["z", "x", "y"]
    sym_names = ["reg", "inv"] if symmetrized_readout else ["reg"]
    sym_indices = list(range(len(sym_names)))

    sweep_axes = {
        "qubit": xr.DataArray(qubit_names),
        "basis": xr.DataArray(bases),
        "sym": xr.DataArray(sym_names),
        "gate_count": xr.DataArray(gate_counts),
        "shot_idx": xr.DataArray(np.arange(num_shots)),
        "prepared_state": xr.DataArray([0, 1]),
        "train_shot_idx": xr.DataArray(np.arange(num_training_shots)),
    }

    with program() as prog:
        I, I_st, Q, Q_st, n, n_st = machine.declare_qua_variables()
        I_tr, I_tr_st, Q_tr, Q_tr_st, n_tr, n_tr_st = machine.declare_qua_variables()

        b_idx = declare(int)
        s_idx = declare(int)
        gc_idx = declare(int)
        rep = declare(int)
        ps_idx = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            # 1. Training Shots
            with for_(n_tr, 0, n_tr < num_training_shots, n_tr + 1):
                save(n_tr, n_tr_st)
                with for_each_(ps_idx, [0, 1]):
                    for i_q, qubit in multiplexed_qubits.items():
                        qubit.reset(reset_type, simulate, log_callable=log)
                    align()

                    for i_q, qubit in multiplexed_qubits.items():
                        q_name = qubit_names[i_q]
                        q_cfg = qubit_configs.get(q_name, {})
                        noise_mode = bool(q_cfg.get("noise_mode", False))
                        if not noise_mode:
                            with if_(ps_idx == 1):
                                play("x180", qubit.xy.name)
                    align()

                    for i_q, qubit in multiplexed_qubits.items():
                        q_name = qubit_names[i_q]
                        q_cfg = qubit_configs.get(q_name, {})
                        noise_mode = bool(q_cfg.get("noise_mode", False))

                        if not noise_mode:
                            qubit.resonator.measure("readout", qua_vars=(I_tr[i_q], Q_tr[i_q]))
                            save(I_tr[i_q], I_tr_st[i_q])
                            save(Q_tr[i_q], Q_tr_st[i_q])
                        else:
                            assign(I_tr[i_q], 0.0)
                            assign(Q_tr[i_q], 0.0)
                            save(I_tr[i_q], I_tr_st[i_q])
                            save(Q_tr[i_q], Q_tr_st[i_q])
                    align()

            # 2. Tomography Shots
            with for_(n, 0, n < num_shots, n + 1):
                save(n, n_st)
                with for_each_(b_idx, [0, 1, 2]):
                    with for_each_(s_idx, sym_indices):
                        with for_each_(gc_idx, gate_counts):
                            # Reset
                            for i_q, qubit in multiplexed_qubits.items():
                                qubit.reset(reset_type, simulate, log_callable=log)
                            align()

                            # Phase 1: Init State (all qubits simultaneously, noise_mode qubits skipped)
                            for i_q, qubit in multiplexed_qubits.items():
                                q_name = qubit_names[i_q]
                                q_cfg = qubit_configs.get(q_name, {})
                                noise_mode = bool(q_cfg.get("noise_mode", False))
                                if not noise_mode:
                                    init_st = q_cfg.get("init_state", "0")
                                    play_init_state(qubit, init_st)
                            align()

                            # Phase 2: Target Gate (all qubits per rep — including
                            # noise_mode qubits, which are driven as noise sources)
                            with for_(rep, 0, rep < gc_idx, rep + 1):
                                for i_q, qubit in multiplexed_qubits.items():
                                    q_name = qubit_names[i_q]
                                    q_cfg = qubit_configs.get(q_name, {})
                                    tgt_gt = q_cfg.get("target_gate", "X180")
                                    play_target_gate(qubit, tgt_gt)
                            align()

                            # Phase 3: Basis Rotation (all qubits simultaneously)
                            for i_q, qubit in multiplexed_qubits.items():
                                q_name = qubit_names[i_q]
                                q_cfg = qubit_configs.get(q_name, {})
                                noise_mode = bool(q_cfg.get("noise_mode", False))
                                if not noise_mode:
                                    with if_(b_idx == 0):
                                        play_basis_rotation(qubit, "z")
                                    with elif_(b_idx == 1):
                                        play_basis_rotation(qubit, "x")
                                    with else_():
                                        play_basis_rotation(qubit, "y")
                                    with if_(s_idx == 1):
                                        play("x180", qubit.xy.name)
                            align()

                            # Measurement (noise_mode qubits save dummy zeros)
                            for i_q, qubit in multiplexed_qubits.items():
                                q_name = qubit_names[i_q]
                                q_cfg = qubit_configs.get(q_name, {})
                                noise_mode = bool(q_cfg.get("noise_mode", False))

                                if not noise_mode:
                                    qubit.resonator.measure("readout", qua_vars=(I[i_q], Q[i_q]))
                                    save(I[i_q], I_st[i_q])
                                    save(Q[i_q], Q_st[i_q])
                                else:
                                    assign(I[i_q], 0.0)
                                    assign(Q[i_q], 0.0)
                                    save(I[i_q], I_st[i_q])
                                    save(Q[i_q], Q_st[i_q])
                            align()

        with stream_processing():
            n_st.save("n")
            n_tr_st.save("n_tr")
            for i_q in range(num_qubits):
                I_tr_st[i_q].buffer(2).buffer(num_training_shots).save(f"I_train{i_q + 1}")
                Q_tr_st[i_q].buffer(2).buffer(num_training_shots).save(f"Q_train{i_q + 1}")

                I_st[i_q].buffer(len(gate_counts)).buffer(len(sym_indices)).buffer(3).buffer(num_shots).save(f"I_tomo{i_q + 1}")
                Q_st[i_q].buffer(len(gate_counts)).buffer(len(sym_indices)).buffer(3).buffer(num_shots).save(f"Q_tomo{i_q + 1}")

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
    from scqo_qm.experiments._lib import qm_job, wait_all_streams

    qubit_names = list(sweep_axes["qubit"].values)
    num_qubits = len(qubit_names)
    bases = list(sweep_axes["basis"].values)
    sym_indices = list(sweep_axes["sym"].values)
    gate_counts = list(sweep_axes["gate_count"].values)
    shot_idx = list(sweep_axes["shot_idx"].values)
    prepared_states = list(sweep_axes["prepared_state"].values)
    train_shot_idx = list(sweep_axes["train_shot_idx"].values)

    with qm_job(machine, prog, timeout=timeout, config=config) as job:
        results = job.result_handles
        wait_all_streams(machine, results)

        I_train_list = []
        Q_train_list = []
        I_tomo_list = []
        Q_tomo_list = []

        for i_q in range(num_qubits):
            h_i_tr = results.get(f"I_train{i_q + 1}")
            h_q_tr = results.get(f"Q_train{i_q + 1}")
            h_i_to = results.get(f"I_tomo{i_q + 1}")
            h_q_to = results.get(f"Q_tomo{i_q + 1}")

            missing = []
            if h_i_tr is None: missing.append(f"I_train{i_q + 1}")
            if h_q_tr is None: missing.append(f"Q_train{i_q + 1}")
            if h_i_to is None: missing.append(f"I_tomo{i_q + 1}")
            if h_q_to is None: missing.append(f"Q_tomo{i_q + 1}")

            if missing:
                try:
                    avail = list(results.iter_all())
                except Exception:
                    avail = "unknown"
                raise RuntimeError(f"Tomography result handles missing {missing}. Available handles: {avail}")

            i_tr = h_i_tr.fetch_all()
            q_tr = h_q_tr.fetch_all()
            i_to = h_i_to.fetch_all()
            q_to = h_q_to.fetch_all()

            I_train_list.append(np.transpose(i_tr, (1, 0)))
            Q_train_list.append(np.transpose(q_tr, (1, 0)))

            I_tomo_list.append(np.transpose(i_to, (1, 2, 3, 0)))
            Q_tomo_list.append(np.transpose(q_to, (1, 2, 3, 0)))

        I_train_arr = np.stack(I_train_list, axis=0)
        Q_train_arr = np.stack(Q_train_list, axis=0)
        I_tomo_arr = np.stack(I_tomo_list, axis=0)
        Q_tomo_arr = np.stack(Q_tomo_list, axis=0)

        if log:
            rep = getattr(job, "execution_report", None)
            if callable(rep):
                log(rep())
            elif rep is not None:
                log(rep)

    ds = xr.Dataset(
        data_vars={
            "I_tomo": (("qubit", "basis", "sym", "gate_count", "shot_idx"), I_tomo_arr),
            "Q_tomo": (("qubit", "basis", "sym", "gate_count", "shot_idx"), Q_tomo_arr),
            "I_train": (("qubit", "prepared_state", "train_shot_idx"), I_train_arr),
            "Q_train": (("qubit", "prepared_state", "train_shot_idx"), Q_train_arr),
        },
        coords={
            "qubit": qubit_names,
            "basis": bases,
            "sym": sym_indices,
            "gate_count": gate_counts,
            "shot_idx": shot_idx,
            "prepared_state": prepared_states,
            "train_shot_idx": train_shot_idx,
        },
    )
    return ds


from typing import Any

from scqo import register
from scqo.experiments import QubitTomography


@register
class QMQubitTomography(QubitTomography):
    """Build a multiplexed qubit tomography QUA program on the QM OPX."""

    def probe(self) -> Any:
        from ._reset import check_reset_method
        from scqo_qm.experiments._lib import select_qubits

        machine = self.backend.machine
        qubits = select_qubits(machine, self.params.targets, multiplexed=True)

        prog, sweep_axes = build_program(
            machine,
            qubits,
            qubit_configs=self.params.qubit_configs,
            gate_counts=list(self.params.gate_counts),
            num_shots=int(self.params.num_averages),
            num_training_shots=int(self.params.num_training_shots),
            symmetrized_readout=self.params.symmetrized_readout,
            reset_type=check_reset_method(self),
        )
        return prog, sweep_axes, acquire
