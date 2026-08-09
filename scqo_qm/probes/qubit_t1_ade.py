"""ADE T1-tracking probe: vendor code only (qm/quam) - no qualibrate, no scqo, no scqat.

Per block: ``n_avg`` interleaved shots at each of the three delays t0 / t0+dt /
t0+3*dt (reset -> x180 -> wait -> readout_state), then the ON-FPGA closed form
(arXiv:2602.11912)

    c = (P3-P0)/(P1-P0),  x = sqrt(c-3/4)-1/2,  gamma = -ln(x)/dt

plus the analytic shot-noise sigma via the chain rule — streamed once per
block, in 1/us. This is the repo's first real-time-arithmetic probe, so the
numeric-range rationale lives here:

QUA ``fixed`` is signed 4.28, range [-8, 8). Times are carried in units of
TIME_SCALE_US (= 32 us) so a dt up to ~250 us fits; sqrt/ln arguments and the
squared denominator are floored with relu-clamps built from that range
(LN_ARG_FLOOR / SQRT_ARG_FLOOR / DENOM_SQ_FLOOR), and every derivative is
clamped to +-SAFE_CEILING. A block outside the ADE validity domain therefore
streams a plausible-looking FLOORED number — the host re-derives the validity
mask from the per-shot streams (scqat flags those blocks), which is why the
raw shots are always streamed too.

With ``adaptive_dt`` the next block's dt is retuned from the running gamma
(dt = dt_factor * T1_est), clipped to [min_dt, max_dt] cycles.

This probe cannot use ``_lib.acquire``: its streams are heterogeneous —
per-block scalars, (block, shot) state arrays and a timestamp stream — which
the shared ``XarrayDataFetcher`` refuses (one uniform shape per dataset), so
it ships its own ``acquire()`` (the ``qubit_tomography`` pattern).
"""

import math
from typing import Callable, Dict, Optional

import numpy as np
import xarray as xr
from qm.qua import *

# ---- QUA fixed-point range plumbing (see module docstring) -------------------
QUA_FIXED_MAX = 8.0
SAFE_CEILING = QUA_FIXED_MAX - 2.0
TIME_SCALE_US = 4.0 * QUA_FIXED_MAX
LN_ARG_FLOOR = math.exp(-QUA_FIXED_MAX)
SQRT_ARG_FLOOR = (LN_ARG_FLOOR + 0.5) ** 2
GAMMA_ADAPTIVE_FLOOR = 1.0 / TIME_SCALE_US
DENOM_SQ_FLOOR = 1.0 / SAFE_CEILING

#: the three delays are t0 + DELAY_MULTS * dt (the closed form's 1:3 spacing);
#: the dataset's delay_idx coordinate carries these multipliers as its values.
DELAY_MULTS = (0, 1, 3)

#: per-delay stream base names, index-aligned with DELAY_MULTS.
_STATE_STREAMS = ("state_short", "state_mid", "state_long")


def build_program(
    machine,
    qubits,
    *,
    num_blocks: int,
    n_avg: int,
    t0_cycles: int,
    dt_cycles: int,
    adaptive_dt: bool,
    dt_factor: float,
    min_dt_cycles: int,
    max_dt_cycles: int,
    reset_type: str,
    reset_max_attempts: int = 15,
    simulate: bool = False,
    log: Optional[Callable] = None,
):
    """Build the ADE tracking QUA program. Returns (program, sweep_axes)."""
    num_qubits = len(qubits)

    sweep_axes = {
        "qubit": xr.DataArray(qubits.get_names()),
        "block_idx": xr.DataArray(np.arange(num_blocks)),
    }

    with program() as prog:
        _, _, _, _, n, n_st = machine.declare_qua_variables()

        state = [declare(int) for _ in range(num_qubits)]
        gamma_st = [declare_stream() for _ in range(num_qubits)]
        sigma_st = [declare_stream() for _ in range(num_qubits)]
        dt_st = [declare_stream() for _ in range(num_qubits)]
        shots_st = [
            [declare_stream() for _ in range(num_qubits)] for _ in _STATE_STREAMS
        ]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                machine.initialize_qpu(target=qubit)
            align()

            for i, qubit in multiplexed_qubits.items():
                acc = [declare(int) for _ in range(3)]
                shot = declare(int)

                P0 = declare(fixed)
                P1 = declare(fixed)
                P3 = declare(fixed)
                c = declare(fixed)
                sqrt_arg = declare(fixed)
                xval = declare(fixed)
                dt_scaled = declare(fixed)
                gamma_est = declare(fixed)
                denom = declare(fixed)
                denom_sq = declare(fixed)
                x_plus_half = declare(fixed)
                dgamma_dc = declare(fixed)
                dgamma_dP0 = declare(fixed)
                dgamma_dP1 = declare(fixed)
                dgamma_dP3 = declare(fixed)
                sigma_P0 = declare(fixed)
                sigma_P1 = declare(fixed)
                sigma_P3 = declare(fixed)
                term0 = declare(fixed)
                term1 = declare(fixed)
                term2 = declare(fixed)
                sigma_gamma = declare(fixed)
                gamma_safe = declare(fixed)
                T1_est_scaled = declare(fixed)

                delta_t_cycles = declare(int, value=int(dt_cycles))
                inv_n_avg = 1.0 / n_avg

                with for_(n, 0, n < num_blocks, n + 1):
                    save(n, n_st)
                    for a in acc:
                        assign(a, 0)

                    # interleave the three delays SHOT BY SHOT so P0/P1/P3
                    # sample the same lab-time window while T1 drifts
                    with for_(shot, 0, shot < n_avg, shot + 1):
                        for d, mult in enumerate(DELAY_MULTS):
                            qubit.reset(reset_type, simulate, log_callable=log,
                                        max_attempts=reset_max_attempts)
                            qubit.align()
                            qubit.xy.play("x180")
                            if mult == 0:
                                qubit.wait(t0_cycles)
                            else:
                                qubit.wait(t0_cycles + mult * delta_t_cycles)
                            qubit.align()
                            qubit.readout_state(state[i])
                            assign(acc[d], acc[d] + state[i])
                            save(state[i], shots_st[d][i])

                    assign(P0, Cast.mul_fixed_by_int(inv_n_avg, acc[0]))
                    assign(P1, Cast.mul_fixed_by_int(inv_n_avg, acc[1]))
                    assign(P3, Cast.mul_fixed_by_int(inv_n_avg, acc[2]))

                    # point estimate — every sqrt/ln argument floored into the
                    # fixed range (relu-clamp); dt carried in TIME_SCALE_US units
                    assign(c, Math.div(P3 - P0, P1 - P0))
                    assign(sqrt_arg, SQRT_ARG_FLOOR + Math.relu((c - 0.75) - SQRT_ARG_FLOOR))
                    assign(xval, Math.sqrt(sqrt_arg) - 0.5)
                    assign(xval, LN_ARG_FLOOR + Math.relu(xval - LN_ARG_FLOOR))
                    assign(dt_scaled, Cast.mul_fixed_by_int(4e-3 / TIME_SCALE_US, delta_t_cycles))
                    assign(gamma_est, (-Math.div(Math.ln(xval), dt_scaled)) * (1.0 / TIME_SCALE_US))
                    save(gamma_est, gamma_st[i])  # 1/us
                    save(delta_t_cycles, dt_st[i])

                    # analytic sigma: binomial shot noise through the chain
                    # rule; every derivative clamped to +-SAFE_CEILING
                    assign(denom, P1 - P0)
                    assign(denom_sq, denom * denom)
                    assign(denom_sq, DENOM_SQ_FLOOR + Math.relu(denom_sq - DENOM_SQ_FLOOR))
                    assign(x_plus_half, xval + 0.5)

                    assign(dgamma_dc, Math.div(-1.0, 2.0 * xval * x_plus_half * dt_scaled))
                    assign(dgamma_dc, dgamma_dc * (1.0 / TIME_SCALE_US))
                    assign(dgamma_dc, -SAFE_CEILING + Math.relu(dgamma_dc + SAFE_CEILING))
                    assign(dgamma_dc, SAFE_CEILING - Math.relu(SAFE_CEILING - dgamma_dc))

                    assign(dgamma_dP0, dgamma_dc * Math.div(P3 - P1, denom_sq))
                    assign(dgamma_dP1, dgamma_dc * Math.div(-(P3 - P0), denom_sq))
                    assign(dgamma_dP3, dgamma_dc * Math.div(denom, denom_sq))

                    for dg in (dgamma_dP0, dgamma_dP1, dgamma_dP3):
                        assign(dg, -SAFE_CEILING + Math.relu(dg + SAFE_CEILING))
                        assign(dg, SAFE_CEILING - Math.relu(SAFE_CEILING - dg))

                    assign(sigma_P0, Math.sqrt(P0 * (1.0 - P0) * inv_n_avg))
                    assign(sigma_P1, Math.sqrt(P1 * (1.0 - P1) * inv_n_avg))
                    assign(sigma_P3, Math.sqrt(P3 * (1.0 - P3) * inv_n_avg))

                    assign(term0, dgamma_dP0 * sigma_P0)
                    assign(term0, term0 * term0)
                    assign(term1, dgamma_dP1 * sigma_P1)
                    assign(term1, term1 * term1)
                    assign(term2, dgamma_dP3 * sigma_P3)
                    assign(term2, term2 * term2)
                    assign(sigma_gamma, Math.sqrt(term0 + term1 + term2))
                    save(sigma_gamma, sigma_st[i])  # 1/us

                    # hardware timestamp of the block (zero-amplitude marker)
                    qubit.xy.play("x180", amplitude_scale=0, duration=4,
                                  timestamp_stream=f"time_stamp{i + 1}")

                    if adaptive_dt:
                        # dt_next = dt_factor * T1_est, in cycles: the int
                        # constant absorbs us->cycles (250) and the descale
                        assign(gamma_safe, GAMMA_ADAPTIVE_FLOOR + Math.relu(gamma_est - GAMMA_ADAPTIVE_FLOOR))
                        assign(T1_est_scaled, Math.div(1.0, gamma_safe * TIME_SCALE_US))
                        _delta_t_int_const = round(dt_factor * 250 * TIME_SCALE_US)
                        assign(delta_t_cycles, Cast.mul_int_by_fixed(_delta_t_int_const, T1_est_scaled))
                        with if_(delta_t_cycles < min_dt_cycles):
                            assign(delta_t_cycles, min_dt_cycles)
                        with if_(delta_t_cycles > max_dt_cycles):
                            assign(delta_t_cycles, max_dt_cycles)

        align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                gamma_st[i].buffer(num_blocks).save(f"estimated_gamma{i + 1}")
                sigma_st[i].buffer(num_blocks).save(f"sigma_gamma{i + 1}")
                dt_st[i].buffer(num_blocks).save(f"dt_used{i + 1}")
                for d, name in enumerate(_STATE_STREAMS):
                    shots_st[d][i].buffer(num_blocks, n_avg).save(f"{name}{i + 1}")

    return prog, sweep_axes


def _fetch_values(results, name: str) -> np.ndarray:
    """Fetch one handle; unwrap the structured dtype timestamp streams carry."""
    handle = results.get(name)
    if handle is None:
        try:
            avail = list(results.iter_all())
        except Exception:
            avail = "unknown"
        raise RuntimeError(f"ADE result handle '{name}' missing. Available: {avail}")
    values = np.asarray(handle.fetch_all())
    if values.dtype.names:
        field = "value" if "value" in values.dtype.names else values.dtype.names[0]
        values = values[field]
    return np.squeeze(values)


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
    """Execute and hand-fetch the heterogeneous ADE streams into the canonical
    dataset (gamma/sigma converted 1/us -> 1/s, dt in ns, per-shot states over
    (block_idx, delay_idx, shot_idx), timestamps -> elapsed block_time_s)."""
    from qualang_tools.multi_user import qm_session

    qmm = machine.connect()
    config = config if config is not None else machine.generate_config()

    qubit_names = list(np.atleast_1d(sweep_axes["qubit"].values))
    num_qubits = len(qubit_names)
    n_blocks = sweep_axes["block_idx"].values.size

    with qm_session(qmm, config, timeout=timeout) as qm:
        job = qm.execute(prog)
        results = job.result_handles
        results.wait_for_all_values()

        gamma, sigma, dt_cycles, shots, stamps = [], [], [], [], []
        for i in range(num_qubits):
            gamma.append(_fetch_values(results, f"estimated_gamma{i + 1}"))
            sigma.append(_fetch_values(results, f"sigma_gamma{i + 1}"))
            dt_cycles.append(_fetch_values(results, f"dt_used{i + 1}"))
            shots.append(np.stack(
                [_fetch_values(results, f"{name}{i + 1}") for name in _STATE_STREAMS],
                axis=1,
            ))  # (block, delay, shot)
            stamps.append(_fetch_values(results, f"time_stamp{i + 1}")[:n_blocks])

        if log:
            rep = getattr(job, "execution_report", None)
            if callable(rep):
                log(rep())
            elif rep is not None:
                log(rep)

    stamps_arr = np.asarray(stamps, dtype=float)
    block_time_s = (stamps_arr - stamps_arr[:, :1]) * 4e-9

    return xr.Dataset(
        data_vars={
            "estimated_gamma": (("qubit", "block_idx"),
                                np.asarray(gamma, dtype=float) * 1e6),
            "sigma_gamma": (("qubit", "block_idx"),
                            np.asarray(sigma, dtype=float) * 1e6),
            "dt_ns": (("qubit", "block_idx"),
                      np.asarray(dt_cycles, dtype=float) * 4.0),
            "state": (("qubit", "block_idx", "delay_idx", "shot_idx"),
                      np.asarray(shots, dtype=np.int8)),
            "block_time_s": (("qubit", "block_idx"), block_time_s),
        },
        coords={
            "qubit": qubit_names,
            "block_idx": np.arange(n_blocks),
            "delay_idx": np.array(DELAY_MULTS),
            "shot_idx": np.arange(np.asarray(shots).shape[-1]),
        },
    )
