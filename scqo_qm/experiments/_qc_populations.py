"""Population helpers shared by the swap-based visualization nodes
(`LCH_qc_swap_paramreset`, `LCH_qc_unidirectional_coupling`, `LCH_qc_N_swap`,
`LCH_qc_N_swap_amp`).

Every such node reads out a set of qubits at the end of each shot and sweeps a `round`
axis (plus, for `LCH_qc_N_swap_amp`, a `qubit_amplitude` axis). With state discrimination
the probe saves the **per-shot** discriminated states (`ds_raw["state"]`, dims
`(qubit, shot, round[, qubit_amplitude])`); these helpers turn that into either:

- the **joint** multi-qubit populations P(000), P(001), ... P(111) -- the correlated
  outcome of all measured qubits in the same shot (requires per-shot data); or
- the **per-qubit marginal** populations P(excited) (also works on historical data
  saved under the old `(qubit, round)` schema, which has no `shot` axis).

`joint_state_populations` keeps any non-`(qubit, shot)` dims, so it serves both the 1D
population-vs-round view (`plot_populations`, one line per state) and the 2D
population-vs-(round x amplitude) view (`plot_population_maps`, one color map per state).

The joint computation generalizes the two-qubit precedent in
`scqo_qm.probes.pair_qq_chevron` (P00/P01/P10/P11) to N qubits, but in Python.

Bitstring convention: the **first measured qubit is the leftmost / most-significant
digit** (the `qubit` coordinate order is the order the probe reads them out).
"""

import itertools
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def _qubit_names(state: xr.DataArray) -> List[str]:
    return [str(q) for q in state.qubit.values]


def joint_state_populations(state: xr.DataArray) -> xr.DataArray:
    """Joint multi-qubit populations from per-shot discriminated states.

    `state` must have dims including `qubit` and `shot`; every remaining dim (`round`, or
    `round` and `qubit_amplitude` for the 2D swap-amplitude sweep) is preserved. Returns a
    DataArray with a leading `joint_state` dim -- the bitstring label per measured qubit
    (first measured qubit = leftmost / most-significant digit) -- followed by the remaining
    dims unchanged. The populations sum to 1 over `joint_state` at every remaining-dim point.
    """
    names = _qubit_names(state)
    n = len(names)
    # Binarize to {0, 1} (>=1 -> excited), keeping every dim (qubit, shot + remaining).
    binary = (state >= 1).astype(np.int64)
    # Per-shot integer code with the first measured qubit as the most-significant bit; the
    # per-`qubit` weights align on the `qubit` coord so the sum preserves readout order.
    weights = xr.DataArray(
        (2 ** np.arange(n)[::-1]).astype(np.int64), coords={"qubit": state["qubit"]}, dims="qubit"
    )
    codes = (binary * weights).sum("qubit")  # dims: shot + remaining
    labels = ["".join(bits) for bits in itertools.product("01", repeat=n)]
    # Fraction of shots landing in each bitstring, one entry per label -> leading joint_state.
    return xr.concat(
        [(codes == v).mean("shot") for v in range(2 ** n)],
        dim="joint_state",
    ).assign_coords(joint_state=labels)


def marginal_populations(state: xr.DataArray) -> xr.DataArray:
    """Per-qubit excited-state population vs round, dims `(qubit, round)`.

    Works on the per-shot schema (averages over `shot`) and is a no-op passthrough on the
    old shot-averaged schema (no `shot` dim), where `state` already is the population.
    """
    if "shot" in state.dims:
        return (state >= 1).mean("shot")
    return state


def plot_populations(
    ds_raw: xr.Dataset,
    *,
    multiplexed: bool,
    use_state_discrimination: bool,
    title: str,
    xlabel: str,
) -> Dict[str, plt.Figure]:
    """Build a single combined population-vs-round figure; return ``{key: Figure}``.

    - state discrimination + ``multiplexed`` + per-shot data -> joint populations
      (one line per bitstring ``000``..); key ``"joint_populations"``.
    - state discrimination otherwise -> per-qubit marginal P(excited), one line per
      measured qubit; key ``"populations"``.
    - no state discrimination -> raw ``I`` quadrature, one line per qubit; key ``"raw_I"``.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if use_state_discrimination and "state" in ds_raw:
        state = ds_raw["state"]
        if multiplexed and "shot" in state.dims:
            pops = joint_state_populations(state)
            for label in [str(v) for v in pops.joint_state.values]:
                pops.sel(joint_state=label).plot(x="round", ax=ax, marker="o", label=f"|{label}⟩")
            ax.set_ylabel("Population")
            ax.legend(title="state", ncol=2, fontsize="small")
            key = "joint_populations"
        else:
            marg = marginal_populations(state)
            for q in _qubit_names(state):
                marg.sel(qubit=q).plot(x="round", ax=ax, marker="o", label=q)
            ax.set_ylabel("P(excited)")
            ax.legend(title="qubit")
            key = "populations"
    else:
        for q in [str(v) for v in ds_raw.qubit.values]:
            ds_raw["I"].sel(qubit=q).plot(x="round", ax=ax, marker="o", label=q)
        ax.set_ylabel("I [V]")
        ax.legend(title="qubit")
        key = "raw_I"

    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    return {key: fig}


def plot_population_maps(
    ds_raw: xr.Dataset,
    *,
    multiplexed: bool,
    use_state_discrimination: bool,
    title: str,
    xlabel: str,
    ylabel: str,
    y_dim: str,
) -> Dict[str, plt.Figure]:
    """Build a grid of 2D color maps -- one panel per state (or per qubit); return ``{key: Figure}``.

    The 2D analog of `plot_populations`: the x axis is always `round`, `y_dim` is the second
    swept axis (e.g. `qubit_amplitude`), and the color is the population (or raw ``I``).

    - state discrimination + ``multiplexed`` + per-shot data -> one map per joint bitstring
      state (``000``..); colorbar "population"; key ``"joint_state_maps"``.
    - state discrimination otherwise -> one map per measured qubit's marginal P(excited);
      key ``"population_maps"``.
    - no state discrimination -> one map per qubit's raw ``I`` quadrature; key ``"raw_I_maps"``.
    """
    if use_state_discrimination and "state" in ds_raw:
        state = ds_raw["state"]
        if multiplexed and "shot" in state.dims:
            pops = joint_state_populations(state)  # (joint_state, y_dim, round)
            panels = [(f"|{label}⟩", pops.sel(joint_state=label)) for label in [str(v) for v in pops.joint_state.values]]
            cbar_label, key = "population", "joint_state_maps"
        else:
            marg = marginal_populations(state)  # (qubit, y_dim, round)
            panels = [(q, marg.sel(qubit=q)) for q in _qubit_names(state)]
            cbar_label, key = "P(excited)", "population_maps"
        vmin, vmax = 0.0, 1.0
    else:
        panels = [(q, ds_raw["I"].sel(qubit=q)) for q in [str(v) for v in ds_raw.qubit.values]]
        cbar_label, key = "I [V]", "raw_I_maps"
        vmin, vmax = None, None

    npanels = len(panels)
    ncols = min(npanels, 4)
    nrows = int(np.ceil(npanels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.4 * ncols + 1.5, 3.0 * nrows), sharex=True, sharey=True, squeeze=False
    )
    flat_axes = axes.ravel()
    qm = None
    for idx, (name, da) in enumerate(panels):
        ax = flat_axes[idx]
        qm = da.plot(x="round", y=y_dim, ax=ax, add_colorbar=False, vmin=vmin, vmax=vmax)
        ax.set_title(name)
        # Label only the bottom of each column and the left column (axes are shared).
        ax.set_xlabel(xlabel if idx + ncols >= npanels else "")
        ax.set_ylabel(ylabel if idx % ncols == 0 else "")
    for idx in range(npanels, nrows * ncols):
        flat_axes[idx].axis("off")

    fig.colorbar(qm, ax=list(flat_axes), label=cbar_label, fraction=0.046, pad=0.04)
    fig.suptitle(title)
    return {key: fig}
