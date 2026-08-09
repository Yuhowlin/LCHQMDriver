"""Unidirectional-coupling visualization: a single combined population-vs-round figure.

With state discrimination and `multiplexed` on, the joint multi-qubit populations
(000, 001, ... 111) are plotted on one figure; with `multiplexed` off, each measured
qubit's independent P(excited) is plotted on one figure. Without state discrimination the
raw I quadrature of each qubit is shown instead. The population math + plot live in the
shared `customized.node._qc_populations` helper.
"""

from typing import Dict

import matplotlib.pyplot as plt
import xarray as xr

from scqo_qm.experiments._qc_populations import plot_populations


def plot_rounds_1d(
    ds_raw: xr.Dataset,
    measure_qubits,
    *,
    use_state_discrimination: bool,
    multiplexed: bool = False,
) -> Dict[str, plt.Figure]:
    """Build the combined population-vs-round figure; return ``{key: Figure}``.

    `measure_qubits` is accepted for interface symmetry; the qubit names are taken from the
    dataset. `multiplexed` selects the joint (True) vs per-qubit (False) view when state
    discrimination is on.
    """
    return plot_populations(
        ds_raw,
        multiplexed=multiplexed,
        use_state_discrimination=use_state_discrimination,
        title="Unidirectional coupling",
        xlabel="Number of swap-chain + reset rounds",
    )
