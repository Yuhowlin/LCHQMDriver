"""N-swap x qubit-flux-amplitude visualization: a grid of 2D color maps.

With state discrimination and `multiplexed` on, one 2D color map is drawn per joint
multi-qubit state (000, 001, ... 111): population vs the number of swaps N (x) and the
swept ctrl flux amplitude (y), on a shared 0-1 color scale. With `multiplexed` off, one map
is drawn per measured qubit's P(excited); without state discrimination, one map per qubit's
raw I. The population math + plot live in the shared `customized.node._qc_populations`
helper.
"""

from typing import Dict

import matplotlib.pyplot as plt
import xarray as xr

from scqo_qm.experiments._qc_populations import plot_population_maps


def plot_amp_state_maps(
    ds_raw: xr.Dataset,
    measure_qubits,
    *,
    use_state_discrimination: bool,
    multiplexed: bool = False,
) -> Dict[str, plt.Figure]:
    """Build the grid of 2D population maps (one per state); return ``{key: Figure}``.

    `measure_qubits` is accepted for interface symmetry; the qubit names are taken from the
    dataset. `multiplexed` selects the per-joint-state (True) vs per-qubit (False) view when
    state discrimination is on.
    """
    return plot_population_maps(
        ds_raw,
        multiplexed=multiplexed,
        use_state_discrimination=use_state_discrimination,
        title="N-swap vs ctrl flux amplitude",
        xlabel="Number of swaps N",
        ylabel="ctrl flux amplitude [V]",
        y_dim="qubit_amplitude",
    )
