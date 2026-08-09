"""Importing this package registers every QM experiment into the scqo catalog.

Add a line here for each new experiment module so its ``@register`` runs
(manual on purpose; ``tests/test_experiment_registration.py`` enforces
completeness in both directions).
"""

from . import pair_swap_chevron  # noqa: F401  (import side effect: @register)
from . import pair_swap_flux_map  # noqa: F401  (import side effect: @register)
from . import pair_zz_coupler  # noqa: F401  (import side effect: @register)
from . import qc_n_swap_amp  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey_cryoscope  # noqa: F401  (import side effect: @register)
from . import qubit_deterministic_benchmarking  # noqa: F401  (import side effect: @register)
from . import qubit_drag_alternating  # noqa: F401  (import side effect: @register)

from . import qubit_drag_equator  # noqa: F401  (import side effect: @register)
from . import qubit_echo  # noqa: F401  (import side effect: @register)
from . import qubit_echo_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_continuous  # noqa: F401  (import side effect: @register)
from . import qubit_parity_switch_discrete  # noqa: F401  (import side effect: @register)
from . import qubit_pi_pulse_error  # noqa: F401  (import side effect: @register)
from . import qubit_power_rabi  # noqa: F401  (import side effect: @register)
from . import qubit_ramsey  # noqa: F401  (import side effect: @register)
from . import qubit_relaxation  # noqa: F401  (import side effect: @register)
from . import qubit_relaxation_flux_pulse  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_cryoscope  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_overlap  # noqa: F401  (import side effect: @register)
from . import qubit_sqrb  # noqa: F401  (import side effect: @register)
from . import qubit_t1_ade  # noqa: F401  (import side effect: @register)
from . import qubit_t1_bayesian  # noqa: F401  (import side effect: @register)
from . import qubit_thermal_population  # noqa: F401  (import side effect: @register)
from . import qubit_tomography  # noqa: F401  (import side effect: @register)
from . import qubit_xyz_delay  # noqa: F401  (import side effect: @register)
from . import qubit_spectroscopy_flux_pulse  # noqa: F401  (import side effect: @register)
from . import readout_frequency  # noqa: F401  (import side effect: @register)
from . import readout_power  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_flux  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_chain  # noqa: F401  (import side effect: @register)
from . import resonator_spectroscopy_power_amp  # noqa: F401  (import side effect: @register)
from . import single_shot_readout  # noqa: F401  (import side effect: @register)
from . import single_shot_readout_gef  # noqa: F401  (import side effect: @register)
