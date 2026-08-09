
from quam_builder.architecture.superconducting.qubit import FluxTunableTransmon
from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.components.flux_line import FluxLine


__all__ = ["ChargeTunableTransmon"]
@quam_dataclass
class ChargeTunableTransmon(FluxTunableTransmon):
    """
    Example QUAM component for a flux tunable transmon qubit.

    Args:
        charge_dispersion (charge bias): .

    """
    # q: FluxLine = None
    charge_dispersion: float = 0.0
    charge_offset: float = 0.0