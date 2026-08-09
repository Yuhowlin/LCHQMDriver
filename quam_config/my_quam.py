"""The QUAM root class every calibration node imports.

``Quam`` is a DEFAULT, not a per-chip switch. It used to be hand-edited between
``FixedFrequencyQuam`` and ``FluxTunableQuam`` depending on which chip was mounted,
which is fragile (an uncommitted edit to a governed file) and cannot express a chip
that is partly fixed and partly tunable.

It no longer needs editing, because a device's root class is a property of the
DEVICE. ``quam/core/quam_instantiation.py`` honours the ``__class__`` stored in
``state.json`` over the class ``.load()`` was called on, so whatever a device stores
is what it loads as — this class only applies to a state file carrying no root
``__class__`` at all.

Two rules follow, and both are load-bearing:

1. A device's root ``__class__`` must name a CONCRETE, stable class — an official
   ``quam_builder`` root, or a lab one under ``customized/quam_builder/``. A device
   must NEVER store ``quam_config.my_quam.Quam``: that path resolves to whatever
   this module currently says, so the device's identity would depend on an
   uncommitted local edit. Repin with ``scripts/migrate_root_class.py``.
2. This default is the MIXED root, not a narrow one. A narrow default would reject
   a mixed tree, which is exactly the case that has no other home.

The per-QUBIT class is a separate, per-device decision, also stored in
``state.json`` (each qubit serialises its own ``__class__``) — see
``scripts/migrate_thermalizing_transmon.py``. It is NOT set here: a ``qubit_type``
ClassVar is build-time and type-hint only, and on a mixed chip there is no single
right answer to bind it to.
"""

from scqo_qm.quam_builder.architecture.superconducting.qpu.mixed_quam import (
    MixedTransmonQuam,
)


class Quam(MixedTransmonQuam):
    """Fallback root for a state file with no stored ``__class__``.

    Handles fixed-frequency, flux-tunable and mixed trees alike, so it is safe as a
    default for any chip.
    """

    pass
