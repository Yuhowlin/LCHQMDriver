"""Joint two-qubit populations -> the joint form scqo's pair swap maps contract for.

The FPGA-averaged swap probes (``pair_qq_chevron``, ``pair_qcq_fixed_time``)
read BOTH members out 2-level and save the four joint populations P00/P01/P10/
P11 as ``state_gg/state_ge/state_eg/state_ee``, with the FIRST digit the
vendor's control qubit. scqo's readout schema stores the same distribution as
ONE variable — ``joint_population`` over a ``joint_state`` label dim whose digit
order is the roster ROLES (high, low) — so the mapping needs the
control/target -> high/low orientation the adapter resolved once in ``probe()``.

This is data SHAPING, deliberately not in ``_vendor.py`` (which is name
resolution). It lives in a mixin because the FPGA-averaged adapters need it
byte-identically and a swap map's reduction is not something either should own
alone. (``qc_n_swap_amp`` does NOT use it — its probe keeps per-shot states and
reduces through scqo's ``states_to_joint_population`` helper instead.)
"""

from __future__ import annotations

import xarray as xr
from scqo.experiments import joint_state_labels


class JointPopulationMixin:
    """``reduce_raw`` for the FPGA-averaged pair swap maps. Requires ``self._high_side``."""

    #: "control" | "target" — which vendor side the roster's `high` role is.
    #: Set once in probe(), via _vendor.role_side(self, "high", ...).
    _high_side: str

    def reduce_raw(self, raw: xr.Dataset) -> xr.Dataset:
        """Vendor-ordered joint populations -> the role-ordered joint form.

        The four FPGA vars ARE the joint distribution (that is what separates a
        genuine single-excitation swap from heating, imperfect reset, or a
        |11>-|02> leak); all this does is reorder the digits from (control,
        target) to the roster's (high, low) and stack them on the labeled
        ``joint_state`` dim the contract requires.
        """
        if "state_ee" not in raw.data_vars:
            raise ValueError(
                f"{type(self).name}: the probe returned no joint populations "
                f"(data_vars={list(raw.data_vars)}). The swap maps contract for "
                f"populations, so they need state discrimination — run "
                f"single_shot_readout and accept its readout_rotation_rad / "
                f"readout_threshold suggestions first.")
        high_is_control = self._high_side == "control"
        parts = {
            "00": raw["state_gg"],
            # label digit order is (high, low): with high==target the vendor's
            # (control, target) single-excitation vars swap places.
            "01": raw["state_ge"] if high_is_control else raw["state_eg"],
            "10": raw["state_eg"] if high_is_control else raw["state_ge"],
            "11": raw["state_ee"],
        }
        labels = joint_state_labels(2)
        joint = xr.concat([parts[label] for label in labels], dim="joint_state")
        joint = joint.assign_coords(joint_state=labels)
        return joint.to_dataset(name="joint_population")
