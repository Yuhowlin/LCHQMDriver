"""Derive a schema-3 roster TOML from a live QUAM tree.

The roster describes the SAMPLE, so a roster for a vendor state that arrives
without one (the test fixtures, the ``scripts/check_real_config.py``
self-test) has to be generated from the tree itself. The REAL roster lives in
``<data_root>/<device>/components.toml`` and is what ``scqo run`` uses — this
generator exists for surfaces that exercise a bare ``state.json`` +
``wiring.json`` folder end-to-end.
"""

from __future__ import annotations


def roster_toml_for(machine) -> str:
    """A schema-3 roster describing whatever QUAM tree is passed in.

    One mode per QUAM qubit (``flux_transmon`` when it carries a z subtree,
    else a fixed ``transmon`` — a flux rider on a fixed transmon is a load
    error, which is exactly the capability-by-construction rule), one
    multiplexed readout feedline, a drive wire each, a flux wire for every
    z-capable mode INCLUDING the couplers, and one composite per QUAM
    qubit_pair named ``<low>_<high>`` — so the backend has to make the
    membership join that QM's coupler-named pairs require.

    The coupler MODE is named ``<low>_<high>_c``, matching the real device
    files (``5Q4C/components.toml`` declares ``[modes.q1_q2_c]`` beside
    ``[composites.q1_q2]``) rather than reusing the QUAM pair key. The key is
    not usable: QM names its pairs after the coupler, and on the live 5Q4C
    tree that name is ``q1_q2`` — identical to the composite this function
    derives, so the roster refused the whole fixture with "one name, one
    entity". Deriving both names from the members keeps them distinct on any
    tree, and keeps the mode name join going through the roster exactly as
    the real setup does.
    """
    modes, composites, lines, flux_riders = [], [], [], []
    lines.append("[lines.fl]\nreadout = ["
                 + ", ".join(f'"{n}"' for n in machine.qubits) + "]")
    for name, q in machine.qubits.items():
        has_z = getattr(q, "z", None) is not None
        kind = "flux_transmon" if has_z else "transmon"
        modes.append(f'[modes.{name}]\nkind = "{kind}"')
        lines.append(f'[lines.xy_{name}]\ndrive = ["{name}"]')
        if has_z:
            flux_riders.append(name)
    for key, qp in (getattr(machine, "qubit_pairs", {}) or {}).items():
        low, high = qp.qubit_control.name, qp.qubit_target.name
        block = [f"[composites.{low}_{high}]", 'kind = "qubit_pair"',
                 f'high = "{high}"', f'low = "{low}"']
        if getattr(qp, "coupler", None) is not None:
            # derived from the MEMBERS, not from `key`: QM names its pairs after
            # the coupler, and on the live tree that name collides with the
            # composite above (see the docstring)
            coupler = f"{low}_{high}_c"
            modes.append(f'[modes.{coupler}]\nkind = "flux_transmon"')
            flux_riders.append(coupler)
            block.append(f'coupler = "{coupler}"')
        composites.append("\n".join(block))
    lines += [f'[lines.z_{t}]\nflux = ["{t}"]' for t in flux_riders]
    return "\n\n".join(["schema = 3", *modes, *composites, *lines]) + "\n"
