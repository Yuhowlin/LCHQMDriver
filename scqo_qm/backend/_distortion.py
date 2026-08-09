"""Neutral flux-distortion facts -> QM OPX ``exponential_filter`` config values.

Pure config-value helpers (no push, no quam import): turn the scqo flux-channel
facts ``distortion_amp[]`` (relative amplitudes ``A_i``) + ``distortion_tau_s[]``
(seconds) into the value a QM LF-FEM z port's ``opx_output.exponential_filter``
accepts. The live OPX (QOP >= 3.3) takes the SUM form directly; QOP 3.4.1 takes
the single-pole CASCADE (from scqat), with an overall ``scale`` applied to the
FIR/waveform. Amplitudes are already relative (``amp/a_dc``), so the SUM mapping
is purely ``tau: seconds -> ns``.

These are library helpers a notebook/CLI calls with the recorded facts.
``apply_exponential_filter`` additionally WRITES the value onto a loaded QUAM (still
no quam import — it duck-types ``qubits[t].z.opx_output`` — and never persists; the
caller saves). Wiring any of this into an auto-push knob is a separate decision: the
``distortion_*`` facts stay physical.json record-only today.
"""

from __future__ import annotations

from typing import Any, Sequence

#: OPX LF-FEM sample period (1 GS/s), seconds.
OPX_TS_S = 1e-9


def to_exponential_filter(
    amps: Sequence[float], taus_s: Sequence[float]
) -> list[list[float]]:
    """The QOP >= 3.3 SUM value for ``z.opx_output.exponential_filter``:
    ``[[A_i, tau_i_ns], ...]`` — amplitudes verbatim (already relative), tau s->ns.
    """
    if len(amps) != len(taus_s):
        raise ValueError(
            f"amps ({len(amps)}) and taus_s ({len(taus_s)}) must be equal length")
    return [[float(a), float(t) * 1e9] for a, t in zip(amps, taus_s)]


def to_exponential_filter_cascade(
    amps: Sequence[float], taus_s: Sequence[float], *, ts_s: float = OPX_TS_S
) -> dict[str, Any]:
    """The QOP 3.4.1 single-pole CASCADE value + scale:
    ``{"exponential_filter": [[A_c, tau_c_ns], ...], "scale": float}``.

    Apply ``scale`` to the FIR coefficients or the flux-waveform amplitude per your
    QOP. The facts are relative (a_dc factored out), so ``a_dc=1.0`` is passed to
    the decomposition (which also satisfies its ``a_dc > MIN_A_DC`` guard).
    """
    from scqat.tools.flux_predistortion import exp_sum_to_cascade

    casc = exp_sum_to_cascade(amps, taus_s, a_dc=1.0, ts_s=ts_s)
    return {
        "exponential_filter": [
            [a, t * 1e9] for a, t in zip(casc["amps_c"], casc["taus_c_s"])
        ],
        "scale": casc["scale"],
    }


def clear_exponential_filter(machine, target: str) -> dict[str, Any]:
    """Remove ALL taps from ``machine.qubits[target].z.opx_output.exponential_filter``
    — the fresh-line reset before a clean-slate cryoscope characterization (a
    full correction must be MEASURED on a cleared line). Returns
    ``{"removed": [the taps that were set]}``; does NOT persist (caller saves).
    Same target/z guards as :func:`apply_exponential_filter`.
    """
    try:
        qubit = machine.qubits[target]
    except (KeyError, TypeError):
        try:
            have = sorted(machine.qubits)
        except Exception:
            have = "?"
        raise ValueError(
            f"{target!r} is not a qubit in this machine (have {have})") from None
    z = getattr(qubit, "z", None)
    if z is None:
        raise ValueError(
            f"{target!r} has no flux (z) line — no exponential_filter to clear")
    port = z.opx_output
    removed = [list(pair) for pair in (port.exponential_filter or [])]
    port.exponential_filter = []
    return {"removed": removed}


def apply_exponential_filter(
    machine, target: str, amps: Sequence[float], taus_s: Sequence[float], *,
    replace: bool = True, form: str = "sum", ts_s: float = OPX_TS_S,
) -> dict[str, Any]:
    """Write the cryoscope distortion taps to a qubit's OPX z-output filter.

    Sets ``machine.qubits[target].z.opx_output.exponential_filter`` from the
    recorded flux facts and returns ``{"exponential_filter": <written value>,
    "scale": float}``. The taps are fed as MEASURED (relative ``A_i``, ``tau`` in
    seconds) — the OPX firmware builds the inverse, so do NOT pre-invert them; this
    mirrors ``calibrations/18_cryoscope.py``.

    Args:
        machine: the loaded QUAM (``Quam.load(<setup backend_config dir>)`` — the
            same state.json the scqo session loads; ``state_sync="pull"`` makes it
            authoritative).
        target: qubit name, e.g. ``"q1"`` (the flux channel is ``<target>.z``).
        amps: ``distortion_amp`` — relative amplitudes ``A_i = amp/a_dc``.
        taus_s: ``distortion_tau_s`` — time constants in SECONDS.
        replace: ``True`` OVERWRITES the port filter (scqo REPLACE fact semantics —
            for a fresh full correction measured on a cleared line); ``False``
            EXTENDS it (iterative refinement of a residual measured with the current
            filter active).
        form: ``"sum"`` (QOP >= 3.3, the direct sum value; ``scale`` is ``1.0``) or
            ``"cascade"`` (QOP 3.4.1 single-pole cascade — ALSO apply the returned
            ``scale`` to the flux-waveform amplitude yourself).
        ts_s: OPX sample period, only used by the cascade discretization.

    Does NOT persist — call ``machine.save()`` afterwards (batch several targets
    first if you like). Raises ``ValueError`` for an unknown ``target``, a target
    with no ``z`` line, an unknown ``form``, or ``form="cascade"`` with
    ``replace=False`` (a cascade is a whole-response decomposition, not a stack).
    """
    try:
        qubit = machine.qubits[target]
    except (KeyError, TypeError):
        try:
            have = sorted(machine.qubits)
        except Exception:
            have = "?"
        raise ValueError(
            f"{target!r} is not a qubit in this machine (have {have})") from None
    z = getattr(qubit, "z", None)
    if z is None:
        raise ValueError(
            f"{target!r} has no flux (z) line — no exponential_filter to set")

    if form == "sum":
        value: list = to_exponential_filter(amps, taus_s)
        scale = 1.0
    elif form == "cascade":
        if not replace:
            raise ValueError(
                "form='cascade' is a whole-response decomposition and cannot be "
                "EXTENDED onto existing taps — use replace=True")
        casc = to_exponential_filter_cascade(amps, taus_s, ts_s=ts_s)
        value, scale = casc["exponential_filter"], casc["scale"]
    else:
        raise ValueError(f"unknown form {form!r} (use 'sum' or 'cascade')")

    port = z.opx_output
    if replace or not port.exponential_filter:
        # wholesale (re)assignment builds a fresh list of plain [A, tau] pairs.
        port.exponential_filter = list(value)
    else:
        # EXTEND in place: reassigning old+new re-parents the existing QuamList
        # children and QUAM refuses that ("Cannot overwrite parent attribute").
        # Appending fresh pairs is how 18_cryoscope grows the filter.
        port.exponential_filter.extend(value)
    return {"exponential_filter": port.exponential_filter, "scale": scale}
