"""Instrument probes: the acquisition half of each LEGACY dual-path experiment.

TRANSITIONAL PACKAGE: dies with the Stage C fusion of the v1 restructure —
every module here gets inlined into its scqo_qm/experiments/ class and deleted.

FROZEN LAYER (2026-08-09): new experiments do NOT add modules here — they are
scqo-only fused files in `scqo_qm/experiments/` (CLAUDE.md "Adding a new
experiment"). A probe here is the code shared between the frozen set's two
orchestrators: the qualibrate shells in `calibrations/LCH_*.py` and the scqo
`QMBackend` call exactly the same functions. Everything qualibrate-specific
(parameter schema, scqat analysis adapter, state-update policy) lives in
`customized/node/LCH_<name>/` instead. The shared helpers (`_lib`,
`_flux_limits`, `_amp_limits`) stay importable from fused files.

Import rules for everything under this package:
- MAY import: qm.qua, quam, qualang_tools, qualibration_libs.core / .data
  (framework-free utilities), numpy, xarray.
- MUST NOT import: qualibrate, scqo, or scqat - probes acquire, they never fit.
- Functions take explicit data in (machine, qubits, plain kwargs) and return
  plain data out (program, xr.Dataset) - never a `node`.
"""
