# LCHQMDriver — project guide

## Project Overview
Superconducting qubit calibration system for Quantum Machines OPX1000 hardware (MW-FEM + LF-FEM), built on three layers: **qm-qua** → **quam** → **qualibrate** (see Workspace Packages for details).

## Related — shared experiment API
This repo is the **QM reference backend** for **`scqo`**, the vendor-neutral protocol/parameters API shared with the Qblox driver, so the same experiment runs on either instrument through one `Session`. The QM backend lives in `customized/scqo/`; qualibrate's `calibrations/` path runs the same probes directly (see **Probes vs shells**). Design, the `Session` contract, and cross-repo terminology (Experiment = probe + estimator) live in `SCQO\CLAUDE.md`; this repo's analysis nodes consume scqat **estimators** (`scqat.estimators`).

## Folder Roles

| Folder | Role | Editable? |
|--------|------|-----------|
| `quam_config/` | QUAM class definition + scripts to generate initial config for `quam_state/`. `my_quam.py` is the entrypoint class used everywhere. | Yes |
| `quam_state/` | Serialized instrument config files (`state.json`, `wiring.json`). Not code. | Generated output |
| `calibrations/` | Calibration node scripts shown in the qualibrate GUI. Mix of **vendored official** nodes (copied in by `sync_official.py`) and **custom `LCH_*`** nodes. | Only `LCH_*` files |
| `calibration_utils/` | Vendored official support code (copied in), satisfies relative imports from official nodes. | No (regenerate via `sync_official.py`) |
| `customized/` | All lab-specific code: custom pulses, macros, QUAM component extensions, and calibration node logic. | Yes |
| `data/` | Data storage only. No code. Skip. | N/A |

## Custom vs Official Nodes

**Official nodes** (vendored from `qua-libs` via `sync_official.py`):
- `calibrations/<name>.py` → copied-in, do NOT edit (overwritten on next sync)
- `calibration_utils/<name>/` → copied-in, do NOT edit (overwritten on next sync)

**Custom LCH nodes** (this lab's own code):
- `calibrations/LCH_<name>.py` → calibration script (GUI entry point)
- `customized/node/LCH_<name>/` → qualibrate-side code for the node: `parameters.py` (GUI schema; required unless the node reuses a vendored `Parameters`), plus the node's `analysis.py` (scqat adapter) and `update.py` (state-update policy) once extracted. See **Probes vs shells** below for what goes where.
- `customized/probes/<name>.py` → the instrument-acquisition half, shared with scqo. See **Probes vs shells**.
- `customized/components/` → shared pulse shapes, macros, QUAM extensions

## Probes vs shells (`customized/probes/` + `customized/node/`)
LCH nodes are being refactored so qualibrate is a thin shell, not the architecture. The split is by
**who calls the code**, which is also an **import rule**:

| Folder | Side | May import |
|---|---|---|
| `customized/probes/<name>.py` | acquisition: params in → `xr.Dataset` out. Called by BOTH the qualibrate shell and the scqo `QMBackend` (`customized/scqo/`, live since v0.1.x). | qm.qua, quam, qualang_tools, `qualibration_libs.core`/`.data`. **NEVER** qualibrate, scqo, or scqat — probes acquire, they never fit. |
| `customized/node/LCH_<name>/` | the qualibrate node: `parameters.py` (GUI schema), `analysis.py` (scqat estimate adapter), `update.py` (pure `compute_update` + `apply_update` state-update policy). | qualibrate, vendored official params, scqat (lazy, inside `fit`). |
| `calibrations/LCH_<name>.py` | qualibrate shell: `@node.run_action` glue (~3–10 lines each) unpacking `node.parameters` into probe/analysis/update calls. | everything. |

Shared probe helpers (`select_qubits` — the node-free `get_qubits`; `acquire` — the shared
execute-and-fetch) live in `customized/probes/_lib.py`.

**Virtual-detuning sign — a SILENT failure.** A probe realizing scqo's
`frequency_detuning_hz` must ramp the phase **negative on EVERY backend**: the second pi/2's
phase has to run BACKWARD relative to the free precession of a qubit sitting above its drive,
so the observed fringe is `applied + err` — scqo's shared `estimate()` writes
`drive_freq += (f_fit − applied)` for both backends. A frame rotation and a pulse-axis phase
are the SAME handedness (`frame_rotation_2pi` documents its argument as "the angle to add to
the current phase" and applies `envelope × e^{+iφ}`, exactly like Qblox's `Rxy(phi=…)`), so
the negation is not compensating a cross-vendor asymmetry — it is genuinely required on both.
An earlier version of this note claimed the opposite handedness; that claim was false and it
is what produced the Qblox sign confusion (fixed 2026-08-01, pinned by
`LCHQBDriver/tests/test_ramsey_detuning.py`). Get it backwards
and every accepted update DOUBLES the residual detuning instead of cancelling it, while the fit
still converges and looks clean (chipA q1, 2026-07-28: +47.9 k → +95.7 k → +191.5 k Hz).
BOTH frame-ramping probes carry the negation: `customized/probes/qubit_ramsey.py`
(`customized/node/LCH_Ramsey/update.py` ADDS `d_f01` to match) and
`customized/probes/pair_qcq_zz_coupler_freq.py`, where the stakes are lower — un-negated, its
reported `zz = f_fit − detuning` is sign-flipped, but the zero crossing that feeds the only
writeback is sign-invariant, so no accepted update was ever wrong. The official nodes leave the
ramp un-negated (`06a_ramsey.py` subtracts to compensate; `19_zz_off_jazz.py` takes only
argmin|ζ|, which cannot tell) — do not copy either sign into a probe.

**Why this split:** the scqo contract is that a driver contributes only `probe()`; estimate/update are
inherited from `scqo.experiments` + scqat. So the probe is the one piece both orchestrators share and
must stay framework-free, while analysis/update are qualibrate-path adapters. ("probes" matches scqo's
canonical vocabulary and avoids colliding with scqo's `Experiment`.)

**Two orchestration paths, one estimator implementation:**
- **scqo-driven** (live, migrated experiments only): scqo `Session` owns the probe→estimate→update
  lifecycle, data saving, and run history; it calls `customized/probes/<name>.py` and runs the scqat
  estimator itself.
- **QM-only** (today): qualibrate owns orchestration, saving, GUI approval; the LCH node calls its own
  `customized/node/LCH_<name>/{analysis,update}`. Vendored **official** nodes keep QM's built-in
  analysis and never run under scqo.
- In **both** paths the estimator *implementation* lives once in scqat — only the calling shell differs.

**Lab-tuned parameter defaults** go in the lab `Parameters` subclass (e.g. `LCHNodeSpecificParameters`
in `customized/node/LCH_power_rabi/parameters.py`), **never** in vendored `calibration_utils/` (which
`sync_official.py` reverts).

**Readout-power punchout probes:** both `resonator_spectroscopy_power_chain` (sweeps full-scale
power) and `resonator_spectroscopy_power_amp` (sweeps amplitude prefactors) take absolute-dBm inputs
(`min/max_power_dbm`).

### Flux headroom — `customized/probes/_flux_limits.py`
**The DAC rail is a property of the PORT, not a constant.** An OPX1000 LF-FEM analog output
reaches ±0.5 V in `direct` mode and ±2.5 V in `amplified` mode, and a stored waveform peak at or
above its port's full scale is clipped on hardware while the SIMULATOR SHOWS NOTHING. A probe that
hardcodes 0.5 refuses every amplified-mode config outright (`state_lib/10Q` runs all ten flux ports
`amplified` with `const` at 1.25 V) — which is what three probes did until 2026-07-29. **No probe
carries its own rail constant; do not reintroduce one.**

**Two frames, two entry points**, mirroring scqo's `FluxSweepParameters` /
`FluxPulseSweepParameters` 1:1 so the same word describes the experiment and the check:
- `check_flux_bias_absolute` — `set_dc_offset` REPLACES the standing bias, so the swept value IS
  the line voltage and **no idle term is added**;
- `check_flux_pulse_relative` — `play("const", amplitude_scale=…)` rides ON the standing bias, so
  the check is `|idle + excursion|` and the DAC emits the SUM.

Confusing the two is itself silent: adding `idle_v` in the absolute frame refuses legal sweeps,
omitting it in the relative frame admits clipping ones. Probes with their own `flux_point` argument
pass it to `idle_offset_v`; probes without one use `declared_idle_offset_v`, because what
`initialize_qpu` applies IS the declaration and no override can disagree. A **coupler** names its
points `off`/`on` but its attributes `decouple_offset`/`interaction_offset` — resolving by the
`<point>_offset` convention alone raises on every coupler.

**Severity is split by whether the DAC LIES**, and the split is load-bearing:
- *clipping* (stored op or idle bias past the rail) → refuse, per-probe and in the session audit;
- *reach* (the `const` = rail/2 convention) → **advisory only**. An undersized `const` emits exactly
  what was asked, there is just less range available, and the `amplitude_scale` bound already
  refuses the moment a sweep actually needs more. The live 5Q4C couplers sit at 0.15 V and have
  always run — making that fatal would block every session. It is enforced in exactly ONE place,
  `quam_fields.flux_headroom_warnings`; the per-sweep helpers deliberately do not.

`quam_fields.flux_headroom_problems` / `_warnings` audit the whole tree once from
`scqo/backend_factory.py`, beside `flux_point_problems`, so a bad config reports every offending
port at once instead of one probe dying on the first it touches.

**Flux-amplitude sweeps: absolute volts or prefactor.** Both pair swap probes take
`amp_mode="absolute"|"prefactor"` (and `flux_role` selecting which member's z carries the pulse).
scqo drives them in `"absolute"`, where the swept values ARE the emitted volts;
`pair_qq_chevron`'s default stays `"prefactor"` — a factor on the QUAM-computed |11>-|02>
amplitude — because that is what the qualibrate node uses. The chevron's two QUA branches (baked
below 17 ns, stretched `const` above) must emit the same volts for a given sweep point; the
arithmetic is factored into the pure `resolve_amplitudes` helper and pinned by
`tests/test_pair_swap_probes.py`.

Prefactor mode needs `freq_vs_flux_01_quad_term` on the control qubit, and **7 of the 9 live chipA
pairs have it unset** (only `coupler_q5_q6` and `coupler_q6_q7` resolve), so the qualibrate chevron
node cannot run on them — it used to die on a bare `ZeroDivisionError`, and now refuses naming the
missing field. Absolute mode never reads it and works on all nine, which is why scqo drives it that
way. Fix the underlying gap by measuring the flux arch (`qubit_spectroscopy_flux_pulse`).

The partial-swap calibration workflow (pair_swap_flux_map → the `quam_config/register_*.py`
scripts → qc_n_swap_amp) is documented in SCQO TUTORIAL §12.

**Readout output at the scqo boundary (the readout schema — SCQO TUTORIAL §11 / CLAUDE.md digest):**
the shot axis is `shot_idx`; per-shot discriminated data stays `state` (integer LEVELS, qutrit-capable);
FPGA-averaged discriminated data is `population` — the backend renames the averaged `state` stream when
the experiment's contract accepts `population` and no accepted form carries per-shot `state`. The pair
maps store `joint_population` over role-ordered `joint_state` labels ("00".."11", digits high,low):
`_pair_roles.JointPopulationMixin` reorders the FPGA `state_gg/ge/eg/ee` digits (control,target) →
(high,low), while `qc_n_swap_amp` keeps per-shot member states and reduces through scqo's shared
`states_to_joint_population` (its `readout_mode="shot"` skips the reduction and stores every shot).
A combo a probe cannot realize is refused by name — `qc_n_swap_amp` refuses non-control
`drive_side`/`flux_side` (the swap macro's `ctrl_amp` is the only swept knob).

**T1-tracking probes (`qubit_t1_ade` / `qubit_t1_bayesian`):** the repo's first on-FPGA-arithmetic
probes (`Math.div/ln/sqrt/exp` on `fixed` ∈ [-8, 8); the numeric-range rationale — clip floors,
TIME_SCALE_US, the u = 1/k reparametrization — lives in each probe's docstring). Their streams are
heterogeneous (per-block scalars + per-shot arrays + timestamp streams), which `XarrayDataFetcher`
refuses, so each ships its OWN `acquire()` (the tomography pattern) and the shell returns the
3-tuple `(prog, sweep_axes, probe_module)`. ms→cycles conversions MUST use `Cast.mul_int_by_fixed`
— the fixed-arithmetic product wraps modulo 16 to a garbage wait (pinned by
`tests/test_t1_tracking_shells.py`). The Bayesian probe reads QUAM's `confusion_matrix` directly
off the tree (dead to SCQO per the placement rule); both shells refuse a missing readout
threshold, and the Bayesian one a missing confusion matrix, BY NAME before any QUA is built.

**Migration status:** qualibrate-node migration is in progress; the `customized/` split into a
standalone QM-backend repo (symmetric with LCHQBDriver) is decided but deferred until migration
completes — the shells→probes import rule above is the boundary the split will cut along.

### State authority during the transition (scqo `state_sync` rule)
Two writers exist for QUAM today: unmigrated qualibrate nodes (write QUAM directly) and scqo's
`RecordingDevice` (owns its own state JSON). To prevent a stale scqo state file from clobbering
fresher QUAM calibrations at startup, **QM sessions MUST run `state_sync="pull"`** (scqo's default):
the vendor wins at startup, scqo loads only its change history, and pushes only values it freshly
measures. The migration finish line is flipping this device to `"push"` — do that only when no
qualibrate node writes QUAM anymore. (`customized/scqo/backend_factory.py` enforces
this — the guard fires before any QUAM state is loaded.)
**Which QUAM state loads** is decided by the device's cooldown setup alone: the SELECTED
`[<cycle>.setup.<name>]` block's vendor folder — DERIVED since scqo v0.9 as
`<device>/<cycle>/<name>/backend_config/` (no path key in the registry; users pick a setup with
`scqo user --setup <name>`, a single-setup cycle auto-selects) — must hold `state.json` +
`wiring.json` under exactly those canonical names — never rely on `~/.qualibrate` resolution for
scqo sessions; keep qualibrate's own `[quam] state_path` pointed at the same folder on machines
that run both stacks (re-point it each new cooldown).

### scqo student surface
Students use the **`scqo` command** (`run/find/accept/suggest/set/tag/state/user/device/doctor`) from any
directory in `.venv-qm`, selecting a sample and setup with
`scqo user --device <name> [--setup <name>]` (written to `~/.scqo/user.toml`); `scqo run <name>` is
the one way to run an experiment (never add per-command wrappers or launcher stubs). This repo
contributes `customized/scqo/backend_factory.py`, registered under the `scqo.backends` entry-point
group (name `qm`): `build_backend(cfg, setup, roster)` fires the `state_sync="pull"` guard BEFORE any
QUAM state is touched, then loads the setup's vendor folder (`setup["instrument_config"]`, injected by
scqo from the registry keys; canonical names `state.json` +
`wiring.json` — the single QUAM-state authority; loud SystemExit when missing) and threads the device
ROSTER into the backend — the driver serves a view per CHANNEL ENTITY (`q1_xy` → QUAM `q1.xy`,
`q1_ro` → `q1.resonator`, `q1_z` → `q1.z`, a coupler mode's `*_z` → the pair's `TunableCoupler`) plus
a composite view over the QUAM `qubit_pair` for the per-operation gate knobs, so every name resolves
through the roster and never by string arithmetic. `simulated` is the
practice mode. Only migrated experiments run under scqo here; all other calibrations still run
through the qualibrate GUI (legacy, frozen; do not merge).

The my_quam root-class toggle governs test fixtures too: the `machine` fixture skips when the live
`quam_state/` doesn't match the active root (e.g. flux-tunable pairs under a `FixedFrequencyQuam`
root).

## Key Entrypoints
- `quam_config/my_quam.py` → defines the `Quam(FluxTunableQuam)` class imported by every calibration node and config script. The custom-type bindings (`qubit_type = ChargeTunableTransmon`, `qubit_pair_type = LCH_FluxTunableTransmonQCQPair`) are **toggled in/out per experiment** — they are intentionally commented out by default and uncommented only when a run needs the custom charge-tunable types. Do NOT treat either state as "wrong"; read the live class body to see what is active, and ask before flipping it.
- `customized/quam_builder/` → custom qubit type `ChargeTunableTransmon`
- `customized/qubit_pair/` → custom qubit pair `LCH_FluxTunableTransmonQCQPair`

## Operational Notes (verified against the working tree)
- **Official code is VENDORED (copied), and committed.** Official `calibrations/<name>.py` and `calibration_utils/<name>/` are real files copied in by `sync_official.py` from `calibration_links.toml` (source = an external `qua-libs_official` checkout, overridable via the `QUA_LIBS_OFFICIAL` env var). They ARE tracked in git, so the repo is self-contained — clone + `pip install -e .` runs official nodes with no external checkout present. `official_sync.json` records which upstream commit is currently vendored. **Do not edit vendored official files in place; do not symlink them.** Why copy: qualibrate scans the library folder non-recursively and needs each node to be a real `.py` file (see `qualibrate/qualibration_node.py` `scan_folder_for_instances`), and copies avoid the Windows-admin requirement, git directory-symlink quirks, and the old cp950 encoding patch (recent qualibrate already reads node files as UTF-8).
- **Updating official (~every 2 months):** `git pull` in the `qua-libs_official` checkout → `python sync_official.py` → `git diff` (review upstream changes, especially `calibration_utils/` that `LCH_*` nodes import) → commit. The custom `LCH_*` files and `customized/` have distinct names and are never touched by the sync.
- **`calibrations/offline_graph/`** holds `LCH_graph_*.py` post-processing/graph scripts. Editable lab code, same `LCH_` convention as nodes. qualibrate does not list them (it scans `calibrations/` non-recursively); run them manually.
- **Environments (2026-07-05):** the scqo path runs in the uv venv `D:\github\.venv-qm`,
  rebuildable from `requirements-qm.lock.txt` (exact pins frozen from `LCHQM_test`; see
  SCQO/INSTALL.md §1). Sibling envs: `.venv-view` (data browsing, no instrument libs —
  the lab's daily default) and `.venv-qblox` (Qblox measurement). Conda is being retired
  lab-wide; `qm.bat` already targets `.venv-qm` (`qm.bat conda` = legacy fallback) and
  the conda envs get deleted after one validated qualibrate GUI session.
- **Run / setup** (`qm.bat` activates `.venv-qm`; `qm.bat conda` forces the legacy fallback):
  - `qm.bat` (Windows) / `qm.command` (macOS/Linux) → wrappers that activate the env and run `qualibrate start` (launches the GUI server). These replaced the old Windows-only `start_server.bat` / `setup_qualibrate_config.bat`.
  - `setup-qualibrate-config` → one-time qualibrate config setup.
- **Packaging:** `pyproject.toml` — Python `>=3.10,<3.13`, black `line-length = 120`. Wheel packages: `calibrations`, `calibration_utils`, `quam_config`, `customized`.
- **Tests:** `tests/conftest.py` holds the shared fixture chip — a schema-3 `ROSTER_TOML` (q1/q2
  flux-tunable, fixed-frequency q3, the coupler mode `q1_q2_c` with its own flux wire, one
  multiplexed feedline) plus a stub QUAM tree whose names deliberately disagree with it
  (`coupler_q1_q2` vs `q1_q2`/`q1_q2_c`), so every resolution has to go through the roster.
  `tests/test_scqo_glue.py` (scqo↔backend glue, the per-kind + per-operation fieldmap drift alarms,
  the `components()` witness), `tests/test_qm_backend.py` (entity surface on the stub; probe
  equivalence + the absolute-power chain on the LIVE quam_state, skipped when the root-class toggle
  does not match), `tests/test_experiment_surface.py` (`customized/scqo/experiments/_vendor.py`, the
  probes' one door out of the neutral surface). How to run it: **## Tests** below.
- **External analysis dependency.** LCH analysis nodes lazily import `scqat` (`D:\github\scqat`, the lab's analysis tool that **replaced** the older `qcat`/`D:\github\QCAT`), installed editable. It is **not declared in `pyproject.toml`** and must be installed in the runtime env — now present (editable) in both `LCHQM` and `LCHQM_test` (the launcher env; verified 2026-06-07) — or those nodes raise `ImportError` at plot time. Official (non-`LCH_`) nodes do not depend on it and run regardless. The qcat→scqat migration of the active `LCH_*` nodes is complete (`calibrations/exclude/` still references qcat); see `ANALYSIS_MIGRATION.md`.

- **Active reset** (`reset_method="active"`) lives in `customized/scqo/experiments/_reset.py`, the
  ONE door every shell resolves its `reset_type` through (`check_reset_method`), with `QMBackend.acquire`
  re-checking before `probe()`. Opt-in is per shell (`supports_active_reset`, default DENY) and limited to
  the six coherent-drive carriers (relaxation/ramsey/echo/power_rabi + the two T1 trackers
  qubit_t1_ade/qubit_t1_bayesian); everything else refuses BY NAME —
  the readout-sweep probes, `single_shot_readout` (it IS the discriminator calibration), the driven-dwell
  spectroscopies. The sequence is QUAM's `BaseTransmon.reset_qubit_active` (repeat-until-success). Four
  QM-specific rules, each a SILENT failure if broken: (1) `active_reset_rounds` → QUAM `max_attempts` is an
  UPPER bound (the loop exits early), not Qblox's exactly-N; (2) BOTH `readout_threshold` AND
  `readout_rus_threshold` are required even at rounds=1 (the `while_` exit is built regardless), and an
  uncalibrated value is `None` on QM — `I > None` dies deep in the QUA DSL naming nothing, so the guard
  refuses first; (3) `readout_depletion_s` must be governed — QUAM's 16 ns factory default is refused
  (never `None` on QM, so a None-only guard would be dead code); (4) `thermalization_time_ns` + `active`
  is refused, not ignored. Offline proves the policy and that the QUA program builds; the feedback loop
  is hardware, and the chipA walkthrough that closes that gap is the REMAINING step (the QUA path has
  mileage through the `LCH_graph_*` qualibrate scripts, but the scqo shells need their own validation).
- **Sequence preview** (`scqo run <name> --preview` → `QMBackend.preview`): dumps
  `generate_qua_script(prog, machine.generate_config())` to `qua_script.py` (offline), then
  AUTO-TRIES the gateway simulator → `simulated_waveforms.html` (plotly; the wiring
  `network` host, a 2 s TCP probe gates the attempt, any failure degrades to script-only
  with a PreviewWarning — a WEDGED gateway still costs the vendor request deadline).
  Simulation runs on the gateway server, never the qubits. `--no-simulate` = guaranteed
  offline; `--simulate-ns` widens the 20 µs default window (cap `_SIMULATE_MAX_NS`; a
  thermal shot's leading ms-scale wait can make the default window legitimately EMPTY —
  warned, no file). Six shells acquire INSIDE `probe()` and are refused by
  name BEFORE it runs, each declaring `probe_self_acquires = "<why>"` (backend constant
  `SELF_ACQUIRING_ATTR`; default-ALLOW polarity — contrast `supports_active_reset`'s
  default-DENY): pair_swap_chevron, qc_n_swap_amp, qubit_drag_equator, qubit_drag_alternating,
  qubit_ramsey_cryoscope, qubit_xyz_delay. Any NEW self-acquiring shell must declare it —
  `tests/test_preview.py`'s census enforces the set both directions.

## Tests

**Always `uv run --no-sync pytest tests/ -q`. The `--no-sync` is not optional.**
`scqo` is declared as an OPTIONAL extra here (`[project.optional-dependencies] scqo`) — a
deliberate choice, so this repo still works for qualibrate users with no scqo installed. But
`uv run` **syncs the env to declared deps by default**, and extras are not declared, so a bare
`uv run pytest` silently **uninstalls the editable `scqo` + `scqat`** from `.venv`. The symptom
is `tests/test_experiment_surface.py` and `tests/test_qm_backend.py` failing collection with
`ModuleNotFoundError: No module named 'scqo'`. Repair:

```
uv pip install -e D:\github\SCQO -e D:\github\scqat
```

(LCHQBDriver has no such trap — it declares `scqo` as a hard dependency, so plain `uv run` is
safe there. Do not copy its command over here.)

**Then just run the whole suite: ~258 tests, ~40 s.** At this size a per-file selection map would
cost more attention than it saves — unlike SCQO (618 tests, ~10 min) and scqat (329 / ~84 s), the full
suite IS the targeted run. Run it before every commit.

The narrowing worth knowing: most of this suite is **pure unit tests needing no QM/QUAM/hardware**,
so they are instant — loop on those while iterating, and take the full suite before committing.

| File | Covers | Needs QM stack? |
|---|---|---|
| `test_quam_fields.py` | the single neutral-field ↔ QUAM mapping (stub qubit) | no |
| `test_flux_pulse_amplitude.py` | `probes/_flux_limits.py`: the two frames, the rail per port, the idle sum, the remedy messages | no |
| `test_flux_headroom_guard.py` | the whole-tree audit + the clipping-vs-reach severity split | no |
| `test_reset_method.py` | the active-reset door: the opt-in census (4 carriers), the refusals (opt-in, thermalization override, uncalibrated discriminator/depletion), the `acquire()` backstop | no |
| `test_qc_populations.py` | shared swap-reset population helpers | no |
| `test_preview.py` | the `probe_self_acquires` census (exactly the six, both directions) + preview's refusal fires BEFORE `probe()` on the stub | no |
| `test_power_rabi_update.py`, `test_ramsey_update.py`, `test_readout_frequency_update.py` | the pure `update()` decisions of the matching `LCH_*` node | no |
| `test_experiment_surface.py` | `customized/scqo/experiments/_vendor.py` — the probes' one door out of the neutral surface | yes |
| `test_qm_backend.py` | entity surface on the stub; probe equivalence, absolute-power chain + active-reset program build on the LIVE quam_state | yes |
| `test_scqo_glue.py` | the `scqo` CLI works in THIS venv + the qm factory (slowest, ~7 s) | yes |

## Workspace Packages (Read-Only)
The vendor dependency stack (`qm` QUA control → `quam` hardware abstraction → `quam_builder` → `qualibrate` GUI) is available read-only in the workspace; do NOT modify. For the SC-qubit repo layout see the global workspace map `C:\Users\shiau\.claude\CLAUDE.md`.

## Rules for the AI assistant

1. **Do NOT edit vendored official files.** Non-`LCH_` files in `calibrations/` and everything in `calibration_utils/` are official `qua-libs` code copied in by `sync_official.py`. Edits would be overwritten on the next sync — change behavior in `LCH_*` / `customized/` instead, or update upstream and re-sync.
2. **Always present a plan before modifying code.** Only modify code after the user gives explicit approval.
3. **Editable code lives in:** `customized/`, `quam_config/`, and `LCH_*` files in `calibrations/`.
4. **Skip `data/`** — data storage only, no code to read or modify.
5. **Flag critical dependencies.** When creating or modifying customized code, if any dependency (from `qm`, `quam`, `quam_builder`, `qualibrate`, or other packages) is critical to the implementation, explicitly tell the user which dependencies are involved.
6. **Check workspace completeness.** If the workspace is missing expected folders (e.g., `qm`, `quam`, `quam_builder`, `qualibrate`), notify the user so they can add them on the current device.
7. **Report conflicts.** If existing code contradicts these instructions (e.g., an LCH node has unnecessary analysis.py/plotting.py files, or imports don't follow the expected pattern), inform the user before making changes.
