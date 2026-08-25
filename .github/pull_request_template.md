<!--
Contribution guide: https://github.com/shiau109/SCQO/blob/main/CONTRIBUTING.md
Run the preflight first and paste its output below:
    python SCQO/scripts/check_contribution.py
-->

## What this changes

<!-- One paragraph. What behaviour differs afterwards? -->

## Sibling pull requests

A feature crossing repos cannot be atomic here (no shims ship), so it lands as several
PRs merging in the order **scqat -> SCQO -> drivers**. A red SCQO check is expected
while its scqat half is unmerged.

- [ ] This change is confined to one repo, **or** the sibling PRs are linked here:

| repo | PR | merge order |
|---|---|---|
| scqat | | 1 |
| SCQO | | 2 |
| scqo-qblox / scqo-qm | | 3 |

## How it was verified

**Exact command(s) run:**

```
```

**Result:**

```
```

**Validation level** — pick one, and be honest; this string goes into the release ledger:

- [ ] `offline` — the test suites pass
- [ ] `hardware <chip> <date>` — it ran on real hardware
- [ ] `unverified` — changed but not run (say why)

## Preflight output

```
```

## Release note

Draft the `RELEASES.d/<slug>.toml` fragment here — **do not commit it**. The ledger may
only list complete features, so a maintainer commits it once the last repo's PR lands.
Format: https://github.com/shiau109/SCQO/blob/main/RELEASES.d/README.md

```toml
name = ""
repos = { }
kind = ""          # breaking | additive | fix
coupling = ""      # lockstep floors, especially SILENT-failure ones
validated = ""
notes = ""
```

## Checklist

- [ ] Targeted tests pass and the exact command is above.
- [ ] No shim, alias or compatibility layer added (house rule: clean cutover + an upgrade note).
- [ ] New experiment? Promotion checklist worked, and `python scripts/update_docs.py`
      re-run so the generated blocks include it.
