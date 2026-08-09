"""Repin a device's QUAM ROOT ``__class__`` to the lab's mixed transmon root.

Why: ``quam/core/quam_instantiation.py`` honours the stored ``__class__`` over the
class ``.load()`` was called on, at the ROOT as well as for children. So a device
that stores ``quam_config.my_quam.Quam`` resolves to whatever that module currently
says — an uncommitted local edit, not a stable identity. A root must name a
CONCRETE class.

``MixedTransmonQuam`` handles fixed-frequency, flux-tunable and mixed trees alike,
so it is the one root every device can name. Neither current device is actually
mixed; pinning both anyway is deliberate, so the mixed path gets real-config
exercise from the all-fixed (chipA) and all-tunable (5Q4C) ends before a genuinely
mixed chip depends on it.

Only the root ``__class__`` string changes. Every qubit's own ``__class__`` is left
alone — this is the inverse of ``migrate_thermalizing_transmon.py``, which rewrites
the children and asserts the root is unchanged.

    python scripts/migrate_root_class.py --dry-run <dir> [<dir> ...]
    python scripts/migrate_root_class.py <dir> [<dir> ...]

<dir> is a backend_config folder (or a state.json path). Idempotent: a file already
pinned is reported and skipped. The original is copied to
``<name>.pre-rootclass.bak`` first.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

NEW_ROOT = (
    "scqo_qm.quam_builder.architecture.superconducting.qpu"
    ".mixed_quam.MixedTransmonQuam"
)


def migrate(state_path: pathlib.Path, *, dry_run: bool = False) -> bool:
    """Repin one ``state.json`` root. Returns True when it changed."""
    text = state_path.read_text(encoding="utf-8")
    before = json.loads(text)
    root = before.get("__class__")
    qubits = before.get("qubits", {})
    child_classes = {n: q.get("__class__") for n, q in qubits.items()}

    print(f"\n=== {state_path}")
    print(f"    root __class__ : {root}")
    print(f"    qubits         : {len(qubits)}  pairs: {len(before.get('qubit_pairs', {}))}")

    if root == NEW_ROOT:
        print("    already pinned - nothing to do")
        return False
    if root is None:
        print("    NO ROOT __class__ - refusing to guess; add one by hand")
        return False

    # Replace only the ROOT occurrence. A root class string should never also be a
    # child's class, so requiring exactly one occurrence keeps a blind string
    # replace honest instead of hoping.
    needle = f'"{root}"'
    n = text.count(needle)
    if n != 1:
        print(f"    REFUSED - {needle} appears {n}x, cannot target the root safely")
        return False

    new_text = text.replace(needle, f'"{NEW_ROOT}"')
    print(f"    {root.rsplit('.', 1)[-1]} -> {NEW_ROOT.rsplit('.', 1)[-1]}")

    if dry_run:
        print("    --dry-run: not written")
        return False

    backup = state_path.with_suffix(".json.pre-rootclass.bak")
    shutil.copy(state_path, backup)
    print(f"    backup         : {backup.name}")
    state_path.write_text(new_text, encoding="utf-8", newline="")

    after = json.loads(state_path.read_text(encoding="utf-8"))
    assert after.get("__class__") == NEW_ROOT, "root not repinned - restore the backup"
    assert set(after) == set(before), "top-level keys changed - restore the backup"
    assert {
        n: q.get("__class__") for n, q in after.get("qubits", {}).items()
    } == child_classes, "a qubit __class__ changed - restore the backup"
    print("    OK - only the root changed")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("paths", nargs="+",
                        help="backend_config folders, or state.json paths")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would change, write nothing")
    args = parser.parse_args(argv)

    changed = 0
    for raw in args.paths:
        p = pathlib.Path(raw)
        state = p if p.suffix == ".json" else p / "state.json"
        if not state.is_file():
            print(f"\n=== {state}\n    NOT FOUND - skipped")
            continue
        changed += bool(migrate(state, dry_run=args.dry_run))

    print(f"\n{changed} file(s) changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
