"""Registration completeness for ``customized/scqo/experiments``.

Registration is a MANUAL side-effect import line per module in the package
``__init__.py`` (no autodiscovery). A module missing its line is a silently
unregistered experiment: the class exists, ``@register`` never runs, and
``scqo run <name>`` reports an unknown experiment while every test still
passes. Static (AST) on purpose, like
``test_amp_limits.test_only_one_module_defines_the_bound``: no scqo/QM stack
needed, and the check works even for a module nothing imports.
"""

import ast
from pathlib import Path

EXPERIMENTS = Path(__file__).resolve().parents[1] / "customized" / "scqo" / "experiments"


def _imported_in_init():
    tree = ast.parse((EXPERIMENTS / "__init__.py").read_text(encoding="utf-8"))
    return {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module is None
        for alias in node.names
        if not alias.name.startswith("_")
    }


def test_every_experiment_module_has_its_import_line():
    modules = {
        path.stem
        for path in EXPERIMENTS.glob("*.py")
        if path.name != "__init__.py" and not path.name.startswith("_")
    }
    imported = _imported_in_init()

    missing = sorted(modules - imported)
    assert not missing, (
        "experiment modules with no `from . import <name>` line in "
        "customized/scqo/experiments/__init__.py — their @register never runs, "
        f"so the experiment is invisible to scqo: {missing}"
    )

    stale = sorted(imported - modules)
    assert not stale, (
        "customized/scqo/experiments/__init__.py imports modules that no longer "
        f"exist: {stale}"
    )
