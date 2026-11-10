"""Architecture guard: GUI / computation-core separation.

Layers (lower must not import higher):
    ade_dls.analysis, ade_dls.core, ade_dls.utils   numerical kernels, parsers
    ade_dls.engine                                  Qt-free orchestration (analyzers, loading,
                                                    preprocessing, provenance, exports)
    ade_dls.gui                                     PyQt5 only

The checks run in subprocesses so that the state of this test process (PyQt5 may be installed)
cannot hide a dependency.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "ade_dls"
CORE_PACKAGES = ["analysis", "core", "utils", "engine", "benchmark"]

# Modules that moved from ade_dls.gui.* into ade_dls.engine.* (old path stays as an alias).
MOVED = {
    "ade_dls.gui.analysis.cumulant_analyzer": "ade_dls.engine.cumulant_analyzer",
    "ade_dls.gui.analysis.laplace_analyzer": "ade_dls.engine.laplace_analyzer",
    "ade_dls.gui.analysis.cumulant_plotting": "ade_dls.engine.cumulant_plotting",
    "ade_dls.gui.core.correlation_preprocessing": "ade_dls.engine.correlation_preprocessing",
    "ade_dls.gui.core.provenance": "ade_dls.engine.provenance",
    "ade_dls.gui.core.quality_assessment": "ade_dls.engine.quality_assessment",
    "ade_dls.gui.export.csv_export": "ade_dls.engine.export.csv_export",
    "ade_dls.gui.export.plot_series_export": "ade_dls.engine.export.plot_series_export",
}


def _core_module_names():
    names = []
    for pkg in CORE_PACKAGES:
        base = PKG / pkg
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(ROOT).with_suffix("")
            parts = list(rel.parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            names.append(".".join(parts))
    return names


CORE_MODULES = _core_module_names() + ["ade_dls.headless"]


def _run(code):
    """Run `code` in a clean interpreter (-I ignores PYTHONPATH, so the repo root is inserted)."""
    prelude = "import sys; sys.path.insert(0, %r)\n" % str(ROOT)
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-I", "-c", prelude + code], cwd=ROOT,
                          capture_output=True, text=True, timeout=300, env=env)


def test_engine_package_exists():
    assert (PKG / "engine" / "__init__.py").exists()


@pytest.mark.parametrize("module", CORE_MODULES)
def test_core_layer_does_not_import_gui_or_qt(module):
    """Static check: no import of ade_dls.gui / PyQt in the core layers (AST, no execution)."""
    path = ROOT / (module.replace(".", "/") + ".py")
    if not path.exists():
        path = ROOT / module.replace(".", "/") / "__init__.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            targets = [("." * node.level) + (node.module or "")]
        else:
            continue
        for t in targets:
            if t.startswith(("ade_dls.gui", "PyQt", "PySide")) or ".gui" in t:
                bad.append(f"line {node.lineno}: {t}")
    assert not bad, f"{module} imports the GUI layer: {bad}"


def test_core_layers_import_without_pyqt():
    """Runtime check: every core module imports although PyQt5 is made unimportable."""
    code = (
        "import sys, importlib\n"
        "sys.modules['PyQt5'] = None\n"
        "for m in %r:\n"
        "    importlib.import_module(m)\n"
        "assert not [k for k in sys.modules if k.startswith('ade_dls.gui')], 'gui layer was imported'\n"
        "print('ok')\n" % (CORE_MODULES,)
    )
    res = _run(code)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]


@pytest.mark.parametrize("old,new", sorted(MOVED.items()))
def test_old_gui_paths_are_aliases_of_engine_modules(old, new):
    """Backwards compatibility: the old import path yields the very same module object."""
    if old.startswith("ade_dls.gui.core."):
        pytest.importorskip("PyQt5")        # importing the package ade_dls.gui.core pulls in the Qt pipeline wrapper
    code = (
        "import importlib\n"
        "a = importlib.import_module(%r)\n"
        "b = importlib.import_module(%r)\n"
        "assert a is b, (a, b)\n"
        "print('ok')\n" % (old, new)
    )
    res = _run(code)
    assert res.returncode == 0 and "ok" in res.stdout, res.stderr[-2000:]


def test_engine_modules_have_no_leftover_copies_under_gui():
    """The moved modules exist once: the gui files are thin aliases (< 25 lines)."""
    for old in MOVED:
        path = ROOT / (old.replace(".", "/") + ".py")
        assert path.exists(), path
        assert len(path.read_text(encoding="utf-8").splitlines()) < 25, f"{old} is not an alias"
