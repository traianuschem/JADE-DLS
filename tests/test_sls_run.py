"""Tests for ade_dls.engine.sls_run (Qt-free SLS helpers shared by the result dialogs) and
headless.run_sls. Expected strings/values are written out by hand from the documented rules;
the SLS numerics themselves are covered by test_sls_number_weighting.py."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from ade_dls.engine import sls_run

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"
DIALOGS = Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "dialogs"


# ----------------------------------------------------------------------------- q2 range
@pytest.mark.parametrize("q2_min,q2_max,expected", [
    (0, 0, None),
    (1e-4, 0, (1e-4, float("inf"))),
    (0, 2e-4, (0.0, 2e-4)),
    (1e-4, 2e-4, (1e-4, 2e-4)),
])
def test_q2_range_from_limits(q2_min, q2_max, expected):
    assert sls_run.q2_range_from_limits(q2_min, q2_max) == expected


# ----------------------------------------------------------------------------- cell formatting
@pytest.mark.parametrize("val,expected", [
    (float("nan"), "—"), (None, "—"),
    (0.5, "0.5000"), (0.01, "0.0100"), (10000.0, "10000.0000"),
    (0.001, "1.0000e-03"), (20000.0, "2.0000e+04"), (-0.005, "-5.0000e-03"),
    ("Total", "Total"), (3, "3.0000"),
])
def test_format_sls_cell(val, expected):
    assert sls_run.format_sls_cell(val) == expected


@pytest.mark.parametrize("column,val,expected", [
    ("qRg_max", 1.31, True), ("qRg_max", 1.3, False), ("qRg_max", 0.5, False),
    ("qRg_max", "x", False), ("qRg_max", float("nan"), False),
    ("Rg [nm]", 99.0, False),
])
def test_is_qrg_warning_flags_only_qrg_max_above_limit(column, val, expected):
    assert sls_run.is_qrg_warning(column, val) is expected


# ----------------------------------------------------------------------------- file discovery
def test_asc_files_in_folder_matches_any_case_and_ignores_other_files(tmp_path):
    for name in ("a.asc", "B.ASC", "c.txt", "d.asc.bak"):
        (tmp_path / name).write_text("x")
    found = sorted(Path(p).name for p in sls_run.asc_files_in_folder(str(tmp_path)))
    assert found == ["B.ASC", "a.asc"]
    assert all(Path(p).parent == tmp_path for p in sls_run.asc_files_in_folder(str(tmp_path)))


def test_intensity_file_paths_uses_existing_files_once(tmp_path):
    (tmp_path / "a.asc").write_text("x")
    (tmp_path / "b.asc").write_text("x")
    base = pd.DataFrame({"filename": ["a.asc", "a.asc", "b.asc", "missing.asc", ""],
                         "folder": [str(tmp_path)] * 5})
    assert [Path(p).name for p in sls_run.intensity_file_paths(base)] == ["a.asc", "b.asc"]


@pytest.mark.parametrize("base", [None, pd.DataFrame(), pd.DataFrame({"filename": ["a"]})])
def test_intensity_file_paths_without_usable_basedata_is_empty(base):
    assert sls_run.intensity_file_paths(base) == []


# ----------------------------------------------------------------------------- load_intensity / run_sls
class _Analyzer:
    def __init__(self, base=None, df_intensity=None, ok=True):
        self.df_basedata, self.df_intensity, self.ok, self.loaded = base, df_intensity, ok, []
        self.sls_args = None

    def load_intensity_data(self, paths):
        self.loaded.append(list(paths))
        return self.ok

    def run_sls_analysis(self, n_populations, q2_range=None, exponent=6):
        self.sls_args = (n_populations, q2_range, exponent)
        return "summary"


def test_load_intensity_is_noop_when_already_loaded():
    an = _Analyzer(df_intensity=pd.DataFrame({"x": [1]}))
    assert sls_run.load_intensity(an) is True and an.loaded == []


def test_load_intensity_prefers_basedata_paths_over_folder(tmp_path):
    (tmp_path / "a.asc").write_text("x")
    other = tmp_path / "other"
    other.mkdir()
    (other / "z.asc").write_text("x")
    an = _Analyzer(pd.DataFrame({"filename": ["a.asc"], "folder": [str(tmp_path)]}))
    assert sls_run.load_intensity(an, str(other)) is True
    assert [Path(p).name for p in an.loaded[0]] == ["a.asc"]


def test_load_intensity_falls_back_to_folder(tmp_path):
    (tmp_path / "z.ASC").write_text("x")
    an = _Analyzer(pd.DataFrame({"filename": ["a"]}))
    assert sls_run.load_intensity(an, str(tmp_path)) is True
    assert [Path(p).name for p in an.loaded[0]] == ["z.ASC"]


def test_load_intensity_without_any_files_or_failed_loading_returns_false(tmp_path):
    assert sls_run.load_intensity(_Analyzer(pd.DataFrame({"filename": ["a"]}))) is False
    assert sls_run.load_intensity(_Analyzer(), str(tmp_path)) is False        # folder without .asc
    (tmp_path / "a.asc").write_text("x")
    assert sls_run.load_intensity(_Analyzer(ok=False), str(tmp_path)) is False


def test_run_sls_converts_dialog_limits_to_q2_range():
    an = _Analyzer()
    assert sls_run.run_sls(an, 2, 1e-4, 0.0, exponent=5) == "summary"
    assert an.sls_args == (2, (1e-4, float("inf")), 5)
    sls_run.run_sls(an, 1)
    assert an.sls_args == (1, None, 6)


# ----------------------------------------------------------------------------- dialogs
@pytest.mark.parametrize("name,analyzer", [("regularized_results_dialog.py", "laplace_analyzer"),
                                           ("laplace_postfit_dialog.py", "analyzer")])
def test_dialogs_use_the_shared_sls_helpers(name, analyzer):
    """Structure check (no Qt needed): the dialogs must not keep their own copies."""
    src = (DIALOGS / name).read_text(encoding="utf-8")
    assert f"run_sls(self.{analyzer}," in src
    assert "format_sls_cell(row[col])" in src and "is_qrg_warning(col, row[col])" in src
    assert "asc_files_in_folder(folder)" in src
    assert f"self.{analyzer}.run_sls_analysis" not in src
    assert "float('inf')" not in src and "os.listdir" not in src


# ----------------------------------------------------------------------------- headless
def test_headless_run_sls_without_intensity_data_raises():
    from ade_dls import headless
    with pytest.raises(ValueError, match="No intensity data"):
        headless.run_sls(_Analyzer(pd.DataFrame({"filename": ["a"]})))


def test_headless_run_sls_passes_arguments_through():
    from ade_dls import headless
    an = _Analyzer(df_intensity=pd.DataFrame({"x": [1]}))
    assert headless.run_sls(an, 2, (1e-4, 2e-4), 5) == "summary"
    assert an.sls_args == (2, (1e-4, 2e-4), 5)


@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_run_sls_on_real_series_gives_one_population_plus_total():
    from ade_dls import headless
    data = headless.load_series(str(ALV))
    lap, _ = headless.run_regularized(data, {"decay_times": np.logspace(-8, 1, 26)})
    summary = headless.run_sls(lap, 1)
    assert list(summary["Population"].astype(str)) == ["1", "Total"]
    assert summary["I-fraction [%]"].iloc[0] == pytest.approx(100.0, rel=1e-6)
    assert summary["Rh [nm]"].iloc[0] == pytest.approx(
        float(lap.regularized_final_results["Rh [nm]"].iloc[0]), rel=1e-6)
    assert lap.sls_summary is summary
    # same result when driven through the dialog helper (limits 0/0 = all angles)
    again = sls_run.run_sls(lap, 1, 0.0, 0.0)
    pd.testing.assert_frame_equal(summary, again)


# ----------------------------------------------------------------------------- dialog table (Qt)
_APP = None


@pytest.mark.parametrize("module,cls,attr", [
    ("regularized_results_dialog", "RegularizedResultsDialog", "sls_summary_table"),
    ("laplace_postfit_dialog", "LaplacePostFitRefinementDialog", "_sls_summary_table"),
])
def test_dialog_summary_table_uses_shared_formatting(module, cls, attr):
    pytest.importorskip("PyQt5.QtWidgets")
    import importlib
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication, QTableWidget
    global _APP
    _APP = QApplication.instance() or QApplication([])      # module-level: widgets crash without a live app
    dialog_cls = getattr(importlib.import_module(f"ade_dls.gui.dialogs.{module}"), cls)
    fake = SimpleNamespace(**{attr: QTableWidget()})
    df = pd.DataFrame({"Population": ["1", "Total"], "qRg_max": [1.95, 0.5], "I0": [0.001, float("nan")]})
    dialog_cls._sls_populate_table(fake, df)
    table = getattr(fake, attr)
    assert (table.rowCount(), table.columnCount()) == (2, 3)
    assert [table.item(r, 1).text() for r in range(2)] == ["1.9500", "0.5000"]
    assert [table.item(r, 2).text() for r in range(2)] == ["1.0000e-03", "—"]
    assert table.item(0, 1).background().color().name().lower() == "#ffb6c1"      # qRg_max > 1.3
    assert table.item(1, 1).background().color().name().lower() != "#ffb6c1"
