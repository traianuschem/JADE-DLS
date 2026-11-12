"""Wiring of the optional solvent subtraction (ade_dls.analysis.sls.subtract_solvent) through
engine.sls_run.run_sls, headless.run_sls, LaplaceAnalyzer.run_sls_analysis and the two dialogs
(source-text checks, no Qt)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from ade_dls.engine import sls_run

DIALOGS = Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "dialogs"


class _OldAnalyzer:
    """Old-style fake: no solvent_folder keyword accepted."""
    def __init__(self, df_intensity=None):
        self.df_intensity = df_intensity
        self.df_basedata = None
        self.kwargs = None

    def run_sls_analysis(self, n_populations, q2_range=None, exponent=6):
        self.kwargs = dict(n_populations=n_populations, q2_range=q2_range, exponent=exponent)
        return "summary"


class _NewAnalyzer(_OldAnalyzer):
    def run_sls_analysis(self, n_populations, q2_range=None, exponent=6, solvent_folder=None):
        self.kwargs = dict(n_populations=n_populations, q2_range=q2_range, exponent=exponent,
                           solvent_folder=solvent_folder)
        return "summary"


# ----------------------------------------------------------------------------- engine.sls_run
@pytest.mark.parametrize("folder", [None, ""])
def test_sls_run_without_solvent_folder_keeps_old_call_signature(folder):
    an = _OldAnalyzer()
    assert sls_run.run_sls(an, 2, 1e-4, 0.0, exponent=5, solvent_folder=folder) == "summary"
    assert an.kwargs == dict(n_populations=2, q2_range=(1e-4, float("inf")), exponent=5)


def test_sls_run_with_solvent_folder_passes_it_on():
    an = _NewAnalyzer()
    sls_run.run_sls(an, 1, solvent_folder="some/folder")
    assert an.kwargs == dict(n_populations=1, q2_range=None, exponent=6, solvent_folder="some/folder")


# ----------------------------------------------------------------------------- headless
def test_headless_run_sls_passes_solvent_folder_only_when_set():
    from ade_dls import headless
    df = pd.DataFrame({"x": [1]})
    old = _OldAnalyzer(df)
    assert headless.run_sls(old, 1) == "summary"
    assert old.kwargs == dict(n_populations=1, q2_range=None, exponent=6)
    new = _NewAnalyzer(df)
    headless.run_sls(new, 2, (1e-4, 2e-4), 5, solvent_folder="sol")
    assert new.kwargs == dict(n_populations=2, q2_range=(1e-4, 2e-4), exponent=5, solvent_folder="sol")


# ----------------------------------------------------------------------------- LaplaceAnalyzer
def _make_analyzer():
    from ade_dls.engine.laplace_analyzer import LaplaceAnalyzer
    an = object.__new__(LaplaceAnalyzer)
    angles = np.array([30.0, 60.0, 90.0, 120.0])
    q2 = (angles / 100.0) ** 2 * 1e-3
    names = [f"f{i}.asc" for i in range(4)]
    an.regularized_data = pd.DataFrame({
        "filename": names, "angle [°]": angles, "q^2": q2,
        "normalized_area_percent_pop1": [100.0] * 4})
    an.regularized_final_results = pd.DataFrame({"Fit": ["pop1"], "Rh [nm]": [50.0]})
    an.df_intensity = pd.DataFrame({
        "filename": names, "angle [°]": angles,
        "MeanCR_corr [kHz]": [100.0, 120.0, 150.0, 200.0]})
    return an


def test_run_sls_analysis_uses_subtracted_intensity_and_keeps_original(monkeypatch):
    import ade_dls.analysis.sls as sls
    calls = []

    def fake(df, folder):
        calls.append(folder)
        out = df.copy()
        out["MeanCR_corr [kHz]"] = out["MeanCR_corr [kHz]"] / 2
        return out

    monkeypatch.setattr(sls, "subtract_solvent", fake)
    an = _make_analyzer()
    original = an.df_intensity.copy()
    an.run_sls_analysis(1, solvent_folder="solv")
    assert calls == ["solv"]
    pd.testing.assert_frame_equal(an.df_intensity, original)
    np.testing.assert_allclose(an.sls_data["MeanCR_corr [kHz]"].to_numpy(),
                               original["MeanCR_corr [kHz]"].to_numpy() / 2, rtol=1e-6)


def test_run_sls_analysis_without_solvent_folder_never_calls_subtract(monkeypatch):
    import ade_dls.analysis.sls as sls

    def boom(*a, **k):
        raise AssertionError("subtract_solvent must not be called")

    monkeypatch.setattr(sls, "subtract_solvent", boom)
    an = _make_analyzer()
    an.run_sls_analysis(1)
    np.testing.assert_allclose(an.sls_data["MeanCR_corr [kHz]"].to_numpy(),
                               [100.0, 120.0, 150.0, 200.0], rtol=1e-6)


def test_run_sls_analysis_lets_value_error_from_subtract_solvent_propagate(monkeypatch):
    import ade_dls.analysis.sls as sls

    def bad(df, folder):
        raise ValueError("no solvent files")

    monkeypatch.setattr(sls, "subtract_solvent", bad)
    with pytest.raises(ValueError, match="no solvent files"):
        _make_analyzer().run_sls_analysis(1, solvent_folder="x")


# ----------------------------------------------------------------------------- dialogs (source)
@pytest.mark.parametrize("name", ["regularized_results_dialog.py", "laplace_postfit_dialog.py"])
def test_dialogs_offer_solvent_checkbox_folder_field_and_pass_folder(name):
    import re
    src = (DIALOGS / name).read_text(encoding="utf-8")
    assert re.search(r"QCheckBox\([^)]*solvent", src, re.IGNORECASE), "no solvent QCheckBox"
    assert "QLineEdit" in src
    assert re.search(r"solvent\w*\s*=\s*QLineEdit\(", src, re.IGNORECASE), "no solvent QLineEdit"
    assert "solvent_folder=" in src
