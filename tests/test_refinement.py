"""Tests for ade_dls.engine.refinement (Method C / D post-fit bookkeeping, no Qt) and the headless
refinement wrappers. The analyzer's own numerics are tested in test_cumulants*.py; here a stub
analyzer isolates the assembling logic, plus one run on tests/data/alv_mono for Method B."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ade_dls.engine import refinement

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"
D_COL = "D [m²/s]"


def _stub(files=("a", "b", "c", "d"), orig=True, recompute_error=None):
    calls = []

    def recompute(included, q_range):
        calls.append((list(included), q_range))
        if recompute_error:
            raise recompute_error
        return pd.DataFrame({"Rh [nm]": [10.0], D_COL: [2e-11], "R_squared": [0.99]})

    an = SimpleNamespace(
        method_c_fit=pd.DataFrame({"filename": [f for f in files for _ in range(2)]}),
        recompute_method_c_diffusion=recompute,
        method_c_recompute_n_points=7, method_c_recompute_n_files_in_range=3,
        method_c_regression_stats={"slope": 1.5},
    )
    if orig:
        an.method_c_results = pd.DataFrame({"Rh [nm]": [12.0], D_COL: [1.8e-11], "R_squared": [0.95]})
    return an, calls


# ----------------------------------------------------------------------------- refine_method_c
def test_refine_method_c_passes_included_files_and_q_range_to_the_analyzer():
    an, calls = _stub()
    refinement.refine_method_c(an, (1e-4, 2e-4), ["b"])
    assert calls == [(["a", "c", "d"], (1e-4, 2e-4))]


def test_refine_method_c_adds_metadata_columns_and_original_results():
    an, _ = _stub()
    result, stats = refinement.refine_method_c(an, (1e-4, 2e-4), ["b", "c"])
    row = result.iloc[0]
    assert (row["N files (total)"], row["N files (included)"], row["N files (excluded)"]) == (4, 2, 2)
    assert row["q-range applied"] == "1.000e-04 – 2.000e-04 nm⁻²"
    assert (row["Original Rh [nm]"], row["Original D [m²/s]"], row["Original R²"]) == (12.0, 1.8e-11, 0.95)
    assert row["Rh [nm]"] == 10.0                         # recomputed value is untouched


def test_refine_method_c_refinement_info_dict():
    an, _ = _stub()
    _, stats = refinement.refine_method_c(an, None, ["a"])
    assert stats["slope"] == 1.5                          # analyzer stats are kept
    info = stats["refinement_info"]
    assert info == {
        "n_total": 4, "n_included": 3, "n_excluded": 1, "n_data_points": 7, "n_files_in_range": 3,
        "excluded_files": ["a"], "q_range": None, "q_range_str": "Full range",
        "original_rh": 12.0, "original_d": 1.8e-11, "original_r2": 0.95,
    }
    assert an.method_c_regression_stats == {"slope": 1.5}  # not mutated


def test_refine_method_c_without_original_results_leaves_comparison_columns_out():
    an, _ = _stub(orig=False)
    result, stats = refinement.refine_method_c(an)
    assert "Original Rh [nm]" not in result.columns
    info = stats["refinement_info"]
    assert info["original_rh"] is None and info["original_d"] is None and info["excluded_files"] == []


@pytest.mark.parametrize("excluded", [["a", "b", "c"], ["a", "b", "c", "d"]])
def test_refine_method_c_needs_at_least_two_remaining_files(excluded, capsys):
    an, calls = _stub()
    assert refinement.refine_method_c(an, None, excluded) is None
    assert calls == [] and "Too few files" in capsys.readouterr().out


def test_refine_method_c_exactly_two_files_is_allowed():
    an, calls = _stub()
    assert refinement.refine_method_c(an, None, ["a", "b"]) is not None


def test_refine_method_c_without_method_c_fit_returns_none(capsys):
    assert refinement.refine_method_c(SimpleNamespace()) is None
    assert refinement.refine_method_c(SimpleNamespace(method_c_fit=None)) is None
    assert "No Method C data" in capsys.readouterr().out


def test_refine_method_c_swallows_analyzer_errors_and_reports(capsys):
    an, _ = _stub(recompute_error=ValueError("boom"))
    assert refinement.refine_method_c(an) is None
    assert "recompute_method_c_diffusion failed: boom" in capsys.readouterr().out


# ----------------------------------------------------------------------------- Method D exclusion
def test_exclude_files_from_method_d_drops_rows_and_resets_index():
    an = SimpleNamespace(method_d_data=pd.DataFrame({"filename": ["a", "b", "a", "c"], "x": [1, 2, 3, 4]}))
    assert refinement.exclude_files_from_method_d(an, ["a"]) == 2
    assert list(an.method_d_data["filename"]) == ["b", "c"] and list(an.method_d_data.index) == [0, 1]


@pytest.mark.parametrize("excluded", [None, []])
def test_exclude_files_from_method_d_nothing_excluded_is_noop(excluded):
    df = pd.DataFrame({"filename": ["a", "b"]})
    an = SimpleNamespace(method_d_data=df)
    assert refinement.exclude_files_from_method_d(an, excluded) == 0
    assert an.method_d_data is df


def test_exclude_files_from_method_d_without_method_d_data_is_noop():
    assert refinement.exclude_files_from_method_d(SimpleNamespace(), ["a"]) == 0
    assert refinement.exclude_files_from_method_d(SimpleNamespace(method_d_data=None), ["a"]) == 0


# ----------------------------------------------------------------------------- headless
def test_headless_refine_cumulant_c_equals_engine_function():
    from ade_dls import headless
    an, _ = _stub()
    r1, s1 = headless.refine_cumulant_c(an, (1e-4, 2e-4), ["a"])
    an2, _ = _stub()
    r2, s2 = refinement.refine_method_c(an2, (1e-4, 2e-4), ["a"])
    pd.testing.assert_frame_equal(r1, r2)
    assert s1 == s2


def test_headless_refine_cumulant_d_excludes_before_reclustering():
    from ade_dls import headless
    seen = {}

    def refine(clustering_params=None, mode_params=None, combined_q_range=None):
        seen.update(files=list(an.method_d_data["filename"]), clustering=clustering_params,
                    modes=mode_params, q=combined_q_range)
        return "result"

    an = SimpleNamespace(method_d_data=pd.DataFrame({"filename": ["a", "b", "c"]}), refine_method_d=refine)
    out = headless.refine_cumulant_d(an, {"distance_threshold": 2.0}, {}, (1e-4, 2e-4), ["b"])
    assert out == "result"
    assert seen == {"files": ["a", "c"], "clustering": {"distance_threshold": 2.0}, "modes": None,
                    "q": (1e-4, 2e-4)}


@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_refine_cumulant_b_excluding_a_fit_changes_the_regression():
    from ade_dls import headless
    data = headless.load_series(str(ALV))
    an, full = headless.run_cumulant_b(data, (1e-6, 1e-2))
    labels = list(data["basedata"]["filename"])
    refined = headless.refine_cumulant_b(an, excluded_files=labels[:2])
    assert refined is not None and np.isfinite(refined.iloc[0][D_COL])
    assert refined.iloc[0][D_COL] != pytest.approx(full.iloc[0][D_COL], rel=1e-9, abs=0)
