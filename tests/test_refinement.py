"""Tests for ade_dls.engine.refinement (Method C / D post-fit bookkeeping, no Qt) and the headless
refinement wrappers. The analyzer's own numerics are tested in test_cumulants*.py; here a stub
analyzer isolates the assembling logic, plus one run on tests/data/alv_mono for Method B."""
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
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


# ----------------------------------------------------------------------------- Laplace (NNLS / Regularized)
class _LaplaceStub:
    """Records the calls refine_laplace makes; tables are plain DataFrames."""

    def __init__(self, with_snapshot=True):
        table = pd.DataFrame({"filename": ["a", "b", "c", "d"], "q^2": [1.0, 2.0, 3.0, 4.0]})
        table.index = table.index + 1
        self.nnls_data = table.copy()
        self.regularized_data = table.copy()
        self._nnls_data_snapshot = table.copy() if with_snapshot else None
        self._regularized_data_snapshot = table.copy() if with_snapshot else None
        self.nnls_final_results = "nnls-final"
        self.regularized_final_results = "reg-final"
        self.calls = []

    def calculate_nnls_diffusion_coefficients(self, **kw):
        self.calls.append(("nnls_diff", kw, list(self.nnls_data["filename"])))

    def _calculate_nnls_final_results(self, **kw):
        self.calls.append(("nnls_final", kw))

    def calculate_regularized_diffusion_coefficients(self, **kw):
        self.calls.append(("reg_diff", kw, list(self.regularized_data["filename"])))

    def _calculate_regularized_final_results(self, **kw):
        self.calls.append(("reg_final", kw))


@pytest.mark.parametrize("args,expected", [
    ({}, False),
    ({"q_range": None, "excluded_files": [], "clustering_params": {}, "per_pop_ranges": {}}, False),
    ({"q_range": (1e-4, 2e-4)}, True),
    ({"excluded_files": ["a"]}, True),
    ({"clustering_params": {"use_clustering": False}}, True),
    ({"per_pop_ranges": {1: (1.0, 2.0)}}, True),
])
def test_laplace_refinement_has_changes(args, expected):
    assert refinement.laplace_refinement_has_changes(**args) is expected


def test_refine_laplace_nnls_excludes_files_and_recomputes_with_filtered_clustering_keys():
    an = _LaplaceStub()
    out = refinement.refine_laplace(
        an, "NNLS", (1.0, 3.0), ["b", "d"],
        {"use_clustering": False, "clustering_strategy": "simple", "distance_threshold": 1.5,
         "min_abundance": 0.4}, {1: (1.0, 2.0)}, fit_through_origin=True)
    assert out == "nnls-final"
    assert list(an.nnls_data["filename"]) == ["a", "c"] and list(an.nnls_data.index) == [1, 2]
    assert an.calls == [
        ("nnls_diff", {"x_range": (1.0, 3.0), "per_pop_ranges": {1: (1.0, 2.0)}, "use_clustering": False,
                       "clustering_strategy": "simple", "distance_threshold": 1.5}, ["a", "c"]),
        ("nnls_final", {"append_mode": True, "label_suffix": " (Post-Refined)"}),
    ]                                                   # NNLS gets neither min_abundance nor fit_through_origin
    assert list(an.regularized_data["filename"]) == ["a", "b", "c", "d"]    # other method untouched


def test_refine_laplace_regularized_passes_min_abundance_and_fit_through_origin():
    an = _LaplaceStub()
    out = refinement.refine_laplace(an, "Regularized", None, None,
                                    {"min_abundance": 0.4, "use_clustering": True, "ignored": 1},
                                    {}, fit_through_origin=True)
    assert out == "reg-final"
    assert an.calls[0] == ("reg_diff", {"x_range": None, "per_pop_ranges": None,
                                        "fit_through_origin": True, "min_abundance": 0.4,
                                        "use_clustering": True}, ["a", "b", "c", "d"])
    assert an.calls[1] == ("reg_final", {"append_mode": True, "label_suffix": " (Post-Refined)"})


def test_refine_laplace_restores_the_snapshot_so_refinements_do_not_accumulate():
    an = _LaplaceStub()
    refinement.refine_laplace(an, "Regularized", excluded_files=["a", "b"])
    assert list(an.regularized_data["filename"]) == ["c", "d"]
    refinement.refine_laplace(an, "Regularized", excluded_files=["d"])
    assert list(an.regularized_data["filename"]) == ["a", "b", "c"]
    refinement.refine_laplace(an, "Regularized", q_range=(1.0, 2.0))
    assert list(an.regularized_data["filename"]) == ["a", "b", "c", "d"]
    assert an._regularized_data_snapshot is not an.regularized_data      # snapshot itself stays pristine


def test_refine_laplace_without_snapshot_works_on_the_current_table():
    an = _LaplaceStub(with_snapshot=False)
    refinement.refine_laplace(an, "NNLS", excluded_files=["a"])
    assert list(an.nnls_data["filename"]) == ["b", "c", "d"]


def test_refine_laplace_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown Laplace method"):
        refinement.refine_laplace(_LaplaceStub(), "Method X")


def test_analysis_view_delegates_laplace_refinement_to_the_engine():
    """Structure check (no Qt needed): the view must not re-implement snapshot / exclusion logic."""
    src = (Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "widgets" / "analysis_view.py"
           ).read_text(encoding="utf-8")
    assert "refine_laplace(the_analyzer, method_name" in src
    assert "laplace_refinement_has_changes(" in src
    assert "_nnls_data_snapshot" not in src and "_regularized_data_snapshot" not in src
    assert "calculate_nnls_diffusion_coefficients(" not in src
    assert "calculate_regularized_diffusion_coefficients(" not in src


@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_refine_regularized_excludes_distributions_and_restores_on_next_call():
    from ade_dls import headless
    data = headless.load_series(str(ALV))
    an, _ = headless.run_regularized(data, {"decay_times": np.logspace(-8, 1, 26)})
    names = list(an.regularized_data["filename"])
    assert len(names) == 6
    headless.refine_regularized(an, excluded_files=names[:2])
    assert list(an.regularized_data["filename"]) == names[2:]
    final = headless.refine_regularized(an, q_range=None, per_pop_ranges=None,
                                        clustering_params={"use_clustering": True})
    assert list(an.regularized_data["filename"]) == names          # restored from the snapshot
    assert final is an.regularized_final_results
    assert any("(Post-Refined)" in str(x) for x in final["Fit"])


@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_refine_nnls_marks_results_as_post_refined():
    from ade_dls import headless
    data = headless.load_series(str(ALV))
    an, _ = headless.run_nnls(data, {"decay_times": np.logspace(-8, 1, 26)})
    names = list(an.nnls_data["filename"])
    final = headless.refine_nnls(an, excluded_files=names[:1])
    assert len(an.nnls_data) == 5 and final is an.nnls_final_results
    assert any("(Post-Refined)" in str(x) for x in final["Fit"])


# ----------------------------------------------------------------------------- Method A / B bookkeeping
def test_refine_method_a_returns_result_and_refinement_info():
    calls = []
    an = SimpleNamespace(run_method_a=lambda q_range=None: calls.append(q_range) or "df-a")
    result, stats = refinement.refine_method_a(an, (1e-4, 2e-4))
    assert result == "df-a" and calls == [(1e-4, 2e-4)]
    assert stats == {"refinement_info": {
        "q_range_str": "1.000e-04 – 2.000e-04 nm⁻²", "q_range": (1e-4, 2e-4),
        "excluded_files": [], "n_total": None, "n_included": None, "n_excluded": 0}}
    assert refinement.refine_method_a(an, None)[1]["refinement_info"]["q_range_str"] == "Full range"


def test_refine_method_a_returns_none_when_the_analyzer_returns_none():
    assert refinement.refine_method_a(SimpleNamespace(run_method_a=lambda q_range=None: None), (1, 2)) is None


def test_refine_method_b_takes_counts_from_the_analyzer_and_uses_ascii_q_range_text():
    seen = {}

    def refine(q_range=None, excluded_files=None):
        seen.update(q=q_range, ex=excluded_files)
        return "df-b"

    an = SimpleNamespace(refine_method_b=refine,
                         _method_b_refinement_info={"n_total": 6, "n_included": 4, "n_excluded": 2})
    result, stats = refinement.refine_method_b(an, (1e-4, 2e-4), ["a", "b"])
    assert result == "df-b" and seen == {"q": (1e-4, 2e-4), "ex": ["a", "b"]}
    assert stats["refinement_info"] == {
        "q_range_str": "1.000e-04 - 2.000e-04 nm-2", "q_range": (1e-4, 2e-4), "excluded_files": ["a", "b"],
        "n_total": 6, "n_included": 4, "n_excluded": 2}


def test_refine_method_b_without_exclusions_passes_none_and_tolerates_missing_info():
    seen = {}
    an = SimpleNamespace(refine_method_b=lambda q_range=None, excluded_files=None: seen.update(ex=excluded_files) or "df")
    _, stats = refinement.refine_method_b(an, None, [])
    assert seen["ex"] is None
    assert stats["refinement_info"] == {"q_range_str": "Full range", "q_range": None, "excluded_files": [],
                                        "n_total": None, "n_included": None, "n_excluded": 0}
    assert refinement.refine_method_b(SimpleNamespace(refine_method_b=lambda **kw: None)) is None


def test_headless_refine_cumulant_a_equals_engine_function():
    from ade_dls import headless
    an = SimpleNamespace(run_method_a=lambda q_range=None: "df-a")
    assert headless.refine_cumulant_a(an, (1.0, 2.0)) == refinement.refine_method_a(an, (1.0, 2.0))


def test_analysis_view_delegates_method_a_b_bookkeeping_to_the_engine():
    src = (Path(__file__).resolve().parents[1] / "ade_dls" / "gui" / "widgets" / "analysis_view.py"
           ).read_text(encoding="utf-8")
    assert "refine_method_a(self.cumulant_analyzer" in src and "refine_method_b(self.cumulant_analyzer" in src
    assert "_method_b_refinement_info" not in src and "cumulant_analyzer.run_method_a" not in src
