"""Tests for ade_dls.engine.filtering (Qt-free data operations behind the filter dialogs) and
headless.exclude_files. Inputs are synthetic dictionaries built here; one test uses tests/data/alv_mono."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ade_dls.engine import filtering

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"


def _data(labels=("a", "b", "c", "d")):
    return {
        "basedata": pd.DataFrame({"filename": list(labels), "angle [°]": [30.0, 60.0, 90.0, 120.0][:len(labels)]}),
        "countrates": {l: np.arange(3) for l in labels},
        "correlations": {l: pd.DataFrame({"t": [1.0], "g2": [1.0]}) for l in labels},
        "num_files": len(labels),
    }


# ----------------------------------------------------------------------------- helpers
def test_excluded_files_keeps_order_of_original():
    assert filtering.excluded_files({"c": 1, "a": 2, "b": 3}, {"a": 0}) == ["c", "b"]
    assert filtering.excluded_files([], ["x"]) == []
    assert filtering.excluded_files(["a"], []) == ["a"]


def test_restrict_basedata_filters_rows_and_updates_count():
    d = _data()
    filtering.restrict_basedata(d, ["b", "d"])
    assert list(d["basedata"]["filename"]) == ["b", "d"] and d["num_files"] == 2


def test_sync_basedata_to_correlations_is_noop_without_both_entries():
    d = _data()
    del d["correlations"]
    filtering.sync_basedata_to_correlations(d)
    assert len(d["basedata"]) == 4 and d["num_files"] == 4
    d = _data()
    d["correlations"].pop("a")
    filtering.sync_basedata_to_correlations(d)
    assert list(d["basedata"]["filename"]) == ["b", "c", "d"] and d["num_files"] == 3


def test_correlations_to_filter_follows_countrate_step():
    d = _data()
    fd = dict(d)
    fd["countrates"] = {k: v for k, v in d["countrates"].items() if k != "b"}
    assert list(filtering.correlations_to_filter(d, fd)) == ["a", "c", "d"]
    del fd["countrates"]
    assert list(filtering.correlations_to_filter(d, fd)) == ["a", "b", "c", "d"]


# ----------------------------------------------------------------------------- apply_*
def test_apply_countrate_filter_reduces_all_three_tables():
    d = _data()
    kept = {k: v for k, v in d["countrates"].items() if k in ("a", "c")}
    excluded = filtering.apply_countrate_filter(d, kept)
    assert excluded == ["b", "d"]
    assert list(d["countrates"]) == list(d["correlations"]) == ["a", "c"]
    assert list(d["basedata"]["filename"]) == ["a", "c"] and d["num_files"] == 2


def test_apply_countrate_filter_without_optional_tables():
    d = {"countrates": {"a": 1, "b": 2}}
    assert filtering.apply_countrate_filter(d, {"b": 2}) == ["a"]
    assert d == {"countrates": {"b": 2}}


def test_apply_correlation_filter_leaves_countrates_untouched():
    d = _data()
    kept = {k: v for k, v in d["correlations"].items() if k != "c"}
    assert filtering.apply_correlation_filter(d, kept) == ["c"]
    assert list(d["correlations"]) == ["a", "b", "d"]
    assert list(d["countrates"]) == ["a", "b", "c", "d"]
    assert list(d["basedata"]["filename"]) == ["a", "b", "d"] and d["num_files"] == 3


def test_exclude_files_removes_from_every_table_and_reports_present_labels():
    d = _data()
    removed = filtering.exclude_files(d, ["b", "zzz", "d"])
    assert sorted(removed) == ["b", "d"]
    assert list(d["countrates"]) == list(d["correlations"]) == ["a", "c"]
    assert list(d["basedata"]["filename"]) == ["a", "c"] and d["num_files"] == 2


def test_exclude_files_nothing_to_remove_changes_nothing():
    d = _data()
    assert filtering.exclude_files(d, []) == [] and d["num_files"] == 4


def test_exclude_files_is_equivalent_to_the_dialog_path_for_the_same_selection():
    """GUI path: countrate filter then correlation filter; headless path: exclude_files."""
    a, b = _data(), _data()
    keep = ("a", "d")
    filtering.apply_countrate_filter(a, {k: v for k, v in a["countrates"].items() if k in keep})
    filtering.apply_correlation_filter(a, {k: v for k, v in a["correlations"].items() if k in keep})
    filtering.exclude_files(b, ["b", "c"])
    assert list(a["countrates"]) == list(b["countrates"])
    assert list(a["correlations"]) == list(b["correlations"])
    pd.testing.assert_frame_equal(a["basedata"], b["basedata"])
    assert a["num_files"] == b["num_files"]


# ----------------------------------------------------------------------------- summary text
def test_noise_correction_lines_use_documented_defaults_for_percentages():
    assert filtering.noise_correction_lines(None) == []
    assert filtering.noise_correction_lines({"baseline_correction": True}) == [
        "  • Baseline correction: last 5 %"]
    assert filtering.noise_correction_lines(
        {"baseline_correction": True, "baseline_pct": 7, "intercept_correction": True}
    ) == ["  • Baseline correction: last 7 %", "  • Intercept correction: first 2 %"]


def test_filtering_summary_text_variants():
    d = _data()
    d["num_files"] = 3
    base = ("Filtering complete!\n\nOriginal files: 4\nExcluded: 1\nFinal dataset: 3 files\n\n"
            "Base data: 4 entries\nCorrelations: 4 datasets\nCount rates: 4 datasets\n\n")
    assert filtering.filtering_summary(4, d, None) == base + "Noise correction: not applied"
    assert filtering.filtering_summary(4, d, {}) == base + "Noise correction: not applied"
    assert filtering.filtering_summary(4, d, {"x": 1}) == base + "Noise correction: active (no corrections selected)"
    assert filtering.filtering_summary(4, d, {"intercept_correction": True, "intercept_pct": 3}) == (
        base + "Noise correction active:\n  • Intercept correction: first 3 %")


# ----------------------------------------------------------------------------- headless
@pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")
def test_headless_exclude_files_recomputes_processed_correlations_and_results():
    from ade_dls import headless

    data = headless.load_series(str(ALV))
    labels = list(data["basedata"]["filename"])
    _, before = headless.run_cumulant_a(data)
    removed = headless.exclude_files(data, labels[:2])
    assert sorted(removed) == sorted(labels[:2])
    assert data["num_files"] == 4 and len(data["processed_correlations"]) == 4
    assert set(data["processed_correlations"]) == set(labels[2:])
    assert not {"df_basedata", "c", "delta_c"} & set(data)
    _, after = headless.run_cumulant_a(data)
    assert after.iloc[0]["D [m²/s]"] != pytest.approx(before.iloc[0]["D [m²/s]"], rel=1e-9, abs=0)
