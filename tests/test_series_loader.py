"""Tests for ade_dls.engine.loading.SeriesLoader (Qt-free loader extracted from the GUI worker)
and its thin Qt wrapper ade_dls.gui.core.data_loader.DataLoadWorker.

Real data: tests/data/alv_mono (6 measurement units: angles 30/60/90 deg, 2 repeats each).
"""
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ade_dls.core.parsers import detect_parser
from ade_dls.engine.loading import DataLoadError, SeriesLoader

ALV = Path(__file__).resolve().parent / "data" / "alv_mono"

# header values of the fixture files (wavelength [nm], refractive index) and angles per unit
WAVELENGTH, N_REF = 632.8, 1.332
ANGLES = [30.0, 30.0, 60.0, 60.0, 90.0, 90.0]

pytestmark = pytest.mark.skipif(not ALV.exists(), reason="tests/data/alv_mono missing")


@pytest.fixture
def parser():
    p = detect_parser(str(ALV))
    assert p is not None
    return p


def _load(parser, **kw):
    prog, steps = [], []
    loader = SeriesLoader(str(ALV), parser, on_progress=lambda *a: prog.append(a),
                          on_step=lambda n, d: steps.append(n), **kw)
    return loader, loader.run(), prog, steps


# ----------------------------------------------------------------------------- oracles
def test_oracle_result_keys_and_counts(parser):
    _, r, _, _ = _load(parser)
    assert set(r) == {"data_folder", "instrument", "files", "num_files", "basedata", "countrates",
                      "correlations", "cumulant_mode", "cumulants", "errors", "total_files",
                      "successful_files"}
    assert r["instrument"] == "ALV" and r["cumulant_mode"] == "frequency"
    assert (r["num_files"], r["total_files"], r["successful_files"]) == (6, 6, 6)
    assert r["errors"] == []
    assert len(r["countrates"]) == len(r["correlations"]) == len(r["cumulants"]) == 6


def test_oracle_q_vectors_from_formula(parser):
    """q = 4 pi n / lambda * sin(theta/2); q^2 = q*q (angles are the fixture header values)."""
    _, r, _, _ = _load(parser)
    b = r["basedata"]
    assert b["angle [°]"].tolist() == ANGLES
    expected_q = [4 * math.pi * N_REF / WAVELENGTH * math.sin(math.radians(a) / 2) for a in ANGLES]
    np.testing.assert_allclose(b["q"].to_numpy(float), expected_q, rtol=1e-12)
    np.testing.assert_allclose(b["q^2"].to_numpy(float), np.square(expected_q), rtol=1e-12)
    # literal cross-check at 30 deg: 4*pi*1.332/632.8 * sin(15 deg) = 6.846e-3 nm^-1
    assert b["q"].iloc[0] == pytest.approx(6.8461e-3, rel=1e-4)


def test_file_order_is_sorted_independent_of_filesystem_order(parser, monkeypatch):
    """glob.glob returns files in filesystem order (arbitrary on Linux, sorted on NTFS). The series
    order (rows of basedata, hence all downstream tables) must not depend on it (CI finding, Linux)."""
    import glob as _glob
    real = _glob.glob

    def reversed_glob(*args, **kwargs):
        return list(reversed(real(*args, **kwargs)))

    monkeypatch.setattr("ade_dls.core.parsers.alv_parser.glob.glob", reversed_glob)
    files = parser.get_file_list(str(ALV))
    assert files == sorted(files)
    _, r, _, _ = _load(parser)
    assert r["basedata"]["angle [°]"].tolist() == ANGLES


def test_oracle_labels_line_up_across_tables(parser):
    _, r, _, _ = _load(parser)
    labels = list(r["basedata"]["filename"])
    assert len(set(labels)) == 6
    assert set(r["correlations"]) == set(r["countrates"]) == set(r["cumulants"]) == set(labels)
    assert list(r["basedata"].index) == [1, 2, 3, 4, 5, 6]


def test_oracle_equals_direct_parser_calls(parser):
    _, r, _, _ = _load(parser)
    for f in r["files"]:
        label = parser.get_label(f)
        direct = parser.extract_basedata(f).iloc[0]
        row = r["basedata"][r["basedata"]["filename"] == label].iloc[0]
        for c in ("angle [°]", "temperature [K]", "wavelength [nm]", "refractive_index"):
            assert row[c] == pytest.approx(direct[c], rel=1e-12)
        pd.testing.assert_frame_equal(r["correlations"][label], parser.extract_correlations(f))


def test_progress_and_step_sequence(parser):
    _, _, prog, steps = _load(parser)
    assert steps == ["find_files", "basedata", "countrates", "correlations", "cumulants"]
    stages = [p[0] for p in prog]
    assert stages == sorted(stages) and stages[0] == 0 and prog[-1] == (5, 5, "Calculating scattering vectors...")
    assert all(p[1] == 5 for p in prog)


# ----------------------------------------------------------------------------- metamorphic
def test_meta_optional_parts_can_be_skipped(parser):
    _, r, _, steps = _load(parser, load_countrates=False, load_correlations=False,
                           load_cumulants=False)
    assert "countrates" not in r and "correlations" not in r
    assert r["cumulants"] == {}
    assert steps == ["find_files", "basedata"]
    _, full, _, _ = _load(parser)
    pd.testing.assert_frame_equal(r["basedata"], full["basedata"])


def test_meta_two_runs_are_identical(parser):
    _, a, _, _ = _load(parser)
    _, b, _, _ = _load(parser)
    pd.testing.assert_frame_equal(a["basedata"], b["basedata"])
    assert a["files"] == b["files"]


# ----------------------------------------------------------------------------- edge / errors
def test_edge_empty_folder_raises_with_gui_message(parser, tmp_path):
    with pytest.raises(DataLoadError, match="No data files found in"):
        SeriesLoader(str(tmp_path), parser).run()


def test_edge_cancel_before_run_returns_none_and_emits_nothing(parser):
    steps = []
    loader = SeriesLoader(str(ALV), parser, on_step=lambda n, d: steps.append(n))
    loader.cancel()
    assert loader.run() is None
    assert steps == ["find_files"]          # cancelled right after the file search


def test_edge_single_bad_file_is_isolated_and_reported(parser, monkeypatch):
    files = parser.get_file_list(str(ALV))
    bad = files[2]
    orig = parser.extract_basedata

    def flaky(f):
        if f == bad:
            raise RuntimeError("corrupt")
        return orig(f)

    monkeypatch.setattr(parser, "extract_basedata", flaky)
    _, r, _, _ = _load(parser)
    assert len(r["basedata"]) == 5
    assert r["successful_files"] == 5 and r["total_files"] == 6
    assert len(r["errors"]) == 1
    err = r["errors"][0]
    assert err["step"] == "basedata" and "RuntimeError: corrupt" in err["error"]


def test_edge_no_correlations_at_all_raises(parser, monkeypatch):
    monkeypatch.setattr(parser, "extract_correlations", lambda f: None)
    with pytest.raises(DataLoadError, match="Failed to extract correlation data"):
        SeriesLoader(str(ALV), parser).run()


def test_edge_unexpected_exception_is_wrapped(parser, monkeypatch):
    def boom(folder):
        raise RuntimeError("boom")

    monkeypatch.setattr(parser, "get_file_list", boom)
    with pytest.raises(DataLoadError, match="Error loading data: boom"):
        SeriesLoader(str(ALV), parser).run()


def test_edge_missing_metadata_columns_keep_frame_without_q(parser):
    loader = SeriesLoader(str(ALV), parser)
    df = pd.DataFrame({"angle [°]": [30.0], "filename": ["x"]})
    out = loader._calculate_q_vectors(df)
    assert "q" not in out.columns and out.equals(df)


def test_edge_rows_with_missing_metadata_are_dropped(parser):
    loader = SeriesLoader(str(ALV), parser)
    df = pd.DataFrame({"angle [°]": [30.0, np.nan], "wavelength [nm]": [632.8, 632.8],
                       "refractive_index": [1.332, 1.332], "filename": ["a", "b"]})
    out = loader._calculate_q_vectors(df)
    assert list(out["filename"]) == ["a"] and out["q^2"].notna().all()


# ----------------------------------------------------------------------------- Qt wrapper
def test_gui_worker_is_a_thin_wrapper_around_the_series_loader(parser):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ade_dls.gui.core.data_loader import DataLoadWorker

    worker = DataLoadWorker(str(ALV), parser)
    finished, errors, steps = [], [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.step_complete.connect(lambda n, d: steps.append(n))
    worker.run()                      # synchronous call; no event loop needed for same-thread slots
    assert errors == [] and len(finished) == 1
    assert finished[0]["num_files"] == 6 and "q^2" in finished[0]["basedata"].columns
    assert steps == ["find_files", "basedata", "countrates", "correlations", "cumulants"]


def test_gui_worker_reports_loader_errors_through_the_error_signal(parser, tmp_path):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ade_dls.gui.core.data_loader import DataLoadWorker

    worker = DataLoadWorker(str(tmp_path), parser)
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.run()
    assert finished == [] and len(errors) == 1 and errors[0].startswith("No data files found in")


def test_gui_worker_cancel_emits_neither_finished_nor_error(parser):
    pytest.importorskip("PyQt5")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from ade_dls.gui.core.data_loader import DataLoadWorker

    worker = DataLoadWorker(str(ALV), parser)
    finished, errors = [], []
    worker.finished.connect(finished.append)
    worker.error.connect(errors.append)
    worker.cancel()
    worker.run()
    assert finished == [] and errors == [] and worker.is_cancelled
