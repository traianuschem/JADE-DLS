"""Golden subset of the synthetic-data study (reference: unmodified JADE-DLS modules) and ground truth.

Fast tests check the integrity of ``tests/data/synthetic_golden`` and the evaluation rule. The slow tests
regenerate the simulated samples (``ade_dls.benchmark``), run ADE-DLS headless with the settings of the study
and compare with the stored JADE results (tolerances: deterministic kernels 1e-6, optimizers 1e-3).
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ade_dls import headless
from ade_dls.benchmark import runner
from ade_dls.benchmark import synthetic as syn
from ade_dls.benchmark.evaluate import evaluate

GOLDEN_DIR = Path(__file__).resolve().parent / "data" / "synthetic_golden"
GOLDEN = json.loads((GOLDEN_DIR / "golden.json").read_text(encoding="utf-8"))["cells"]
CELLS = sorted(GOLDEN)
METHODS = ("B", "C2", "C4", "D", "NNLS", "Reg")
MONO_CELLS = [k for k in CELLS if GOLDEN[k]["truth"]["n_pop"] == 1]

DET = 1e-6   # deterministic kernels
OPT = 1e-3   # optimizer results


# ----------------------------------------------------------------------------- integrity (fast)
def test_golden_file_matches_sha256():
    expected = (GOLDEN_DIR / "golden.json.sha256").read_text().split()[0]
    assert hashlib.sha256((GOLDEN_DIR / "golden.json").read_bytes()).hexdigest() == expected


def test_golden_has_ten_cells_with_all_methods():
    assert len(CELLS) == 10
    for key in CELLS:
        assert set(GOLDEN[key]["methods"]) == set(METHODS), key


def test_golden_covers_mono_bi_and_trimodal_with_and_without_noise():
    n_pops = {GOLDEN[k]["truth"]["n_pop"] for k in CELLS}
    levels = {GOLDEN[k]["noise_level"] for k in CELLS}
    assert n_pops == {1, 2, 3} and 0.0 in levels and {1.0, 2.0} <= levels


@pytest.mark.parametrize("key", CELLS)
def test_scenario_truth_equals_golden_truth(key):
    cell = GOLDEN[key]
    truth = syn.Scenario(cell["geometry"], runner.all_geometries()[cell["geometry"]],
                         cell["noise_level"], cell["noise_mode"]).truth()
    ref = cell["truth"]
    assert truth["n_pop"] == ref["n_pop"]
    for name in ("rh_pops", "fractions"):
        assert truth[name] == pytest.approx(ref[name], rel=1e-12)
    for name in ("rh_z", "pdi_mix", "rh_z_reg_origin", "rh_z_reg_free"):
        assert truth[name] == pytest.approx(ref[name], rel=1e-12)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("key", CELLS)
def test_evaluate_reproduces_stored_evaluation(key, method):
    cell = GOLDEN[key]
    st = cell["methods"][method]
    ev = evaluate({"method": method, **st["result"]}, cell["truth"])
    ref = st["eval"]
    for name in ("count_ok", "success", "spurious"):
        assert ev[name] == ref[name]
    assert ev["rel_err"] == pytest.approx(ref["rel_err"], rel=1e-9, abs=1e-12)
    if ref["rel_err_z"] is not None:
        assert ev["rel_err_z"] == pytest.approx(ref["rel_err_z"], rel=1e-9, abs=1e-12)


def test_evaluate_tolerance_boundary_and_counts():
    truth = {"n_pop": 2, "rh_pops": [20.0, 80.0], "rh_z": 40.0}
    ok = evaluate({"method": "D", "n_found": 2, "rh": [24.0, 64.0]}, truth)          # +20 % / -20 %
    assert ok["success"] is True and ok["count_ok"] is True and ok["spurious"] is False
    out = evaluate({"method": "D", "n_found": 2, "rh": [24.1, 80.0]}, truth)          # +20.5 %
    assert out["success"] is False and out["count_ok"] is True
    more = evaluate({"method": "D", "n_found": 3, "rh": [10.0, 40.0, 80.0]}, truth)
    assert more["spurious"] is True and more["success"] is False and more["rel_err"] == []
    fewer = evaluate({"method": "NNLS", "n_found": 1, "rh": [40.0]}, truth)
    assert fewer["count_ok"] is False and fewer["spurious"] is False
    assert evaluate({"method": "D", "n_found": None, "rh": []}, truth)["success"] is None
    cum = evaluate({"method": "B", "n_found": 1, "rh": [44.0]}, {**truth, "n_pop": 1, "rh_pops": [40.0]})
    assert cum["rel_err_z"] == pytest.approx(0.1)


# ----------------------------------------------------------------------------- ADE vs stored JADE results (slow)
@pytest.fixture(scope="module")
def series(tmp_path_factory):
    """Simulated and loaded series per golden cell, built once."""
    cache = {}

    def get(key):
        if key not in cache:
            cell = GOLDEN[key]
            folder = tmp_path_factory.mktemp(key.replace(".", "_"))
            cache[key] = runner.simulate_series(cell["geometry"], cell["noise_level"], cell["noise_mode"],
                                                cell["rep"], str(folder))
        return cache[key]
    return get


_RUNS = {}   # (cell, method) -> result of the full 39-file run, shared by the tests of this module


def _run(series, key, method, data=None):
    st = GOLDEN[key]["methods"][method]
    if data is not None:
        return runner.run_method(data, method, st["settings"], st["weighted"]), st
    if (key, method) not in _RUNS:
        _RUNS[(key, method)] = runner.run_method(series(key)[0], method, st["settings"], st["weighted"])
    return _RUNS[(key, method)], st


@pytest.mark.slow
@pytest.mark.parametrize("key", CELLS)
def test_simulated_series_has_39_files_and_truth(series, key):
    data, ds = series(key)
    assert len(data["correlations"]) == 39
    assert ds.truth["rh_pops"] == pytest.approx(GOLDEN[key]["truth"]["rh_pops"], rel=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("key", CELLS)
def test_method_B_matches_stored_jade_result(series, key):
    res, st = _run(series, key, "B")
    assert res["n_found"] == st["result"]["n_found"] == 1
    assert res["rh"] == pytest.approx(st["result"]["rh"], rel=DET)


@pytest.mark.slow
@pytest.mark.parametrize("key", [k for k in MONO_CELLS if "pdi0.1__s1" in k or "pdi0.2__s2" in k])
@pytest.mark.parametrize("method", ["C2", "C4"])
def test_method_C_matches_stored_jade_result(series, key, method):
    res, st = _run(series, key, method)
    assert res["rh"] == pytest.approx(st["result"]["rh"], rel=OPT)
    assert res["pdi"] == pytest.approx(st["result"]["pdi"], rel=OPT, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("key", CELLS)
def test_method_D_gap2_matches_stored_jade_result(series, key):
    """Method D at gap_threshold 2.0 (the value of the reference runs). Per-file mode selection is numerically
    sensitive (library versions): the per-file tolerances apply to files with equal mode count."""
    res, st = _run(series, key, "D")
    ref = st["result"]
    assert st["settings"]["d_gap_threshold"] == 2.0
    assert res["n_found"] == ref["n_found"]
    tab = pd.DataFrame(ref["table"]).merge(res["table"], on="filename", suffixes=("_s", "_a"))
    assert len(tab) == 39
    equal = tab["n_modes_s"] == tab["n_modes_a"]
    assert equal.mean() >= 0.9
    eq = tab[equal]
    gm = np.abs(eq["gamma_mean_a"].to_numpy(float) / eq["gamma_mean_s"].to_numpy(float) - 1)
    assert (gm <= OPT).mean() >= 0.9 and gm.max() <= 5e-2          # observed: max 1.1e-2 (one file)
    bt = np.abs(eq["beta_a"].to_numpy(float) / eq["beta_s"].to_numpy(float) - 1)
    assert bt.max() <= 1e-2
    tol = 3e-3 if key in MONO_CELLS else 3e-2                       # observed: mono <= 7.4e-4, other <= 1.4e-2 (20/80 sample)
    assert res["rh"] == pytest.approx(ref["rh"], rel=tol)


@pytest.mark.slow
@pytest.mark.parametrize("method", ["B", "D"])
@pytest.mark.parametrize("key", CELLS)
def test_ground_truth_verdict_equals_stored_verdict(series, key, method):
    """ADE-DLS and the JADE modules reach the same verdict against the ground truth (population count and
    Rh within +-20 %), also for the samples where both methods fail (e.g. close bimodal sizes for B)."""
    res, st = _run(series, key, method)
    ev = evaluate(res, GOLDEN[key]["truth"])
    for name in ("count_ok", "success", "spurious"):
        assert ev[name] == st["eval"][name], name


# Per-file NNLS / Regularized fits on a subset of files (a full 39-file fit takes minutes)
LAPLACE_KEY = "mono_pdi0.1__s1_correlated_r000"
MATCH_FILES = ["_a030_0001", "_a090_0001", "_a090_0002", "_a150_0003"]
NNLS_DIFFER_FILES = ["_a030_0002", "_a100_0002", "_a140_0002"]


def _subset_data(series, key, suffixes):
    data, _ = series(key)
    drop = [d for d in data["correlations"] if not any(d.endswith(s + ".ASC") for s in suffixes)]
    sub = dict(data)
    headless.exclude_files(sub, drop)
    return sub


def _assert_per_file_peaks_equal(res, st, n_files):
    ref = pd.DataFrame(st["result"]["table"])
    m = ref.merge(res["table"], on="filename", suffixes=("_s", "_a"))
    assert len(m) == n_files
    cols = [c for c in ref.columns if c.startswith("tau_")]
    assert cols
    for c in cols:
        # a column only one of the two tables has is not suffixed by the merge
        a = m[c + "_s"].to_numpy(float) if c + "_s" in m else (m[c].to_numpy(float) if c in m else np.full(len(m), np.nan))
        b = m[c + "_a"].to_numpy(float) if c + "_a" in m else np.full(len(m), np.nan)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{c}: peak present in only one of the results"
        ok = ~np.isnan(a)
        assert b[ok] == pytest.approx(a[ok], rel=OPT), c


@pytest.mark.slow
@pytest.mark.parametrize("method", ["NNLS", "Reg"])
def test_laplace_per_file_peaks_match_stored_jade_table(series, method):
    sub = _subset_data(series, LAPLACE_KEY, MATCH_FILES)
    res, st = _run(series, LAPLACE_KEY, method, data=sub)
    _assert_per_file_peaks_equal(res, st, len(MATCH_FILES))


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="FINDING (synthetic benchmark): the per-file NNLS peak lists of "
                   "regularized_optimized.nnls_optimized differ from the JADE-module result for ~13-20 % of the "
                   "files (weak/edge peaks of the ill-posed problem); strong peaks agree. Not changed without approval.")
def test_nnls_per_file_peak_lists_documents_deviation_from_jade(series):
    sub = _subset_data(series, LAPLACE_KEY, NNLS_DIFFER_FILES)
    res, st = _run(series, LAPLACE_KEY, "NNLS", data=sub)
    _assert_per_file_peaks_equal(res, st, len(NNLS_DIFFER_FILES))
