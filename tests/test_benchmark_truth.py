"""Ground-truth checks of the headless chain on small simulated series (noise-free monomodal, 6 files)."""
import matplotlib
matplotlib.use("Agg")

import pytest

from ade_dls.benchmark import runner
from ade_dls.benchmark.evaluate import evaluate

SETTINGS = {"b_fit_limits": [0, 0.0002], "b_q2_range": [0, 0.001], "b_through_origin": False,
            "d_gap_threshold": 2.0}


@pytest.fixture(scope="module")
def mono(tmp_path_factory):
    folder = tmp_path_factory.mktemp("mono_nf")
    return runner.simulate_series("mono_pdi0.05", 0.0, "white", 0, str(folder), angles=(30, 90, 150), runs=2)


def test_simulate_series_loads_requested_files(mono):
    data, ds = mono
    assert len(data["correlations"]) == 6
    assert ds.truth["rh_pops"] == [50.0]


def test_method_B_recovers_true_radius_within_3_percent(mono):
    data, ds = mono
    res = runner.run_method(data, "B", SETTINGS, weighted=False)
    ev = evaluate(res, ds.truth)
    assert res["n_found"] == 1 and ev["success"] is True
    assert abs(ev["rel_err_z"]) < 0.03


@pytest.mark.slow
def test_method_D_finds_one_population_within_5_percent(mono):
    data, ds = mono
    res = runner.run_method(data, "D", SETTINGS, weighted=False)
    ev = evaluate(res, ds.truth)
    assert res["n_found"] == 1 and ev["count_ok"] is True
    assert abs(ev["rel_err"][0]) < 0.05


def test_run_method_rejects_unknown_method(mono):
    with pytest.raises(ValueError):
        runner.run_method(mono[0], "X", SETTINGS, weighted=False)
