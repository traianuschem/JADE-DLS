"""JADE 3.1: corrected_intensity (core.preprocessing) and subtract_solvent (analysis.sls)."""
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ade_dls.analysis import sls  # noqa: E402
from ade_dls.core import preprocessing as pp  # noqa: E402
from ade_dls.utils.intensity import build_intensity_dataframe  # noqa: E402

DATA = Path(__file__).parent / "data" / "alv_mono"


def _ci(cr0, cr1, md, ang):
    return (cr0 + cr1) / 2 / (md * 1e-3) * np.sin(np.radians(ang))


# ---------------------------------------------------------------- corrected_intensity
def test_oracle_corrected_intensity_hand_value():
    df = pd.DataFrame({"meancr0 [kHz]": [100.0, 50.0], "meancr1 [kHz]": [300.0, 70.0],
                       "monitordiode [cps]": [2000.0, 1000.0], "angle [°]": [90.0, 30.0]})
    out = pp.corrected_intensity(df)
    assert isinstance(out, pd.Series)
    # (100+300)/2 / 2 * 1 = 100 ; (50+70)/2 / 1 * 0.5 = 30
    assert out.tolist() == pytest.approx([100.0, 30.0], rel=1e-12)


def test_edge_corrected_intensity_nan_propagates_no_fillna():
    df = pd.DataFrame({"meancr0 [kHz]": [np.nan, 1.0, 1.0, 1.0],
                       "meancr1 [kHz]": [1.0, np.nan, 1.0, 1.0],
                       "monitordiode [cps]": [1.0, 1.0, np.nan, 1.0],
                       "angle [°]": [90.0, 90.0, 90.0, np.nan]})
    assert pp.corrected_intensity(df).isna().all()


def test_oracle_build_intensity_dataframe_identical_to_corrected_intensity():
    files = sorted(p for p in DATA.glob("*.ASC") if "averaged" not in p.name)
    df = build_intensity_dataframe(files)
    assert len(df) == 6
    np.testing.assert_allclose(df["MeanCR_corr [kHz]"].to_numpy(),
                               pp.corrected_intensity(df).to_numpy(), rtol=1e-12)
    # hand value for first file (30 deg)
    assert df["MeanCR_corr [kHz]"].iloc[0] == pytest.approx(
        _ci(315.769, 316.18078, 6646.31, 30.0), rel=1e-9)


# ---------------------------------------------------------------- subtract_solvent
@pytest.fixture
def solvent_dir(tmp_path):
    d = tmp_path / "solvent"
    d.mkdir()
    for p in DATA.glob("*.ASC"):          # includes one *_averaged.ASC that must be ignored
        shutil.copy(p, d / p.name)
    return d


def _I_solv():
    return {
        30.0: np.mean([_ci(315.769, 316.18078, 6646.31, 30.0), _ci(326.30319, 326.32947, 6624.1, 30.0)]),
        60.0: np.mean([_ci(589.76869, 594.835, 40864.03, 60.0), _ci(590.22563, 595.94275, 40848.1, 60.0)]),
        90.0: np.mean([_ci(318.60613, 324.07741, 40878.31, 90.0), _ci(325.28947, 330.38209, 40883.6, 90.0)]),
    }


def _sample(angles=(30.0, 60.0, 90.0), vals=(500.0, 800.0, 600.0)):
    return pd.DataFrame({"filename": [f"s{i}" for i in range(len(angles))],
                         "angle [°]": list(angles), "MeanCR_corr [kHz]": list(vals)})


def test_oracle_subtract_solvent_per_angle_mean_and_columns(solvent_dir):
    df = _sample()
    snapshot = df.copy(deep=True)
    out = sls.subtract_solvent(df, str(solvent_dir))
    pd.testing.assert_frame_equal(df, snapshot)          # input untouched
    assert out is not df
    I = _I_solv()
    exp_solv = [I[30.0], I[60.0], I[90.0]]
    assert out["MeanCR_solvent [kHz]"].tolist() == pytest.approx(exp_solv, rel=1e-9)
    assert out["MeanCR_corr_sample [kHz]"].tolist() == [500.0, 800.0, 600.0]
    assert out["MeanCR_corr [kHz]"].tolist() == pytest.approx(
        [500.0 - exp_solv[0], 800.0 - exp_solv[1], 600.0 - exp_solv[2]], rel=1e-9)


def test_oracle_subtract_solvent_ignores_averaged_file(solvent_dir):
    """The averaged file would change the 30 deg mean if it were included."""
    out = sls.subtract_solvent(_sample(), str(solvent_dir))
    I30_with_avg = np.mean([_ci(315.769, 316.18078, 6646.31, 30.0),
                            _ci(326.30319, 326.32947, 6624.1, 30.0),
                            _ci(323.00878, 323.98244, 6637.74, 30.0)])
    assert out["MeanCR_solvent [kHz]"].iloc[0] != pytest.approx(I30_with_avg, rel=1e-9)
    assert out["MeanCR_solvent [kHz]"].iloc[0] == pytest.approx(_I_solv()[30.0], rel=1e-9)


def test_oracle_subtract_solvent_custom_intensity_col(solvent_dir):
    df = _sample().rename(columns={"MeanCR_corr [kHz]": "X"})
    out = sls.subtract_solvent(df, str(solvent_dir), intensity_col="X")
    assert out["MeanCR_corr_sample [kHz]"].tolist() == [500.0, 800.0, 600.0]
    assert out["X"].iloc[0] == pytest.approx(500.0 - _I_solv()[30.0], rel=1e-9)


def test_edge_subtract_solvent_missing_angle_warns_and_uses_zero(solvent_dir, capsys):
    out = sls.subtract_solvent(_sample(angles=(30.0, 45.0), vals=(500.0, 400.0)), str(solvent_dir))
    assert any(ln.startswith("WARNING") for ln in capsys.readouterr().out.splitlines())
    assert out["MeanCR_solvent [kHz]"].iloc[1] == 0.0
    assert out["MeanCR_corr [kHz]"].iloc[1] == 400.0


def test_edge_subtract_solvent_angle_rounding(solvent_dir):
    out = sls.subtract_solvent(_sample(angles=(30.001,), vals=(500.0,)), str(solvent_dir))
    assert out["MeanCR_solvent [kHz]"].iloc[0] == pytest.approx(_I_solv()[30.0], rel=1e-9)


def test_edge_subtract_solvent_nonpositive_net_warns(solvent_dir, capsys):
    out = sls.subtract_solvent(_sample(vals=(500.0, 800.0, 1.0)), str(solvent_dir))
    assert out["MeanCR_corr [kHz]"].iloc[2] < 0
    assert any(ln.startswith("WARNING") for ln in capsys.readouterr().out.splitlines())


def test_edge_subtract_solvent_no_files_raises(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No solvent files found in"):
        sls.subtract_solvent(_sample(), str(empty))


def test_edge_subtract_solvent_only_averaged_raises(tmp_path):
    d = tmp_path / "avg"
    d.mkdir()
    shutil.copy(DATA / "080622_5_0053_averaged.ASC", d / "x_averaged.ASC")
    with pytest.raises(ValueError, match="No solvent files found in"):
        sls.subtract_solvent(_sample(), str(d))


# ---------------------------------------------------------------- Guinier with mean <= 0
def _guinier_frame():
    q2 = {20.0: 1e-4, 30.0: 2e-4, 40.0: 3e-4, 50.0: 4e-4, 60.0: 5e-4}
    rg2 = 100.0  # nm^2 -> Rg = 10 nm
    rows = []
    for ang, q in q2.items():
        val = -5.0 if ang == 60.0 else 100.0 * np.exp(-rg2 * (q * 100) / 3)
        for _ in range(2):
            rows.append({"angle [°]": ang, "q^2": q * 100, "MeanCR_corr [kHz]": val,
                         "I_pop1 [kHz]": val})
    return pd.DataFrame(rows)


def test_char_guinier_total_skips_nonpositive_mean_angle_documents_current_behavior():
    """documents current behavior: angles with mean <= 0 are dropped, no crash."""
    res = sls.compute_guinier_total(_guinier_frame())
    assert res is not None
    assert list(res["grouped"].index) == [20.0, 30.0, 40.0, 50.0]
    assert res["Rg [nm]"] == pytest.approx(10.0, rel=1e-6)


def test_char_guinier_extrapolation_skips_nonpositive_mean_angle_documents_current_behavior():
    """documents current behavior: angles with mean <= 0 are dropped, no crash."""
    res = sls.compute_guinier_extrapolation(_guinier_frame(), 1)
    assert list(res[1]["grouped"].index) == [20.0, 30.0, 40.0, 50.0]
    assert res[1]["Rg [nm]"] == pytest.approx(10.0, rel=1e-6)
