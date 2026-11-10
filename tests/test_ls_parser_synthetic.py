"""CI-capable LS Instruments parser tests on a synthetic folder (no external data)."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from ade_dls.core.parsers.ls_instruments_parser import LSInstrumentsParser

ANGLE = 30.0
N_REF = 1.3326
Q = 4 * np.pi * N_REF / 561e-9 * np.sin(np.radians(ANGLE) / 2)  # 1/m
SUMMARY = {
    "Identifier": 1, "Duration [s]": 12.0, "Temperature [K]": 298.05,
    "Average count rate A [kHz]": 88.25, "Average count rate B [kHz]": 199.5,
    "Scattering angle [deg]": ANGLE, "Scattering vector [1/nm]": Q,
    "Solvent": "Water", "Solvent refractive index": N_REF,
    "Solvent viscosity [mPas]": 0.8925,
}
HDR = "This file contains the complete raw data.\n"


def _write_rep(rep, seed, drop=()):
    rep.mkdir(parents=True)
    row = {k: v for k, v in SUMMARY.items() if k not in drop}
    pd.DataFrame([row]).to_csv(rep / "Summary.csv", index=False, encoding="utf-8-sig")
    rng = np.random.default_rng(seed)
    tau = np.logspace(-6, 1, 150)
    gamma = 2000.0  # 1/s
    g2 = 1.0 + 0.95 * np.exp(-2 * gamma * tau) + rng.normal(0, 2e-3, tau.size)
    with open(rep / "Correlation Function.csv", "w", encoding="utf-8-sig") as f:
        f.write(HDR + "Correlation function\nLag time [s],Value\n")
        for t, g in zip(tau, g2):
            f.write(f"{t:.8E},{g:.8E}\n")
    t = np.arange(0, 200) * 0.0524288
    with open(rep / "Count Trace.csv", "w", encoding="utf-8-sig") as f:
        f.write("Count Rate History\nCorrelation Type: Pseudo Cross Correlation\n"
                "Time [s], CR CHA [kHz], CR CHB [kHz]\n")
        for ti in t:
            f.write(f"{ti:.7f},{88 + rng.normal(0, 5):.4f},{199 + rng.normal(0, 8):.4f}\n")


@pytest.fixture
def folder(tmp_path):
    _write_rep(tmp_path / "Measurement 1" / "Repetition1", 1)
    _write_rep(tmp_path / "Measurement 1" / "Repetition2", 2)
    return tmp_path


def test_detect_and_file_list(folder):
    assert LSInstrumentsParser.detect(str(folder)) == 1.0
    assert len(LSInstrumentsParser().get_file_list(str(folder))) == 2


def test_extract_basedata_values_exact(folder):
    p = LSInstrumentsParser()
    rep = p.get_file_list(str(folder))[0]
    bd = p.extract_basedata(rep)
    assert bd["duration [s]"].iloc[0] == 12.0
    assert bd["meancr0 [kHz]"].iloc[0] == 88.25
    assert bd["meancr1 [kHz]"].iloc[0] == 199.5
    assert np.isnan(bd["monitordiode [cps]"].iloc[0])
    assert bd["wavelength [nm]"].iloc[0] == pytest.approx(561.0, rel=1e-6)


@pytest.mark.parametrize("col,out", [
    ("Duration [s]", "duration [s]"),
    ("Average count rate A [kHz]", "meancr0 [kHz]"),
    ("Average count rate B [kHz]", "meancr1 [kHz]"),
])
def test_missing_summary_column_gives_nan_only_for_that_column(tmp_path, col, out):
    rep = tmp_path / "Measurement 1" / "Repetition1"
    _write_rep(rep, 1, drop=(col,))
    bd = LSInstrumentsParser().extract_basedata(str(rep))
    expected = {"duration [s]": 12.0, "meancr0 [kHz]": 88.25, "meancr1 [kHz]": 199.5}
    for c, v in expected.items():
        if c == out:
            assert np.isnan(bd[c].iloc[0])
        else:
            assert bd[c].iloc[0] == v


def _load_and_preprocess(folder):
    from ade_dls.headless import load_series
    from ade_dls.engine.series import preprocess_series
    data = load_series(str(folder), load_countrates=False, load_cumulants=False,
                       preprocess=False)
    preprocess_series(data)
    return data


def test_end_to_end_weighting_available(folder):
    data = _load_and_preprocess(folder)
    assert data["weighting_available"] is True, data.get("weighting_unavailable_reason")
    assert len(data["weights"]) == 2
    for name, w in data["weights"].items():
        assert np.mean(np.asarray(w, dtype=float)) == pytest.approx(1.0, rel=1e-6), name


def test_end_to_end_weighting_unavailable_without_countrate_column(tmp_path):
    _write_rep(tmp_path / "Measurement 1" / "Repetition1", 1)
    _write_rep(tmp_path / "Measurement 1" / "Repetition2", 2,
               drop=("Average count rate A [kHz]",))
    data = _load_and_preprocess(tmp_path)
    assert data["weighting_available"] is False
    reason = data.get("weighting_unavailable_reason")
    assert isinstance(reason, str) and reason.strip()
