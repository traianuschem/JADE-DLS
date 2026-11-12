"""JADE 3.1: compute_sls_data follows the population columns normalized_area_percent_pop{i}
(clustering result), falling back to the per-peak columns normalized_area_percent_{i}."""
import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from ade_dls.analysis import sls  # noqa: E402

TOL = dict(rel=1e-9, abs=1e-12)


def _intensity():
    return pd.DataFrame({
        "filename": ["A", "B"],
        "angle [°]": [30.0, 60.0],
        "MeanCR_corr [kHz]": [200.0, 100.0],
    })


def test_oracle_pop_columns_win_when_peak_number_differs_from_population():
    """File A: peak1 -> pop2, peak2 -> pop1. Result must follow the _pop columns."""
    reg = pd.DataFrame({
        "filename": ["A", "B"],
        "normalized_area_percent_1": [70.0, 40.0],       # per-peak (must be ignored)
        "normalized_area_percent_2": [30.0, 60.0],
        "normalized_area_percent_pop1": [30.0, 60.0],    # population columns
        "normalized_area_percent_pop2": [70.0, 40.0],
    })
    out = sls.compute_sls_data(reg, _intensity(), 2)
    assert out["I_pop1 [kHz]"].tolist() == pytest.approx([200 * 0.30, 100 * 0.60], **TOL)
    assert out["I_pop2 [kHz]"].tolist() == pytest.approx([200 * 0.70, 100 * 0.40], **TOL)


def test_oracle_fallback_to_peak_columns_pass_through_mode():
    reg = pd.DataFrame({
        "filename": ["A", "B"],
        "normalized_area_percent_1": [70.0, 40.0],
        "normalized_area_percent_2": [30.0, 60.0],
    })
    out = sls.compute_sls_data(reg, _intensity(), 2)
    assert out["I_pop1 [kHz]"].tolist() == pytest.approx([140.0, 40.0], **TOL)
    assert out["I_pop2 [kHz]"].tolist() == pytest.approx([60.0, 60.0], **TOL)


def test_edge_mixed_pop_and_fallback_per_population():
    reg = pd.DataFrame({
        "filename": ["A", "B"],
        "normalized_area_percent_pop1": [10.0, 20.0],
        "normalized_area_percent_2": [30.0, 60.0],
    })
    out = sls.compute_sls_data(reg, _intensity(), 2)
    assert out["I_pop1 [kHz]"].tolist() == pytest.approx([20.0, 20.0], **TOL)
    assert out["I_pop2 [kHz]"].tolist() == pytest.approx([60.0, 60.0], **TOL)


def test_edge_no_area_column_warns_and_skips_population(capsys):
    reg = pd.DataFrame({"filename": ["A", "B"], "normalized_area_percent_pop1": [50.0, 50.0]})
    out = sls.compute_sls_data(reg, _intensity(), 2)
    assert "Warning" in capsys.readouterr().out
    assert "I_pop1 [kHz]" in out.columns
    assert "I_pop2 [kHz]" not in out.columns
