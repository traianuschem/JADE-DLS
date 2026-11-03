# -*- coding: utf-8 -*-
"""Tests for ade_dls/analysis/cumulants.py (spec: _dev/notes/test_spec_cumulants.md).

Categories: test_*_oracle/meta/edge/char (IDs from the spec in the test name),
characterization tests end with ``_documents_current_behavior`` (documents
current behavior, not desired behavior); known findings are
xfail(strict=True, raises=..., reason="FINDING ...").
"""
import copy
import inspect
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

from ade_dls.analysis import cumulants as cu  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore")

TOL = dict(rel=1e-6, abs=1e-12)
PTOL = dict(abs=5e-4)       # p-values read from formatted statsmodels summary
PTOL2 = dict(abs=1.1e-3)    # two rounded p-values compared with each other


@pytest.fixture(autouse=True)
def _close_figs(monkeypatch):
    # tight_layout on the 20x5 inch 3-panel figures dominates runtime; layout is irrelevant here
    monkeypatch.setattr(plt, "tight_layout", lambda *a, **k: None)
    plt.close("all")
    yield
    plt.close("all")


# ----------------------------------------------------------------------------
# helpers (numpy only, independent of statsmodels / production code)
# ----------------------------------------------------------------------------
def ols_hand(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    xb, yb = x.mean(), y.mean()
    sxx = np.sum((x - xb) ** 2)
    sxy = np.sum((x - xb) * (y - yb))
    b = sxy / sxx
    a = yb - b * xb
    e = y - a - b * x
    ssr = np.sum(e ** 2)
    s2 = ssr / (n - 2)
    return dict(n=n, xb=xb, yb=yb, sxx=sxx, sxy=sxy, b=b, a=a, e=e, ssr=ssr, s2=s2,
                se_b=math.sqrt(s2 / sxx), se_a=math.sqrt(s2 * (1.0 / n + xb ** 2 / sxx)),
                R2=1 - ssr / np.sum((y - yb) ** 2))


def origin_hand(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    b0 = np.sum(x * y) / np.sum(x * x)
    e = y - b0 * x
    s2 = np.sum(e ** 2) / max(n - 1, 1)
    sst = np.sum((y - y.mean()) ** 2)
    return dict(b0=b0, e=e, s2=s2, se=math.sqrt(s2 / np.sum(x * x)),
                R2c=(1 - np.sum(e ** 2) / sst) if sst > 0 else float("nan"),
                R2u=1 - np.sum(e ** 2) / np.sum(y ** 2))


def moments(e):
    e = np.asarray(e, float)
    d = e - e.mean()
    m2, m3, m4 = np.mean(d ** 2), np.mean(d ** 3), np.mean(d ** 4)
    return dict(skew=m3 / m2 ** 1.5, kurt=m4 / m2 ** 2 - 3)


def jb_p(e):
    m = moments(e)
    jb = len(e) / 6.0 * (m["skew"] ** 2 + m["kurt"] ** 2 / 4.0)
    return math.exp(-jb / 2.0)


def omni_p_sym(e):
    """D'Agostino K2 p-value for exactly symmetric residuals (z_skew = 0 by construction):
    p = exp(-z_kurt**2 / 2), chi2(2) survival of K2 = z_kurt**2.
    (stats.normaltest(r) itself is not usable as oracle: for skew == 0 up to float noise
    its skewtest z is noise-dominated, giving 0.5475 instead of 0.9407.)"""
    z = stats.kurtosistest(np.asarray(e, float)).statistic
    return math.exp(-z ** 2 / 2.0)


def write_asc(tmp_path, lines, encoding="cp1252", newline="\r\n", name="s.asc"):
    p = tmp_path / name
    p.write_bytes((newline.join(lines) + newline).encode(encoding))
    return p


def corr_df(t, g2m1, weight=None):
    d = {"t [s]": np.asarray(t, float), "g(2)-1": np.asarray(g2m1, float)}
    if weight is not None:
        d["weight"] = np.asarray(weight, float)
    return pd.DataFrame(d)


def mono(t, beta, G):
    return beta * np.exp(-2.0 * G * np.asarray(t, float))


def cumser(t, beta, G, mu2):
    t = np.asarray(t, float)
    return beta * np.exp(2.0 * (-G * t + mu2 * t ** 2 / 2.0))


# ----------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------
Q_LIN = np.array([1, 2, 3, 4, 5], float)
G_LIN = np.array([2.1, 3.9, 6.2, 7.8, 10.0])
E_LIN = np.array([0.04, -0.13, 0.20, -0.17, 0.06])

QB_NORM = [0.002, 0.0045, 0.0075, 0.0115, 0.0195]
QB_LEPTO = [0.001, 0.003, 0.006, 0.010, 0.022]
Q_20 = np.arange(1, 21, dtype=float)


def block_r(qb):
    r = []
    for q in qb:
        r += [q, -q, -q, q]
    return np.array(r)


R_NORM = block_r(QB_NORM)
R_LEPTO = block_r(QB_LEPTO)


def d_rnd():
    rng = np.random.default_rng(20261103)
    q2 = np.linspace(1e-4, 6e-4, 12)
    return q2, 4e6 * q2 + 50 + rng.normal(0, 20, 12)


T_MONO = np.geomspace(1e-6, 1e-2, 60)
T_CUM = np.linspace(0, 2e-4, 21)


def d_mono():
    return corr_df(T_MONO, mono(T_MONO, 0.8, 1500.0))


def d_cum():
    return corr_df(T_CUM, cumser(T_CUM, 0.9, 1000.0, 2e5))


def d_w(weight=True):
    d = pd.DataFrame({"t [s]": [0.0, 1.0, 2.0], "g(2)_mod": [0.0, -1.0, -3.0],
                      "g(2)-1": [0.5] * 3})
    if weight:
        d["weight"] = [1.0, 1.0, 2.0]
    return d


def adc(q, g, **kw):
    df = pd.DataFrame({"q^2": np.asarray(q, float), "g": np.asarray(g, float)})
    kw.setdefault("show_plots", False)
    return cu.analyze_diffusion_coefficient(df, "q^2", "g", **kw)


def adf(df, cols, **kw):
    kw.setdefault("show_plots", False)
    return cu.analyze_diffusion_coefficient(df, "q^2", cols, **kw)


def pc(d, lim, **kw):
    return cu.plot_processed_correlations(d, lim, **kw)


NUMCOLS = ["R_squared", "slope", "slope_se", "intercept", "intercept_se", "Skewness",
           "Kurtosis", "N_points_fit", "X_range_min", "X_range_max",
           "q^2_coef", "q^2_se", "const_coef", "const_se"]
PCOLS = ["JB_p_value", "Omnibus_p_value"]
ADC_COLUMNS = ['Column', 'Method', 'R_squared', 'slope', 'slope_se', 'intercept', 'intercept_se',
               'Normality', 'Skewness', 'Kurtosis', 'JB_p_value', 'Omnibus_p_value',
               'N_points_fit', 'X_range_min', 'X_range_max', 'q^2_coef', 'q^2_se',
               'const_coef', 'const_se']
PC_COLUMNS = ['filename', 'Gamma', 'Gamma_error', 'Gamma_relative_error', 'a', 'weighted',
              'R_squared']
CUM_COLUMNS = ['1st order frequency [1/ms]', '2nd order frequency [1/ms]',
               '3rd order frequency [1/ms]', '2nd order frequency exp param [ms^2]',
               '3rd order frequency exp param [ms^2]',
               '3rd order frequency 3rd param [1/ms^3]']


def cmp_rows(r1, r2, factors=None, skip=(), cols=NUMCOLS):
    factors = factors or {}
    for c in cols:
        if c in skip:
            continue
        assert r1[c] == pytest.approx(r2[c] * factors.get(c, 1.0), rel=1e-6, abs=1e-12,
                                      nan_ok=True), c
    for c in PCOLS:
        if c in skip:
            continue
        assert r1[c] == pytest.approx(r2[c], nan_ok=True, **PTOL2), c
    if "Normality" not in skip:
        assert r1["Normality"] == r2["Normality"]


BOTH = pytest.mark.parametrize("origin", [False, True], ids=["free", "origin"])


# ----------------------------------------------------------------------------
# ASC text
# ----------------------------------------------------------------------------
HEAD = ["ALV-7004/USB", 'Samplename : \t"test"', 'Temperature [K] :\t     298.15000']
B1 = ['"Cumulant 1.Order"',
      '"FluctuationFreq. [1/ms]"\t   1.71578E+000',
      '"DiffCoefficient [µm²/s]"\t   7.17025E+000',
      '"Hydrodyn. Radius [nm]"\t   2.98695E+001']
B2 = ['"Cumulant 2.Order"',
      '"FluctuationFreq. [1/ms]"\t   1.80455E+000',
      '"DiffCoefficient [µm²/s]"\t   7.54121E+000',
      '"Hydrodyn. Radius [nm]"\t   2.83999E+001',
      '"Expansion Parameter µ2"\t   2.58360E-001']
B3 = ['"Cumulant 3.Order"',
      '"FluctuationFreq. [1/ms]"\t   1.83178E+000',
      '"DiffCoefficient [µm²/s]"\t   7.65500E+000',
      '"Hydrodyn. Radius [nm]"\t   2.79776E+001',
      '"Expansion Parameter µ2"\t   3.55120E-001',
      '"Expansion Parameter µ3"\t   8.62400E-002']
TAIL = ['"StandardDeviation"']
BASE = HEAD + B1 + B2 + B3 + TAIL
EXPECT = [1.71578, 1.80455, 1.83178, 0.25836, 0.35512, 0.08624]
CUM_PARAMS = ["1st", "2nd", "3rd", "2nd exp", "3rd exp", "3rd mu3"]


def check_expect(df, expect=EXPECT):
    assert df.shape == (1, 6)
    assert list(df.columns) == CUM_COLUMNS
    for c, v in zip(CUM_COLUMNS, expect):
        assert df[c].iloc[0] == pytest.approx(v, rel=1e-12)


# ============================================================================
# extract_cumulants
# ============================================================================
def test_extract_cumulants_XO01_base_text(tmp_path):
    df = cu.extract_cumulants(write_asc(tmp_path, BASE))
    check_expect(df)
    assert list(df.index) == [0]


@pytest.mark.parametrize("txt,val", [("+1.5e-3", 0.0015), (".5", 0.5), ("-2.0", -2.0),
                                     ("1.0E+2", 100.0)])
def test_extract_cumulants_XO02_number_formats(tmp_path, txt, val):
    lines = HEAD + [B1[0], '"F"\t' + txt] + B1[2:] + B2 + B3 + TAIL
    df = cu.extract_cumulants(write_asc(tmp_path, lines))
    assert df[CUM_COLUMNS[0]].iloc[0] == pytest.approx(val, rel=1e-12)
    assert df[CUM_COLUMNS[1]].iloc[0] == pytest.approx(1.80455, rel=1e-12)


def test_extract_cumulants_XO03_path_and_str_identical(tmp_path):
    p = write_asc(tmp_path, BASE)
    pd.testing.assert_frame_equal(cu.extract_cumulants(p), cu.extract_cumulants(str(p)))


@pytest.mark.parametrize("enc", ["cp1252", "utf-8", "latin-1"])
def test_extract_cumulants_XM01_encodings(tmp_path, enc):
    check_expect(cu.extract_cumulants(write_asc(tmp_path, BASE, encoding=enc)))


@pytest.mark.parametrize("nl", ["\r\n", "\n"])
def test_extract_cumulants_XM02_line_endings(tmp_path, nl):
    check_expect(cu.extract_cumulants(write_asc(tmp_path, BASE, newline=nl)))


def test_extract_cumulants_XM03_extra_header_lines(tmp_path):
    extra = ['"Correlation"', '  1.25000E-004\t  8.5E-001'] * 10
    check_expect(cu.extract_cumulants(write_asc(tmp_path, HEAD + extra + B1 + B2 + B3 + TAIL)))


def test_extract_cumulants_XM04_reverse_block_order(tmp_path):
    check_expect(cu.extract_cumulants(write_asc(tmp_path, HEAD + B3 + B2 + B1 + TAIL)))


def test_extract_cumulants_XM05_first_block_set_wins(tmp_path):
    second = [l.replace("1.71578", "9.99999").replace("1.80455", "8.88888")
              .replace("1.83178", "7.77777").replace("2.58360", "6.66666")
              .replace("3.55120", "5.55555") for l in B1 + B2 + B3]
    check_expect(cu.extract_cumulants(write_asc(tmp_path, BASE + second)))


def test_extract_cumulants_XM06_undefined_cp1252_byte_falls_back(tmp_path):
    lines = [HEAD[0] + "\x81"] + HEAD[1:] + B1 + B2 + B3 + TAIL
    check_expect(cu.extract_cumulants(write_asc(tmp_path, lines, encoding="latin-1")))


def test_extract_cumulants_XE01_missing_file(tmp_path):
    assert cu.extract_cumulants(tmp_path / "nope.asc") is None


def test_extract_cumulants_XE02_directory(tmp_path):
    assert cu.extract_cumulants(tmp_path) is None


def test_extract_cumulants_XE03_first_order_no_number(tmp_path):
    lines = HEAD + [B1[0], '"F"\tn/a'] + B1[2:] + B2 + B3 + TAIL
    assert cu.extract_cumulants(write_asc(tmp_path, lines)) is None


def test_extract_cumulants_XE04_second_order_exp_no_number(tmp_path):
    b2 = B2[:4] + ['"Expansion Parameter µ2"\t---']
    assert cu.extract_cumulants(write_asc(tmp_path, HEAD + B1 + b2 + B3 + TAIL)) is None


def test_extract_cumulants_XE05_no_third_block(tmp_path):
    df = cu.extract_cumulants(write_asc(tmp_path, HEAD + B1 + B2))
    for c, v in zip(CUM_COLUMNS[:2], EXPECT[:2]):
        assert df[c].iloc[0] == pytest.approx(v, rel=1e-12)
    assert df[CUM_COLUMNS[3]].iloc[0] == pytest.approx(0.25836, rel=1e-12)
    assert pd.isna(df[CUM_COLUMNS[2]].iloc[0]) and pd.isna(df[CUM_COLUMNS[4]].iloc[0])
    assert pd.isna(df[CUM_COLUMNS[5]].iloc[0])


def test_extract_cumulants_XE06_truncated_second_block(tmp_path):
    df = cu.extract_cumulants(write_asc(tmp_path, HEAD + B1 + B2[:3]))
    assert df[CUM_COLUMNS[1]].iloc[0] == pytest.approx(1.80455, rel=1e-12)
    for i in (2, 3, 4, 5):
        assert pd.isna(df[CUM_COLUMNS[i]].iloc[0])


def test_extract_cumulants_XE07_first_order_is_last_line(tmp_path):
    df = cu.extract_cumulants(write_asc(tmp_path, HEAD + B2 + B3 + [B1[0]]))
    assert pd.isna(df[CUM_COLUMNS[0]].iloc[0])
    for i in (1, 2, 3, 4, 5):
        assert df[CUM_COLUMNS[i]].iloc[0] == pytest.approx(EXPECT[i], rel=1e-12)


def test_extract_cumulants_XE08_single_encoding_fails(tmp_path):
    p = tmp_path / "bad.asc"
    p.write_bytes(b"abc\xff\xfe\n")
    assert cu.extract_cumulants(p, encodings=["utf-8"]) is None


def test_extract_cumulants_XE09_empty_encodings(tmp_path):
    assert cu.extract_cumulants(write_asc(tmp_path, BASE), encodings=[]) is None


def test_extract_cumulants_XC01_third_exp_stays_mu2_and_mu3_is_separate(tmp_path):
    """F-07 fix (D-11): '3rd order ... exp param' stays mu2 of the 3rd-order fit (PDI_3rd uses it);
    mu3 is exposed in its own column."""
    df = cu.extract_cumulants(write_asc(tmp_path, BASE))
    assert df[CUM_COLUMNS[4]].iloc[0] == pytest.approx(0.35512, rel=1e-12)
    assert df[CUM_COLUMNS[5]].iloc[0] == pytest.approx(0.08624, rel=1e-12)


def _real_block_lines():
    rep_ = {"2.58360E-001": "2.12840E+000", "3.55120E-001": "4.82810E+000",
            "8.62400E-002": "9.14520E+000"}
    out = []
    for l in BASE:
        for k, v in rep_.items():
            l = l.replace(k, v)
        out.append(l)
    return out


def test_extract_cumulants_XF05_mu3_is_read(tmp_path):
    """F-07 fix (D-11): mu3 (line i+5 of the 3rd-order block) is read."""
    df = cu.extract_cumulants(write_asc(tmp_path, _real_block_lines()))
    assert df[CUM_COLUMNS[5]].iloc[0] == pytest.approx(9.1452, rel=1e-12)


def test_extract_cumulants_XE10_missing_mu3_line_gives_nan_keeps_other_columns(tmp_path):
    """Optional mu3 must not discard a file that parsed before (3rd-order block ends at mu2)."""
    df = cu.extract_cumulants(write_asc(tmp_path, HEAD + B1 + B2 + B3[:5] + TAIL))
    assert df is not None
    assert pd.isna(df[CUM_COLUMNS[5]].iloc[0])
    for i in (0, 1, 2, 3, 4):
        assert df[CUM_COLUMNS[i]].iloc[0] == pytest.approx(EXPECT[i], rel=1e-12)


def test_calculate_skewness_3rd_oracle_mu3_over_mu2_pow_1p5(tmp_path):
    """Skewness_3rd = mu3 / mu2(3rd-order fit)^1.5 = 9.1452 / 4.8281^1.5 = 0.862 (real ASC block)."""
    df = cu.extract_cumulants(write_asc(tmp_path, _real_block_lines()))
    s = cu.calculate_skewness_3rd(df)
    assert s.iloc[0] == pytest.approx(9.1452 / 4.8281 ** 1.5, rel=1e-6)
    assert s.iloc[0] == pytest.approx(0.862, rel=1e-3)
    # the former (wrong) value mu2(3.)/mu2(2.)^1.5 = 1.555 must not come back
    assert s.iloc[0] != pytest.approx(1.555, rel=1e-2)


def test_calculate_skewness_3rd_per_row_and_nan_propagation():
    d = pd.DataFrame({CUM_COLUMNS[4]: [4.0, 1.0, np.nan],
                      CUM_COLUMNS[5]: [8.0, -0.5, 1.0]})
    s = cu.calculate_skewness_3rd(d)
    assert list(s.index) == [0, 1, 2]
    assert s.iloc[0] == pytest.approx(8.0 / 4.0 ** 1.5, rel=1e-12)   # = 1.0
    assert s.iloc[1] == pytest.approx(-0.5, rel=1e-12)               # sign preserved
    assert np.isnan(s.iloc[2])


def test_calculate_skewness_3rd_missing_mu3_column_gives_nan_not_error():
    d = pd.DataFrame({CUM_COLUMNS[4]: [4.0, 1.0]})
    s = cu.calculate_skewness_3rd(d)
    assert len(s) == 2 and s.isna().all()


def test_calculate_skewness_3rd_does_not_modify_input():
    d = pd.DataFrame({CUM_COLUMNS[4]: [4.0], CUM_COLUMNS[5]: [8.0]})
    before = d.copy()
    cu.calculate_skewness_3rd(d)
    pd.testing.assert_frame_equal(d, before)


@pytest.mark.parametrize("content", [b"", b"hello\n"], ids=["empty", "hello"])
def test_extract_cumulants_XC02_no_block_gives_all_none_frame_documents_current_behavior(
        tmp_path, content):
    """documents current behavior, not desired behavior (F-08): no None, 1x5 all-NA."""
    p = tmp_path / "e.asc"
    p.write_bytes(content)
    df = cu.extract_cumulants(p)
    assert df is not None and df.shape == (1, 6)
    assert list(df.columns) == CUM_COLUMNS
    assert df.isna().all().all()


@pytest.mark.parametrize("txt", ["2", "2E+000", "1,71578"])
def test_extract_cumulants_XC03_XC04_no_decimal_point_discards_file_documents_current_behavior(
        tmp_path, txt):
    """documents current behavior, not desired behavior (F-08)."""
    lines = HEAD + [B1[0], '"F"\t' + txt] + B1[2:] + B2 + B3 + TAIL
    assert cu.extract_cumulants(write_asc(tmp_path, lines)) is None


def test_extract_cumulants_XC05_signature_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    sig = inspect.signature(cu.extract_cumulants)
    assert list(sig.parameters) == ["file_path", "encodings"]
    assert sig.parameters["encodings"].default == ['Windows-1252', 'utf-8', 'latin1', 'ISO-8859-1']


def test_extract_cumulants_XC06_arbitrary_bytes_default_encodings_documents_current_behavior(
        tmp_path):
    """documents current behavior (F-09): latin-1 never fails, so values are found."""
    p = tmp_path / "b.asc"
    p.write_bytes(bytes(range(256)) + b"\n" + ("\r\n".join(BASE) + "\r\n").encode("latin-1"))
    check_expect(cu.extract_cumulants(p))


# ============================================================================
# calculate_g2_B
# ============================================================================
def test_calculate_g2_B_GO01_filter_and_log():
    d = pd.DataFrame({"t [s]": np.array([1, 2, 3, 4, 5, 6]) * 1e-6,
                      "g(2)-1": [1, np.exp(-2), 0, -0.1, np.exp(-4), np.nan]})
    r = cu.calculate_g2_B({"a": d})["a"]
    np.testing.assert_allclose(r["t [s]"], [1e-6, 2e-6, 5e-6], rtol=1e-12)
    np.testing.assert_allclose(r["g(2)_mod"], [0, -1, -2], atol=1e-12)
    assert r.index.equals(pd.RangeIndex(3))
    assert list(r.columns) == ["t [s]", "g(2)-1", "g(2)_mod"]


def test_calculate_g2_B_GO02_mono():
    r = cu.calculate_g2_B({"m": d_mono()})["m"]
    assert len(r) == 60
    np.testing.assert_allclose(r["g(2)_mod"], 0.5 * np.log(0.8) - 1500 * T_MONO, atol=1e-12)


def test_calculate_g2_B_GO03_cumulant_series():
    r = cu.calculate_g2_B({"c": d_cum()})["c"]
    np.testing.assert_allclose(r["g(2)_mod"], 0.5 * np.log(0.9) - 1000 * T_CUM + 1e5 * T_CUM ** 2,
                               atol=1e-12)


def test_calculate_g2_B_GM01_idempotent():
    d = pd.DataFrame({"t [s]": [1., 2, 3, 4], "g(2)-1": [0.5, -1, 0.2, 0]})
    once = cu.calculate_g2_B({"a": d})
    twice = cu.calculate_g2_B(once)
    pd.testing.assert_frame_equal(once["a"], twice["a"])


def test_calculate_g2_B_GM02_input_not_mutated():
    d = {"a": pd.DataFrame({"t [s]": [1., 2, 3], "g(2)-1": [0.5, -1, 0.2]})}
    cp = copy.deepcopy(d)
    cu.calculate_g2_B(d)
    assert d["a"].equals(cp["a"])


def test_calculate_g2_B_GM03_scaling_shifts_log():
    d = d_cum()
    d2 = d.copy()
    d2["g(2)-1"] *= 0.5
    a = cu.calculate_g2_B({"x": d})["x"]
    b = cu.calculate_g2_B({"x": d2})["x"]
    np.testing.assert_allclose(b["g(2)_mod"], a["g(2)_mod"] + 0.5 * np.log(0.5), atol=1e-12)
    np.testing.assert_allclose(b["t [s]"], a["t [s]"])


def test_calculate_g2_B_GM04_dict_order_and_values():
    ds = {k: pd.DataFrame({"t [s]": [1., 2], "g(2)-1": [v, v / 2]})
          for k, v in zip("abc", [0.9, 0.8, 0.7])}
    fw = cu.calculate_g2_B(ds)
    rv = cu.calculate_g2_B(dict(reversed(list(ds.items()))))
    assert list(fw) == ["a", "b", "c"] and list(rv) == ["c", "b", "a"]
    for k in fw:
        pd.testing.assert_frame_equal(fw[k], rv[k])


def test_calculate_g2_B_GM05_nonpositive_rows_irrelevant():
    base = d_cum()
    junk = pd.DataFrame({"t [s]": [5e-5, 6e-5, 7e-5], "g(2)-1": [0.0, -0.3, -1e-9]})
    mixed = pd.concat([base, junk], ignore_index=True)
    a = cu.calculate_g2_B({"x": base})["x"]
    b = cu.calculate_g2_B({"x": mixed})["x"].sort_values("t [s]").reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_calculate_g2_B_GE01_missing_column_dropped():
    r = cu.calculate_g2_B({"x": pd.DataFrame({"t [s]": [1.0]}), "y": d_cum()})
    assert list(r) == ["y"]


def test_calculate_g2_B_GE02_empty_frame():
    r = cu.calculate_g2_B({"e": pd.DataFrame({"t [s]": [], "g(2)-1": []})})
    assert len(r["e"]) == 0 and "g(2)_mod" in r["e"].columns


def test_calculate_g2_B_GE03_all_nonpositive():
    r = cu.calculate_g2_B({"e": pd.DataFrame({"t [s]": [1., 2], "g(2)-1": [-1., 0.0]})})
    assert len(r["e"]) == 0


def test_calculate_g2_B_GE04_empty_dict():
    assert cu.calculate_g2_B({}) == {}


def test_calculate_g2_B_GE05_string_values_dropped():
    r = cu.calculate_g2_B({"s": pd.DataFrame({"t [s]": [1., 2], "g(2)-1": ["a", "b"]}),
                           "ok": d_cum()})
    assert list(r) == ["ok"]


def test_calculate_g2_B_GC01_nan_dropped_silently_documents_current_behavior(capsys):
    """documents current behavior, not desired behavior (F-13)."""
    d = pd.DataFrame({"t [s]": [1., 2, 3], "g(2)-1": [0.5, np.nan, 0.2]})
    r = cu.calculate_g2_B({"a": d})["a"]
    assert list(r["t [s]"]) == [1., 3.]
    assert "non-positive" not in capsys.readouterr().out


def test_calculate_g2_B_GC02_inf_kept_documents_current_behavior():
    """documents current behavior, not desired behavior (F-13)."""
    d = pd.DataFrame({"t [s]": [1., 2], "g(2)-1": [0.5, np.inf]})
    r = cu.calculate_g2_B({"a": d})["a"]
    assert len(r) == 2 and r["g(2)_mod"].iloc[1] == np.inf


def test_calculate_g2_B_GC03_value_above_one_kept_documents_current_behavior():
    """documents current behavior, not desired behavior: no upper bound."""
    d = pd.DataFrame({"t [s]": [1.], "g(2)-1": [1.5]})
    r = cu.calculate_g2_B({"a": d})["a"]
    assert r["g(2)_mod"].iloc[0] == pytest.approx(0.5 * math.log(1.5), **TOL)
    assert r["g(2)_mod"].iloc[0] > 0


def test_calculate_g2_B_GC04_none_raises_attributeerror_documents_current_behavior():
    """documents current behavior, not desired behavior (F-13): None not caught."""
    with pytest.raises(AttributeError):
        cu.calculate_g2_B({"a": None})


# ============================================================================
# plot_processed_correlations (method B)
# ============================================================================
def g2b(d):
    return cu.calculate_g2_B({"m": d})


def cum_expected():
    """Closed-form OLS of 0.5 ln(g2-1) on t for D-CUM (Gamma, a, R2)."""
    t = T_CUM
    S = np.sum((t - t.mean()) * t ** 2) / np.sum((t - t.mean()) ** 2)
    icpt_t2 = np.mean(t ** 2) - S * t.mean()
    Gamma = 1000.0 - (2e5 / 2) * S
    a = 0.9 * math.exp(2e5 * icpt_t2)
    e = (2e5 / 2) * (t ** 2 - S * t - icpt_t2)
    y = 0.5 * np.log(0.9) - 1000.0 * t + 1e5 * t ** 2
    r2 = 1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)
    return Gamma, a, r2


def test_plot_processed_correlations_BO01_mono():
    r = pc(g2b(d_mono()), (0.0, 5e-4)).iloc[0]
    assert r["Gamma"] == pytest.approx(1500.0, rel=1e-6)
    assert r["a"] == pytest.approx(0.8, rel=1e-6)
    assert r["R_squared"] == pytest.approx(1.0, abs=1e-10)
    assert abs(r["Gamma_error"]) < 1e-6 and abs(r["Gamma_relative_error"]) < 1e-6
    assert r["weighted"] == False  # noqa: E712
    assert r["filename"] == "m"


def test_plot_processed_correlations_BO02_hand_unweighted():
    r = pc({"w": d_w(weight=False)}, (0, 2)).iloc[0]
    assert r["Gamma"] == pytest.approx(1.5, **TOL)
    assert r["a"] == pytest.approx(math.exp(1 / 3), **TOL)
    assert r["R_squared"] == pytest.approx(27 / 28, **TOL)
    assert r["Gamma_error"] == pytest.approx(math.sqrt(1 / 12), **TOL)
    assert r["Gamma_relative_error"] == pytest.approx(math.sqrt(1 / 12) / 1.5 * 100, **TOL)


def test_plot_processed_correlations_BO03_cumulant_series_bias():
    G, a, r2 = cum_expected()
    assert G == pytest.approx(980.0, rel=1e-3)
    assert a == pytest.approx(0.898861, rel=1e-5)
    r = pc(g2b(d_cum()), (0.0, 1.0)).iloc[0]
    assert r["Gamma"] == pytest.approx(G, **TOL)
    assert r["a"] == pytest.approx(a, **TOL)
    assert r["R_squared"] == pytest.approx(r2, **TOL)


def test_plot_processed_correlations_BO04_window_inclusive():
    d = pd.DataFrame({"t [s]": [0., 1, 2, 3], "g(2)_mod": [5., 0, -1, 7]})
    r = pc({"x": d}, (1, 2)).iloc[0]
    assert r["Gamma"] == pytest.approx(1.0, **TOL)
    assert r["a"] == pytest.approx(math.exp(2), **TOL)
    assert r["R_squared"] == pytest.approx(1.0, abs=1e-12)


def test_plot_processed_correlations_BO05_hand_weighted():
    r = pc({"w": d_w()}, (0, 2)).iloc[0]
    assert r["Gamma"] == pytest.approx(17 / 11, **TOL)
    assert r["a"] == pytest.approx(math.exp(4 / 11), **TOL)
    assert r["R_squared"] == pytest.approx(289 / 297, **TOL)
    assert r["weighted"] == True  # noqa: E712
    assert pd.isna(r["Gamma_error"]) and pd.isna(r["Gamma_relative_error"])


def test_plot_processed_correlations_BO06_columns_and_order():
    ds = cu.calculate_g2_B({"z": d_mono(), "y": d_cum(), "x": d_mono()})
    r = pc(ds, (0, 5e-4))
    assert list(r.columns) == PC_COLUMNS
    assert len(r) == 3 and list(r["filename"]) == ["z", "y", "x"]


@pytest.mark.parametrize("s", [1e3])
def test_plot_processed_correlations_BM01_time_scale(s):
    d = d_cum()
    ds = d.copy()
    ds["t [s]"] *= s
    a = pc(g2b(d), (0.0, 1.0)).iloc[0]
    b = pc(g2b(ds), (0.0, s * 1.0)).iloc[0]
    assert b["Gamma"] == pytest.approx(a["Gamma"] / s, **TOL)
    assert b["Gamma_error"] == pytest.approx(a["Gamma_error"] / s, **TOL)
    for c in ("a", "R_squared", "Gamma_relative_error"):
        assert b[c] == pytest.approx(a[c], **TOL)


@pytest.mark.parametrize("weighted", [False, True])
def test_plot_processed_correlations_BM02_beta_scaling(weighted):
    d = d_cum()
    if weighted:
        d["weight"] = 1.0 + 1e4 * d["t [s]"]
    d2 = d.copy()
    d2["g(2)-1"] *= 0.5
    a = pc(g2b(d), (0.0, 1.0)).iloc[0]
    b = pc(g2b(d2), (0.0, 1.0)).iloc[0]
    assert b["Gamma"] == pytest.approx(a["Gamma"], **TOL)
    assert b["R_squared"] == pytest.approx(a["R_squared"], **TOL)
    assert b["a"] == pytest.approx(a["a"] * 0.5, **TOL)
    if not weighted:
        assert b["Gamma_error"] == pytest.approx(a["Gamma_error"], **TOL)


def test_plot_processed_correlations_BM03_dict_permutation():
    ds = cu.calculate_g2_B({"a": d_mono(), "b": d_cum(), "c": d_w(False).assign(**{"g(2)-1": 0.5})})
    r1 = pc(ds, (0, 1e-4))
    r2 = pc(dict(reversed(list(ds.items()))), (0, 1e-4))
    assert list(r2["filename"]) == ["c", "b", "a"]
    pd.testing.assert_frame_equal(r1.set_index("filename").sort_index(),
                                  r2.set_index("filename").sort_index(), rtol=1e-12)


def test_plot_processed_correlations_BM04_points_outside_window_irrelevant():
    base = g2b(d_cum())["m"]
    extra = pd.DataFrame({"t [s]": [5e-4, 6e-4], "g(2)_mod": [100.0, -50.0],
                          "g(2)-1": [0.5, 0.5]})
    more = pd.concat([base, extra], ignore_index=True)
    a = pc({"m": base}, (0, 1e-4))
    b = pc({"m": more}, (0, 1e-4))
    pd.testing.assert_frame_equal(a, b, rtol=1e-12)


def test_plot_processed_correlations_BM05_nonpositive_rows_in_window():
    base = d_cum()
    junk = pd.DataFrame({"t [s]": [2.5e-5, 7.5e-5], "g(2)-1": [-0.2, 0.0]})
    a = pc(g2b(base), (0, 1e-4))
    b = pc(g2b(pd.concat([base, junk], ignore_index=True)), (0, 1e-4))
    pd.testing.assert_frame_equal(a, b, rtol=1e-12)


def test_plot_processed_correlations_BM06_r2_equals_linregress_rvalue_sq():
    d = g2b(d_cum())["m"]
    m = (d["t [s]"] >= 0) & (d["t [s]"] <= 1e-4)
    lr = stats.linregress(d["t [s]"][m], d["g(2)_mod"][m])
    assert pc({"m": d}, (0, 1e-4)).iloc[0]["R_squared"] == pytest.approx(lr.rvalue ** 2, rel=1e-6)


def test_plot_processed_correlations_BM07_weight_scaling():
    d = d_cum()
    d["weight"] = 1.0 + 1e4 * d["t [s]"]
    d7 = d.copy()
    d7["weight"] *= 7
    pd.testing.assert_frame_equal(pc(g2b(d), (0, 1e-4)), pc(g2b(d7), (0, 1e-4)), rtol=1e-9)


def test_plot_processed_correlations_BM08_constant_effective_weights_equal_ols():
    d = d_cum()
    dw = d.copy()
    dw["weight"] = 1.0 / d["g(2)-1"] ** 2
    a = pc(g2b(d), (0, 1.0)).iloc[0]
    b = pc(g2b(dw), (0, 1.0)).iloc[0]
    for c in ("Gamma", "a", "R_squared"):
        assert b[c] == pytest.approx(a[c], rel=1e-6)
    assert pd.isna(b["Gamma_error"]) and np.isfinite(a["Gamma_error"])


def test_plot_processed_correlations_BM09_weights_outside_window_irrelevant():
    d = d_cum()
    d["weight"] = 1.0 + 1e4 * d["t [s]"]
    d2 = d.copy()
    d2.loc[d2["t [s]"] > 1e-4, "weight"] *= 13.0
    a = pc(g2b(d), (0, 1e-4)).iloc[0]
    b = pc(g2b(d2), (0, 1e-4)).iloc[0]
    for c in ("Gamma", "a", "R_squared"):
        assert b[c] == pytest.approx(a[c], rel=1e-6)


def test_plot_processed_correlations_BM10_axis_limits_irrelevant():
    ds = g2b(d_cum())
    a = pc(ds, (0, 1e-4))
    b = pc(ds, (0, 1e-4), xlim=(0, 1), ylim=(-5, 1))
    pd.testing.assert_frame_equal(a, b)


def test_plot_processed_correlations_BM11_direct_g2_mod_column():
    ds = g2b(d_cum())
    direct = {"m": pd.DataFrame({"t [s]": ds["m"]["t [s]"], "g(2)_mod": ds["m"]["g(2)_mod"]})}
    pd.testing.assert_frame_equal(pc(ds, (0, 1e-4)), pc(direct, (0, 1e-4)))


def test_plot_processed_correlations_BE01_empty_dict():
    r = pc({}, (0, 1))
    assert len(r) == 0 and len(r.columns) == 0


def test_plot_processed_correlations_BE02_none_skipped():
    r = pc({"a": None, "b": g2b(d_cum())["m"]}, (0, 1e-4))
    assert len(r) == 1 and r.iloc[0]["filename"] == "b"


def test_plot_processed_correlations_BE03_missing_column_error_row():
    r = pc({"x": pd.DataFrame({"t [s]": [0., 1]}), "y": g2b(d_cum())["m"]}, (0, 1e-4))
    assert len(r) == 2 and "Error" in r.columns
    assert r.iloc[0]["Error"] == "'g(2)_mod'" and pd.isna(r.iloc[0]["Gamma"])
    assert pd.isna(r.iloc[1]["Error"])


def test_plot_processed_correlations_BE04_weighted_missing_g2_column():
    d = pd.DataFrame({"t [s]": [0., 1, 2], "g(2)_mod": [0., -1, -2], "weight": [1., 1, 1]})
    r = pc({"x": d}, (0, 2))
    assert r.iloc[0]["Error"] == "'g(2)-1'"


def _const_y():
    d = pd.DataFrame({"t [s]": [0., 1, 2], "g(2)_mod": [-0.5] * 3})
    return pc({"c": d}, (0, 2)).iloc[0]


def test_plot_processed_correlations_BE05_constant_y_documents_current_behavior():
    """documents current behavior: Gamma -0.0, rel. error inf, a=exp(-1), R2 NaN.
    Gamma_error is 0 (older scipy, spec) or NaN (scipy 1.17.1, observed): version dependent."""
    r = _const_y()
    assert r["Gamma"] == 0
    assert r["Gamma_error"] == 0 or np.isnan(r["Gamma_error"])
    assert r["Gamma_relative_error"] == np.inf
    assert r["a"] == pytest.approx(math.exp(-1), **TOL)
    assert pd.isna(r["R_squared"])


def test_plot_processed_correlations_BE06_nan_in_window():
    d = pd.DataFrame({"t [s]": [0., 1, 2, 3], "g(2)_mod": [0., np.nan, -2, -3]})
    r = pc({"n": d}, (0, 3)).iloc[0]
    assert pd.isna(r["Gamma"])


def test_plot_processed_correlations_BC01_figures_not_closed_documents_current_behavior():
    """documents current behavior, not desired behavior (F-11): +1 figure per valid file."""
    before = len(plt.get_fignums())
    pc({"a": g2b(d_cum())["m"], "n": None, "b": g2b(d_cum())["m"]}, (0, 1e-4))
    assert len(plt.get_fignums()) == before + 2


def test_plot_processed_correlations_BC02_two_points_zero_error_documents_current_behavior():
    """documents current behavior, not desired behavior (F-18): false precision for n == 2."""
    d = pd.DataFrame({"t [s]": [0., 1, 2, 3], "g(2)_mod": [5., 0, -1, 7]})
    r = pc({"x": d}, (1, 2)).iloc[0]
    assert r["Gamma_error"] == 0.0 and r["Gamma_relative_error"] == 0.0


def test_plot_processed_correlations_BC03_rising_data_negative_gamma_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    d = pd.DataFrame({"t [s]": [0., 1, 2], "g(2)_mod": [0., 1, 2]})
    assert pc({"x": d}, (0, 2)).iloc[0]["Gamma"] == pytest.approx(-1.0, **TOL)


def _empty_window_batch():
    d = g2b(d_cum())["m"]
    d3 = pd.DataFrame({"t [s]": [1., 2, 3], "g(2)_mod": [0., -1, -2]})
    return {"ok": d, "bad": d3}


def test_plot_processed_correlations_BC04_empty_window_documents_current_behavior():
    """documents current behavior (F-10, scipy-version dependent): either ValueError aborts the
    whole batch (older scipy, spec) or both files yield NaN rows (scipy 1.17.1, observed)."""
    try:
        r = pc(_empty_window_batch(), (10, 20))
    except ValueError:
        return  # branch A: exception
    assert len(r) == 2 and r["Gamma"].isna().all()  # branch B: NaN rows


def test_plot_processed_correlations_BC05_identical_t_raises_documents_current_behavior():
    """documents current behavior (F-10)."""
    d = pd.DataFrame({"t [s]": [1., 1, 1], "g(2)_mod": [0., -1, -2]})
    with pytest.raises(ValueError):
        pc({"x": d}, (0, 2))


def test_plot_processed_correlations_BC06_weighted_empty_window_documents_current_behavior():
    """documents current behavior (F-10): observed exception type recorded below."""
    d = d_w()
    with pytest.raises(ZeroDivisionError):
        pc({"x": d}, (10, 20))


def test_plot_processed_correlations_BC07_single_point_documents_current_behavior():
    """documents current behavior: accepts exactly NaN-row or an exception (verify@first-run)."""
    d = pd.DataFrame({"t [s]": [0., 1, 2], "g(2)_mod": [0., -1, -2]})
    try:
        r = pc({"x": d}, (1, 1))
    except Exception:
        return
    assert pd.isna(r.iloc[0]["Gamma"])


def test_plot_processed_correlations_BC08_signature_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    sig = inspect.signature(cu.plot_processed_correlations)
    assert list(sig.parameters) == ["dataframes_dict", "fit_x_limits", "xlim", "ylim"]
    assert sig.parameters["xlim"].default is None and sig.parameters["ylim"].default is None


def test_plot_processed_correlations_BC09_gui_copy_empty_window_documents_current_behavior():
    """documents current behavior (F-10, scipy-version dependent): GUI copy either raises
    ValueError (branch A) or returns a NaN row (branch B, scipy 1.17.1 observed)."""
    from ade_dls.gui.analysis import cumulant_plotting as cp
    d3 = pd.DataFrame({"t [s]": [1., 2, 3], "g(2)_mod": [0., -1, -2]})
    try:
        r, _ = cp.plot_processed_correlations_no_show({"bad": d3}, (10, 20))
    except ValueError:
        return
    assert len(r) == 1 and pd.isna(r.iloc[0]["Gamma"])


def test_plot_processed_correlations_P01_gui_copy_parity():
    from matplotlib.figure import Figure
    from ade_dls.gui.analysis import cumulant_plotting as cp
    ds = cu.calculate_g2_B({"mono": d_mono(), "cum": d_cum()})
    ds["w"] = d_w()
    ds["err"] = pd.DataFrame({"t [s]": [0., 1]})
    lim = (0, 5e-4)
    # (0, 5e-4) is outside the D-W window; use per-file windows via two calls
    lim_w = (0, 2)
    ds_a = {k: v for k, v in ds.items() if k != "w"}
    for dd, li in ((ds_a, lim), ({"w": ds["w"]}, lim_w)):
        r_mod = cu.plot_processed_correlations(dd, li)
        r_gui, plots = cp.plot_processed_correlations_no_show(dd, li)
        pd.testing.assert_frame_equal(r_mod, r_gui, rtol=1e-12)
        for k in dd:
            if k != "err":
                assert isinstance(plots[k][0], Figure)


def test_plot_processed_correlations_P02_pipeline_oracle():
    q2 = np.array([1, 2, 3, 4, 5]) * 1e-4
    D = 4e6
    gam = D * q2
    dd = {f"f{i}": corr_df(T_MONO, mono(T_MONO, 0.85, g)) for i, g in enumerate(gam)}
    res = pc(cu.calculate_g2_B(dd), (0, 5e-4))
    np.testing.assert_allclose(res["Gamma"], gam, rtol=1e-6)
    m = res.merge(pd.DataFrame({"filename": list(dd), "q^2": q2}), on="filename")
    free = cu.analyze_diffusion_coefficient(m, "q^2", "Gamma", show_plots=False).iloc[0]
    assert free["slope"] == pytest.approx(4e6, rel=1e-6)
    assert abs(free["intercept"]) < 1e-6 * 2000
    assert free["R_squared"] == pytest.approx(1.0, abs=1e-9)
    org = cu.analyze_diffusion_coefficient(m, "q^2", "Gamma", show_plots=False,
                                           fit_through_origin=True).iloc[0]
    assert org["slope"] == pytest.approx(4e6, rel=1e-6)
    assert free["slope"] * 1e-18 == pytest.approx(4e-12, rel=1e-6)


def test_plot_processed_correlations_P03_unit_consistency_ms_vs_s():
    q2 = np.array([1, 2, 3, 4, 5]) * 1e-4
    gam = 4e6 * q2

    def slope(tscale, gscale, lim):
        dd = {f"f{i}": corr_df(T_MONO * tscale, mono(T_MONO * tscale, 0.85, g * gscale))
              for i, g in enumerate(gam)}
        res = pc(cu.calculate_g2_B(dd), lim)
        m = res.merge(pd.DataFrame({"filename": list(dd), "q^2": q2}), on="filename")
        return cu.analyze_diffusion_coefficient(m, "q^2", "Gamma", show_plots=False).iloc[0]["slope"]

    s_s = slope(1.0, 1.0, (0, 5e-4))
    s_ms = slope(1e3, 1e-3, (0, 5e-4 * 1e3))
    assert s_ms == pytest.approx(s_s / 1e3, rel=1e-6)
    assert s_ms * 1e-15 == pytest.approx(s_s * 1e-18, rel=1e-6)


# ============================================================================
# analyze_diffusion_coefficient
# ============================================================================
def d_norm_free():
    return 3 + 2 * Q_20 + R_NORM


def test_analyze_diffusion_coefficient_RO01_hand_ols():
    r = adc(Q_LIN, G_LIN).iloc[0]
    assert list(adc(Q_LIN, G_LIN).columns) == ADC_COLUMNS
    assert r["slope"] == pytest.approx(1.97, **TOL)
    assert r["intercept"] == pytest.approx(0.09, **TOL)
    assert r["slope_se"] == pytest.approx(math.sqrt(0.091 / 3 / 10), **TOL)
    assert r["intercept_se"] == pytest.approx(math.sqrt(0.091 / 3 * 1.1), **TOL)
    assert r["R_squared"] == pytest.approx(1 - 0.091 / 38.9, **TOL)
    m = moments(E_LIN)
    assert r["Skewness"] == pytest.approx(m["skew"], **TOL)
    assert r["Kurtosis"] == pytest.approx(m["kurt"], **TOL)
    assert r["Skewness"] == pytest.approx(0.0953035, rel=1e-5)
    assert r["Kurtosis"] == pytest.approx(-1.347820, rel=1e-5)
    assert (r["N_points_fit"], r["X_range_min"], r["X_range_max"]) == (5, 1, 5)
    assert r["q^2_coef"] == pytest.approx(r["slope"]) and r["q^2_se"] == pytest.approx(r["slope_se"])
    assert r["const_coef"] == pytest.approx(r["intercept"])
    assert r["const_se"] == pytest.approx(r["intercept_se"])
    assert r["Column"] == "g" and r["Method"] == "g"
    assert r["JB_p_value"] == pytest.approx(jb_p(E_LIN), **PTOL)


def d_b2():
    return pd.DataFrame({"q^2": [1., 2, 3, 4, 5, 6],
                         "gA": [2.1, 3.9, 6.2, 7.8, 10.0, np.nan],
                         "gB": [np.nan, 4, np.nan, 8, 10, 12.5]})


@pytest.mark.parametrize("which", ["lin", "b2B", "rnd"])
def test_analyze_diffusion_coefficient_RO02_vs_linregress(which):
    if which == "lin":
        x, y = Q_LIN, G_LIN
    elif which == "b2B":
        x, y = np.array([2., 4, 5, 6]), np.array([4., 8, 10, 12.5])
    else:
        x, y = d_rnd()
    if which == "b2B":
        r = adf(d_b2(), ["gB"]).iloc[0]
    else:
        r = adc(x, y).iloc[0]
    lr = stats.linregress(x, y)
    assert r["slope"] == pytest.approx(lr.slope, rel=1e-6)
    assert r["intercept"] == pytest.approx(lr.intercept, rel=1e-6)
    assert r["slope_se"] == pytest.approx(lr.stderr, rel=1e-6)
    assert r["intercept_se"] == pytest.approx(lr.intercept_stderr, rel=1e-6)
    assert r["R_squared"] == pytest.approx(lr.rvalue ** 2, rel=1e-6)


def test_analyze_diffusion_coefficient_RO03_normal_residuals_free():
    r = adc(Q_20, d_norm_free()).iloc[0]
    ssr = np.sum(R_NORM ** 2)
    assert ssr == pytest.approx(2.372e-3, rel=1e-12)
    s2 = ssr / 18
    assert r["slope"] == pytest.approx(2.0, rel=1e-6, abs=1e-9)
    assert r["intercept"] == pytest.approx(3.0, abs=1e-9)
    assert r["slope_se"] == pytest.approx(math.sqrt(s2 / 665), rel=1e-6)
    assert r["intercept_se"] == pytest.approx(math.sqrt(s2 * (1 / 20 + 10.5 ** 2 / 665)), rel=1e-6)
    assert r["R_squared"] == pytest.approx(1 - ssr / (4 * 665 + ssr), rel=1e-6)
    assert abs(r["Skewness"]) < 1e-6
    assert r["Kurtosis"] == pytest.approx(moments(R_NORM)["kurt"], rel=1e-6)
    assert r["Kurtosis"] == pytest.approx(-0.644381, rel=1e-5)
    assert r["JB_p_value"] == pytest.approx(jb_p(R_NORM), **PTOL)
    assert r["Omnibus_p_value"] == pytest.approx(omni_p_sym(R_NORM), **PTOL)
    assert r["Omnibus_p_value"] >= 0.05
    assert r["Normality"] == "Normally distributed"


def test_analyze_diffusion_coefficient_RO04_normal_residuals_origin():
    r = adc(Q_20, 2 * Q_20 + R_NORM, fit_through_origin=True).iloc[0]
    ssr = np.sum(R_NORM ** 2)
    assert r["slope"] == pytest.approx(2.0, rel=1e-6)
    assert r["slope_se"] == pytest.approx(math.sqrt(ssr / 19 / 2870), rel=1e-6)
    # D-12: origin fit uses the uncentered R2 = 1 - SSR / sum(y^2) (R_NORM is orthogonal to q)
    assert r["R_squared"] == pytest.approx(1 - ssr / (4 * 2870 + ssr), rel=1e-6)
    assert r["R_squared"] == pytest.approx(origin_hand(Q_20, 2 * Q_20 + R_NORM)["R2u"], rel=1e-6)
    for c in ("intercept", "intercept_se", "const_coef", "const_se"):
        assert r[c] == 0.0
    assert r["JB_p_value"] == pytest.approx(jb_p(R_NORM), rel=1e-6)
    assert r["Omnibus_p_value"] == pytest.approx(omni_p_sym(R_NORM), rel=1e-6)
    assert r["Kurtosis"] == pytest.approx(moments(R_NORM)["kurt"], rel=1e-6)
    assert r["Normality"] == "Normally distributed"


def outl(origin):
    g = (2 * Q_20 if origin else 1 + 2 * Q_20).copy()
    g[9] += 100
    return g


@BOTH
def test_analyze_diffusion_coefficient_RO05_outlier_not_normal(origin):
    r = adc(Q_20, outl(origin), fit_through_origin=origin).iloc[0]
    assert r["Normality"] == "Not normally distributed"
    assert r["Skewness"] > 1 and r["Kurtosis"] > 0
    assert r["JB_p_value"] < 0.05 and r["Omnibus_p_value"] < 0.05
    if not origin:
        h = ols_hand(Q_20, outl(False))
        assert r["slope"] == pytest.approx(h["b"], rel=1e-6)
        assert r["intercept"] == pytest.approx(h["a"], rel=1e-6)


def test_analyze_diffusion_coefficient_RO06_two_columns_with_nan():
    res = adf(d_b2(), ["gA", "gB"])
    assert list(res["Column"]) == ["gA", "gB"]
    a, b = res.iloc[0], res.iloc[1]
    assert a["slope"] == pytest.approx(1.97, **TOL)
    assert a["intercept"] == pytest.approx(0.09, **TOL)
    assert b["slope"] == pytest.approx(2.1, **TOL)
    assert b["intercept"] == pytest.approx(-0.3, **TOL)
    assert b["R_squared"] == pytest.approx(1 - 0.1 / 38.6875, **TOL)
    assert b["slope_se"] == pytest.approx(math.sqrt(0.1 / 2 / 8.75), **TOL)
    assert (b["N_points_fit"], b["X_range_min"], b["X_range_max"]) == (4, 2, 6)


def test_analyze_diffusion_coefficient_RO07_x_range_inclusive():
    q = np.concatenate([Q_LIN, [10., 20]])
    g = np.concatenate([G_LIN, [100., -50]])
    r = adc(q, g, x_range=(1, 5)).iloc[0]
    ref = adc(Q_LIN, G_LIN).iloc[0]
    cmp_rows(r, ref)
    assert r["N_points_fit"] == 5


def test_analyze_diffusion_coefficient_RO08_offset_data():
    q = np.arange(1, 9, dtype=float)
    g = 10 + 0.5 * q
    o = adc(q, g, fit_through_origin=True).iloc[0]
    assert o["slope"] == pytest.approx(77 / 34, **TOL)
    assert o["slope_se"] == pytest.approx(math.sqrt((400 / 17) / 204), **TOL)
    # D-12: uncentered R2 (same as GUI): 1 - SSR/sum(y^2) = 0.863992
    assert o["R_squared"] == pytest.approx(origin_hand(q, g)["R2u"], **TOL)
    assert o["R_squared"] == pytest.approx(0.863992, rel=1e-5)
    f = adc(q, g).iloc[0]
    assert f["slope"] == pytest.approx(0.5, abs=1e-9)
    assert f["intercept"] == pytest.approx(10, abs=1e-9)
    assert f["R_squared"] == pytest.approx(1.0, abs=1e-9)


def test_analyze_diffusion_coefficient_RO09_string_arguments():
    df = pd.DataFrame({"q^2": Q_LIN, "g": G_LIN})
    r = cu.analyze_diffusion_coefficient(df, "q^2", "g", method_names="M", show_plots=False)
    assert len(r) == 1 and r.iloc[0]["Column"] == "g" and r.iloc[0]["Method"] == "M"


def test_analyze_diffusion_coefficient_RO10_constant_y_origin_uncentered_r2():
    """D-12: constant y has zero centered variance but sum(y^2) > 0 -> uncentered R2 is defined:
    SSR = 200 - 180^2/204, R2 = (180^2/204)/200 = 0.794118."""
    q = np.arange(1, 9, dtype=float)
    r = adc(q, [5.0] * 8, fit_through_origin=True).iloc[0]
    assert r["slope"] == pytest.approx(5 * 36 / 204, **TOL)
    assert r["R_squared"] == pytest.approx(180 ** 2 / 204 / 200, **TOL)


def test_analyze_diffusion_coefficient_RO10b_all_zero_y_origin_nan_r2():
    q = np.arange(1, 9, dtype=float)
    r = adc(q, [0.0] * 8, fit_through_origin=True).iloc[0]
    assert pd.isna(r["R_squared"])


# --- metamorphic -------------------------------------------------------------
@BOTH
def test_analyze_diffusion_coefficient_RM01_permutation(origin):
    q, g = d_rnd()
    p = np.random.default_rng(7).permutation(len(q))
    a = adc(q, g, fit_through_origin=origin).iloc[0]
    b = adc(q[p], g[p], fit_through_origin=origin).iloc[0]
    cmp_rows(b, a)


@BOTH
@pytest.mark.parametrize("c", [7.3, 1000.0])
def test_analyze_diffusion_coefficient_RM02_gamma_scaling(origin, c):
    q, g = d_rnd()
    a = adc(q, g, fit_through_origin=origin).iloc[0]
    b = adc(q, g * c, fit_through_origin=origin).iloc[0]
    cmp_rows(b, a, factors={k: c for k in
                            ["slope", "slope_se", "intercept", "intercept_se",
                             "q^2_coef", "q^2_se", "const_coef", "const_se"]})
    if c == 1000.0:
        assert a["slope"] * 1e-15 * 1000 == pytest.approx(b["slope"] * 1e-15, rel=1e-6)
        assert b["slope"] * 1e-18 == pytest.approx(a["slope"] * 1e-15, rel=1e-6)


@BOTH
def test_analyze_diffusion_coefficient_RM03_q2_scaling(origin):
    c = 1e6
    q, g = d_rnd()
    a = adc(q, g, fit_through_origin=origin).iloc[0]
    b = adc(q * c, g, fit_through_origin=origin).iloc[0]
    cmp_rows(b, a, factors={"slope": 1 / c, "slope_se": 1 / c, "q^2_coef": 1 / c,
                            "q^2_se": 1 / c, "X_range_min": c, "X_range_max": c})
    r = adc(q * c, g, x_range=(q[2] * c, q[8] * c), fit_through_origin=origin).iloc[0]
    r0 = adc(q, g, x_range=(q[2], q[8]), fit_through_origin=origin).iloc[0]
    cmp_rows(r, r0, factors={"slope": 1 / c, "slope_se": 1 / c, "q^2_coef": 1 / c,
                             "q^2_se": 1 / c, "X_range_min": c, "X_range_max": c})


def test_analyze_diffusion_coefficient_RM04_gamma_offset_free():
    q, g = d_rnd()
    a = adc(q, g).iloc[0]
    b = adc(q, g + 5.0).iloc[0]
    cmp_rows(b, a, factors={"intercept": 1.0}, skip=("intercept", "const_coef"))
    assert b["intercept"] == pytest.approx(a["intercept"] + 5.0, rel=1e-6)
    assert b["const_coef"] == pytest.approx(a["const_coef"] + 5.0, rel=1e-6)


def test_analyze_diffusion_coefficient_RM05_duplicate_rows_free():
    a = adc(Q_LIN, G_LIN).iloc[0]
    b = adc(np.tile(Q_LIN, 2), np.tile(G_LIN, 2)).iloc[0]
    f = math.sqrt(3 / 8)
    assert f == pytest.approx(math.sqrt((5 - 2) / (2 * (5 - 1))))
    for c in ("slope", "intercept", "R_squared", "Skewness", "Kurtosis"):
        assert b[c] == pytest.approx(a[c], **TOL)
    assert b["slope_se"] == pytest.approx(a["slope_se"] * f, rel=1e-6)
    assert b["intercept_se"] == pytest.approx(a["intercept_se"] * f, rel=1e-6)
    assert b["N_points_fit"] == 2 * a["N_points_fit"]


def test_analyze_diffusion_coefficient_RM05_duplicate_rows_origin():
    g = 2 * Q_20 + R_NORM
    a = adc(Q_20, g, fit_through_origin=True).iloc[0]
    b = adc(np.tile(Q_20, 2), np.tile(g, 2), fit_through_origin=True).iloc[0]
    for c in ("slope", "intercept", "R_squared", "Skewness", "Kurtosis"):
        assert b[c] == pytest.approx(a[c], rel=1e-6, abs=1e-12)
    assert b["slope_se"] == pytest.approx(a["slope_se"] * math.sqrt(19 / 39), rel=1e-6)
    assert b["N_points_fit"] == 40


@BOTH
def test_analyze_diffusion_coefficient_RM06_nan_gamma_rows_ignored(origin):
    q, g = d_rnd()
    a = adc(q, g, fit_through_origin=origin).iloc[0]
    b = adc(np.append(q, [9e-4, 1e-3]), np.append(g, [np.nan, np.nan]),
            fit_through_origin=origin).iloc[0]
    cmp_rows(b, a)


@BOTH
def test_analyze_diffusion_coefficient_RM07_rows_outside_x_range_ignored(origin):
    q, g = d_rnd()
    lo, hi = q[0], q[-1]
    a = adc(q, g, x_range=(lo, hi), fit_through_origin=origin).iloc[0]
    b = adc(np.append(q, [2e-3, 3e-3]), np.append(g, [1e9, -5.0]), x_range=(lo, hi),
            fit_through_origin=origin).iloc[0]
    cmp_rows(b, a)


@BOTH
def test_analyze_diffusion_coefficient_RM08_x_range_equals_prefiltered(origin):
    q, g = d_rnd()
    lo, hi = q[2], q[9]
    m = (q >= lo) & (q <= hi)
    a = adc(q, g, x_range=(lo, hi), fit_through_origin=origin).iloc[0]
    b = adc(q[m], g[m], fit_through_origin=origin).iloc[0]
    cmp_rows(b, a)


def test_analyze_diffusion_coefficient_RM09_multi_vs_single_and_reorder():
    df = d_b2()
    multi = adf(df, ["gA", "gB"])
    for i, c in enumerate(["gA", "gB"]):
        single = adf(df, [c]).iloc[0]
        cmp_rows(multi.iloc[i], single, skip=("Column", "Method"))
        assert multi.iloc[i]["Column"] == c
    rev = adf(df, ["gB", "gA"])
    assert list(rev["Column"]) == ["gB", "gA"]
    cmp_rows(rev.iloc[0], multi.iloc[1])


@pytest.mark.parametrize("idx", [lambda n: np.arange(1, n + 1), lambda n: 10 * np.arange(1, n + 1)])
def test_analyze_diffusion_coefficient_RM10_index_irrelevant(idx):
    df = d_b2()
    a = adf(df, ["gA", "gB"])
    df2 = df.copy()
    df2.index = idx(len(df))
    b = adf(df2, ["gA", "gB"])
    for i in range(2):
        cmp_rows(b.iloc[i], a.iloc[i])


def test_analyze_diffusion_coefficient_RM11_labels_irrelevant():
    q, g = d_rnd()
    a = adc(q, g).iloc[0]
    b = adc(q, g, gamma_unit="1/ms", method_names="Z").iloc[0]
    cmp_rows(b, a)
    assert b["Method"] == "Z"


def test_analyze_diffusion_coefficient_RM12_no_mutation():
    df = d_b2()
    cp = copy.deepcopy(df)
    adf(df, ["gA", "gB"], x_range=(1, 5))
    assert df.equals(cp)


def test_analyze_diffusion_coefficient_RM13_r2_equals_corrcoef_sq():
    q, g = d_rnd()
    assert adc(q, g).iloc[0]["R_squared"] == pytest.approx(np.corrcoef(q, g)[0, 1] ** 2, rel=1e-6)


# --- edges -------------------------------------------------------------------
def test_analyze_diffusion_coefficient_RE01_missing_q2_column():
    with pytest.raises(ValueError, match=re.escape("Column 'q^2' not found in the DataFrame")):
        cu.analyze_diffusion_coefficient(pd.DataFrame({"g": [1., 2]}), "q^2", "g", show_plots=False)


@pytest.mark.parametrize("cols", [["nope"], []])
def test_analyze_diffusion_coefficient_RE02_RE03_no_valid_columns(cols):
    r = adf(pd.DataFrame({"q^2": Q_LIN, "g": G_LIN}), cols)
    assert len(r) == 0 and len(r.columns) == 0


@pytest.mark.parametrize("xr", [(1, 2, 3), 5, np.array([1, 5]), "ab"])
def test_analyze_diffusion_coefficient_RE04_invalid_x_range(xr):
    with pytest.raises(ValueError, match="x_range must be a tuple or list of length 2"):
        adc(Q_LIN, G_LIN, x_range=xr)


@pytest.mark.parametrize("xr", [(2.5, 2.9), (3, 3), (5, 1)])
def test_analyze_diffusion_coefficient_RE05_too_few_points_skipped(xr):
    r = adc(Q_LIN, G_LIN, x_range=xr)
    assert len(r) == 0 and len(r.columns) == 0


def test_analyze_diffusion_coefficient_RE06_list_x_range():
    cmp_rows(adc(Q_LIN, G_LIN, x_range=[1, 5]).iloc[0], adc(Q_LIN, G_LIN).iloc[0])


def test_analyze_diffusion_coefficient_RE07_two_points_no_se():
    r = adc(Q_LIN, G_LIN, x_range=(1, 2)).iloc[0]
    assert r["slope"] == pytest.approx(1.8, **TOL)
    assert r["intercept"] == pytest.approx(0.3, **TOL)
    assert r["R_squared"] == pytest.approx(1.0, abs=1e-9)
    assert not np.isfinite(r["slope_se"])


# --- characterization --------------------------------------------------------
def test_analyze_diffusion_coefficient_RC01_omnibus_nan_below_8_documents_current_behavior():
    """documents current behavior, not desired behavior (F-04): n=5 -> Omnibus NaN."""
    r = adc(Q_LIN, G_LIN).iloc[0]
    assert pd.isna(r["Omnibus_p_value"])
    assert r["Normality"] == "Possibly not normally distributed"


def test_analyze_diffusion_coefficient_RC02_p_value_rounding_documents_current_behavior():
    """documents current behavior (F-03): free-path p-values from formatted summary text
    (rounded to multiples of 0.001, observed); origin path unrounded."""
    q, g = d_rnd()
    r = adc(q, g).iloc[0]
    v = r["Omnibus_p_value"]
    assert abs(v * 1000 - round(v * 1000)) < 1e-9
    res = ols_hand(q, g)["e"]
    assert r["JB_p_value"] == pytest.approx(jb_p(res), **PTOL)


def test_analyze_diffusion_coefficient_RC03_origin_r2_is_uncentered_and_never_negative():
    """F-02 resolved (D-12): origin fit reports the uncentered R2; module == GUI definition.
    (The centered value would be -14.686275 for this data.)"""
    q = np.arange(1, 9, dtype=float)
    r = adc(q, 10 + 0.5 * q, fit_through_origin=True).iloc[0]
    assert r["R_squared"] == pytest.approx(origin_hand(q, 10 + 0.5 * q)["R2u"], **TOL)
    assert 0.0 <= r["R_squared"] <= 1.0
    assert r["R_squared"] != pytest.approx(origin_hand(q, 10 + 0.5 * q)["R2c"], rel=1e-3)


def test_analyze_diffusion_coefficient_RC18_gui_uncentered_r2_documents_current_behavior():
    """documents current behavior (F-02): GUI regression gives uncentered R2 0.863992."""
    from ade_dls.gui.analysis.cumulant_analyzer import CumulantAnalyzer
    q = np.arange(1, 9, dtype=float)
    g = 10 + 0.5 * q
    model, slope, se, *_ = CumulantAnalyzer._fit_gamma_vs_q2(pd.Series(q), pd.Series(g), True)
    assert model.rsquared == pytest.approx(origin_hand(q, g)["R2u"], rel=1e-6)
    assert model.rsquared == pytest.approx(0.863992, rel=1e-5)
    mod = adc(q, g, fit_through_origin=True).iloc[0]
    assert slope == pytest.approx(mod["slope"], rel=1e-6)
    assert se == pytest.approx(mod["slope_se"], rel=1e-6)


def test_analyze_diffusion_coefficient_RC04_lepto_possibly_not_normal_documents_current_behavior():
    """documents current behavior (F-04): kurtosis>0 alone downgrades the label."""
    r = adc(Q_20, 3 + 2 * Q_20 + R_LEPTO).iloc[0]
    assert r["JB_p_value"] >= 0.05 and r["Omnibus_p_value"] >= 0.05
    assert r["Kurtosis"] == pytest.approx(0.094407, rel=1e-4)
    assert r["Normality"] == "Possibly not normally distributed"


def test_analyze_diffusion_coefficient_RC05_method_names_mutated_documents_current_behavior():
    """documents current behavior (F-06): caller's list is extended in place."""
    names = ["A"]
    r = adf(d_b2(), ["gA", "gB"], method_names=names)
    assert names == ["A", ""]
    assert list(r["Method"]) == ["A", ""]


def test_analyze_diffusion_coefficient_RC06_origin_zero_intercepts_documents_current_behavior():
    """documents current behavior (F-17): const_* keys are ADE-specific zeros."""
    r = adc(Q_20, 2 * Q_20 + R_NORM, fit_through_origin=True).iloc[0]
    assert r["intercept"] == r["intercept_se"] == r["const_coef"] == r["const_se"] == 0.0


def test_analyze_diffusion_coefficient_RC07_figures_documents_current_behavior():
    """documents current behavior (F-16): show_plots=False leaks no figure; True leaves one each."""
    before = len(plt.get_fignums())
    adc(Q_LIN, G_LIN, show_plots=False)
    assert len(plt.get_fignums()) == before
    adf(d_b2(), ["gA", "gB"], show_plots=True)
    assert len(plt.get_fignums()) == before + 2


def test_analyze_diffusion_coefficient_RC08_x_range_columns_are_fit_extremes_documents_current_behavior():
    """documents current behavior: X_range_* are data extremes, not x_range bounds."""
    q = np.concatenate([Q_LIN, [10., 20]])
    g = np.concatenate([G_LIN, [100., -50]])
    r = adc(q, g, x_range=(0.5, 5.5)).iloc[0]
    assert (r["X_range_min"], r["X_range_max"]) == (1, 5)


def test_analyze_diffusion_coefficient_RC09_signature_documents_current_behavior():
    """documents current behavior (F-16): position 7 is show_plots."""
    sig = inspect.signature(cu.analyze_diffusion_coefficient)
    assert list(sig.parameters) == ["data_df", "q_squared_col", "gamma_cols", "method_names",
                                    "gamma_unit", "x_range", "show_plots", "fit_through_origin"]
    d = {k: v.default for k, v in sig.parameters.items()}
    assert d["method_names"] is None and d["gamma_unit"] == "1/s" and d["x_range"] is None
    assert d["show_plots"] is True and d["fit_through_origin"] is False


def test_analyze_diffusion_coefficient_RC10_method_defaults_to_column_documents_current_behavior():
    """documents current behavior."""
    r = adf(d_b2(), ["gA", "gB"])
    assert list(r["Method"]) == ["gA", "gB"]


def test_analyze_diffusion_coefficient_RC11_single_point_not_skipped_documents_current_behavior():
    """documents current behavior (F-01/F-05): not skipped (JADE skips). Observed (statsmodels
    0.14.6, scipy 1.17.1): AttributeError from the free fit (spec derivation predicted a
    degenerate row slope=intercept=3.0); origin fit gives a row with N=1, slope 3.0."""
    df = pd.DataFrame({"q^2": [1., 2, 3], "g": [np.nan, 6.0, np.nan]})
    try:
        r = adf(df, ["g"])
    except Exception:
        pass  # branch A (observed, statsmodels 0.14.6): exception from the free fit
    else:  # branch B: degenerate row, slope == intercept == 3.0
        assert r.iloc[0]["N_points_fit"] == 1
        assert r.iloc[0]["slope"] == pytest.approx(3.0, **TOL)
    r = adf(df, ["g"], fit_through_origin=True).iloc[0]
    assert r["N_points_fit"] == 1 and r["slope"] == pytest.approx(3.0, **TOL)


def test_analyze_diffusion_coefficient_RC12_zero_points_raises_documents_current_behavior():
    """documents current behavior (F-01): no valid point raises (JADE skips)."""
    df = pd.DataFrame({"q^2": [1., 2, 3], "g": [np.nan] * 3})
    with pytest.raises(Exception):
        adf(df, ["g"])
    assert len(adf(df, ["g"], x_range=(0, 10))) == 0


def test_analyze_diffusion_coefficient_RC13_single_angle_documents_current_behavior():
    """documents current behavior (F-05): constant q2 column treated as implicit constant;
    observed R_squared == 0 (statsmodels 0.14.6)."""
    r = adc([2., 2, 2], [4., 6, 8]).iloc[0]
    assert r["slope"] == pytest.approx(3.0, **TOL)
    assert r["intercept"] == pytest.approx(3.0, **TOL)
    assert r["R_squared"] == pytest.approx(0.0, abs=1e-12)


def test_analyze_diffusion_coefficient_RC14_nan_q2_raises_documents_current_behavior():
    """documents current behavior (F-15)."""
    q = Q_LIN.copy()
    q[2] = np.nan
    with pytest.raises(Exception):
        adc(q, G_LIN)


def test_analyze_diffusion_coefficient_RC15_inf_gamma_documents_current_behavior():
    """documents current behavior: inf passes the NaN filter; exception or non-finite slope."""
    g = G_LIN.copy()
    g[2] = np.inf
    try:
        r = adc(Q_LIN, g)
    except Exception:
        return
    assert not np.isfinite(r.iloc[0]["slope"])


def test_analyze_diffusion_coefficient_RC16_origin_small_n_documents_current_behavior():
    """documents current behavior (F-04): scipy-version dependent, ValueError or NaN omnibus."""
    try:
        r = adc(Q_LIN, G_LIN, fit_through_origin=True)
    except ValueError:
        return
    r = r.iloc[0]
    assert pd.isna(r["Omnibus_p_value"])
    assert r["Normality"] == "Possibly not normally distributed"


def test_analyze_diffusion_coefficient_RC17_invalid_x_range_without_columns_documents_current_behavior():
    """documents current behavior: x_range validated only inside the column loop."""
    r = adf(pd.DataFrame({"q^2": Q_LIN, "g": G_LIN}), ["nope"], x_range=(1, 2, 3))
    assert len(r) == 0 and len(r.columns) == 0


# --- xfail findings -----------------------------------------------------------
_F01 = ("FINDING F-01: ADE ueberspringt Gamma-Spalten mit <2 Punkten nicht (JADE schon); "
        "Soll offen, Fix-Freigabe offen, siehe _dev/notes/open_discussion_points.md")


@pytest.mark.xfail(strict=True, raises=(AttributeError, ValueError, IndexError), reason=_F01)
def test_analyze_diffusion_coefficient_XF01_single_point_should_be_skipped():
    df = pd.DataFrame({"q^2": [1., 2, 3], "g": [np.nan, 6.0, np.nan]})
    r = adf(df, ["g"])
    assert len(r) == 0


@pytest.mark.xfail(strict=True, raises=(AttributeError, ValueError, IndexError), reason=_F01)
def test_analyze_diffusion_coefficient_XF02_zero_points_should_be_skipped():
    df = pd.DataFrame({"q^2": [1., 2, 3], "g": [np.nan] * 3})
    r = adf(df, ["g"])
    assert len(r) == 0


# ============================================================================
# remove_rows_by_index
# ============================================================================
def d_idx():
    return pd.DataFrame({"v": [10, 20, 30, 40]}, index=[1, 2, 3, 4])


def test_remove_rows_by_index_IO01_two_rows():
    df = d_idx()
    r = cu.remove_rows_by_index(df, "2, 4")
    assert list(r.index) == [1, 3] and list(r["v"]) == [10, 30]
    assert len(df) == 4


def test_remove_rows_by_index_IO02_one_row():
    assert list(cu.remove_rows_by_index(d_idx(), "3").index) == [1, 2, 4]


def test_remove_rows_by_index_IM01_order_irrelevant():
    pd.testing.assert_frame_equal(cu.remove_rows_by_index(d_idx(), "4,2"),
                                  cu.remove_rows_by_index(d_idx(), "2,4"))


def test_remove_rows_by_index_IM02_sequential_equals_joint():
    seq = cu.remove_rows_by_index(cu.remove_rows_by_index(d_idx(), "2"), "4")
    pd.testing.assert_frame_equal(seq, cu.remove_rows_by_index(d_idx(), "2,4"))


def test_remove_rows_by_index_IM03_label_not_position():
    df = pd.DataFrame({"v": [1, 2, 3]}, index=[10, 20, 30])
    assert cu.remove_rows_by_index(df, "1").equals(df)
    assert list(cu.remove_rows_by_index(df, "20").index) == [10, 30]


@pytest.mark.parametrize("s", ["", None, "abc", "1,,2", "2.0", "99", "-1"])
def test_remove_rows_by_index_IE_returns_input_unchanged(s):
    df = d_idx()
    assert cu.remove_rows_by_index(df, s) is df


def test_remove_rows_by_index_IE05_whitespace_tolerated():
    assert list(cu.remove_rows_by_index(d_idx(), " 2 , 3 ").index) == [1, 4]


def test_remove_rows_by_index_IE06_invalid_label_ignored():
    assert list(cu.remove_rows_by_index(d_idx(), "2,99").index) == [1, 3, 4]


def test_remove_rows_by_index_IC01_duplicate_label_documents_current_behavior():
    """documents current behavior, not desired behavior (verify@first-run)."""
    assert list(cu.remove_rows_by_index(d_idx(), "2,2").index) == [1, 3, 4]


def test_remove_rows_by_index_IC02_nonunique_index_documents_current_behavior():
    """documents current behavior: both rows with the label are dropped."""
    df = pd.DataFrame({"v": [1, 2, 3]}, index=[1, 1, 2])
    r = cu.remove_rows_by_index(df, "1")
    assert list(r.index) == [2] and list(r["v"]) == [3]


def test_remove_rows_by_index_IC03_return_identity_documents_current_behavior():
    """documents current behavior (F-19): new object on removal, same object otherwise."""
    df = d_idx()
    assert cu.remove_rows_by_index(df, "2") is not df
    assert cu.remove_rows_by_index(df, "") is df


def test_remove_rows_by_index_IC04_string_index_documents_current_behavior():
    """documents current behavior."""
    df = pd.DataFrame({"v": [1, 2]}, index=["a", "b"])
    assert cu.remove_rows_by_index(df, "1") is df
