# -*- coding: utf-8 -*-
"""Tests for ade_dls/analysis/clustering.py (spec: _dev/notes/test_spec_clustering.md).

Categories: test_oracle_* (closed-form expectations), test_meta_* (metamorphic),
test_edge_* (edge cases), test_char_* (characterization: documents current
behavior, not desired behavior), test_finding_* (xfail strict, known findings).
"""
import copy
import inspect
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from scipy.cluster.hierarchy import fcluster, linkage  # noqa: E402
from scipy.stats import norm  # noqa: E402

from ade_dls.analysis import clustering as cl  # noqa: E402

TOL = dict(rel=1e-6, abs=1e-12)
REL = dict(rel=1e-6, abs=0)  # for tiny values (e.g. D_t ~ 1e-18)


@pytest.fixture(autouse=True)
def _restore_pyplot_state():
    prev = plt.isinteractive()
    yield
    plt.interactive(prev)
    plt.close("all")


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def make_df(q2, logD, filenames=None, extra=None):
    n = len(q2)
    q = np.asarray(q2, dtype=float)
    d = {"filename": list(filenames) if filenames is not None else [f"f{i}" for i in range(n)],
         "q^2": q}
    for col, vals in logD.items():
        v = np.array([np.nan if x is None else x for x in vals], dtype=float)
        d[col] = q * 10.0 ** v
    if extra:
        d.update(extra)
    return pd.DataFrame(d)


def run(df, cols, **kw):
    args = dict(q_squared_col="q^2", normalize_by_q2=True, plot=False, interactive=False)
    args.update(kw)
    return cl.cluster_all_gammas(df, list(cols), **args)


def pop_by_D(info):
    return {i + 1: d for i, d in enumerate(info["population_D_means"])}


def lo_hi(info):
    D = np.array(info["population_D_means"])
    return int(np.argmin(D)) + 1, int(np.argmax(D)) + 1


def partition(info):
    g = info["gammas_df"]
    return frozenset(
        frozenset(zip(sub["filename"], sub["original_col"])) for _, sub in g.groupby("cluster")
    )


def labels_of(info):
    return {(r.file_idx, r.original_col): r.cluster for r in info["gammas_df"].itertuples()}


def gm(x):
    return float(np.exp(np.mean(np.log(np.asarray(x, dtype=float)))))


def by_rank(clustered, info):
    order = np.argsort(info["population_D_means"])
    return np.column_stack(
        [clustered[f"gamma_pop{info['reliable_populations'][o]}"].to_numpy(float) for o in order]
    )


def multiset(info):
    return sorted(zip(info["population_D_means"], info["population_abundances"]))


# --- datasets ---------------------------------------------------------------
Q2_O1 = [1.0, 2.0, 3.0, 4.0]
VL = [0.00, 0.01, 0.02, 0.03]
VH = [0.80, 0.81, 0.82, 0.83]
COLS2 = ["gamma_1", "gamma_2"]


def d_o1():
    return make_df(Q2_O1, {"gamma_1": VL, "gamma_2": VH})


def d_o1_peaks():
    df = d_o1()
    df["tau_1"] = 1.0 / df["gamma_1"]
    df["tau_2"] = 1.0 / df["gamma_2"]
    df["intensity_1"] = [0.6] * 4
    df["intensity_2"] = [0.4] * 4
    df["skewness_1"] = [0.1, 0.2, 0.3, 0.4]
    df["skewness_2"] = [1.0, 1.0, 2.0, 2.0]
    return df


def d_o1x2():
    a = d_o1()
    b = d_o1()
    b["filename"] = [f"f{i}" for i in range(4, 8)]
    return pd.concat([a, b], ignore_index=True)


def d_ord():
    return make_df(Q2_O1, {"gamma_1": VL, "gamma_2": [0.8] * 4})


V_SINGLE = [0.00, 0.02, 0.05, 0.03, 0.01]


def d_single():
    return make_df([1, 2, 3, 4, 5], {"gamma_1": V_SINGLE})


def d_out():
    return make_df(Q2_O1 + [5.0], {"gamma_1": VL + [2.0], "gamma_2": VH + [None]})


V_REF = [0.0, 0.25, 0.5, 0.75, 1.1, 1.1]


def d_ref():
    return make_df([1.0] * 6, {"gamma_1": V_REF})


def r1_spec():
    rng = np.random.default_rng(20261103)
    q2 = np.linspace(1.0, 4.0, 12)
    v1 = rng.normal(0.0, 0.03, 12)
    v2 = rng.normal(0.7, 0.03, 12)
    v3 = rng.normal(1.6, 0.03, 3)
    g3 = [None] * 12
    for k, i in enumerate((0, 4, 8)):
        g3[i] = v3[k]
    return q2, {"gamma_1": list(v1), "gamma_2": list(v2), "gamma_3": g3}, v1, v2, v3


R1COLS = ["gamma_1", "gamma_2", "gamma_3"]


def d_r1():
    q2, lg, _, _, _ = r1_spec()
    return make_df(q2, lg)


def r1_run(df, cols=R1COLS, **kw):
    args = dict(distance_threshold=1.0, min_abundance=0.3, clustering_strategy="silhouette_refined")
    args.update(kw)
    return run(df, cols, **args)


def d_q(N):
    x = 0.15 * norm.ppf((np.arange(N) + 0.5) / N)
    df = pd.DataFrame({"filename": [f"q{i}" for i in range(N)], "q^2": 1.0, "gamma_1": 10.0 ** x})
    return df, x


S_O1 = float(np.mean([0.795 / 0.815, 1 - (0.04 / 3) / 0.805, 1 - (0.04 / 3) / 0.795, 0.765 / 0.785]))


def o01(df=None, **kw):
    args = dict(method="hierarchical", distance_threshold=1.0, min_abundance=0.3,
                clustering_strategy="simple")
    args.update(kw)
    return run(d_o1() if df is None else df, COLS2, **args)


# ----------------------------------------------------------------------------
# Oracle: cluster_all_gammas
# ----------------------------------------------------------------------------
def _check_o01_stats(clustered, info):
    lo, hi = lo_hi(info)
    vL, vH = np.array(VL), np.array(VH)
    q = np.array(Q2_O1)
    D = info["population_D_means"]
    assert D[lo - 1] == pytest.approx(10 ** 0.015, **TOL)
    assert D[hi - 1] == pytest.approx(10 ** 0.815, **TOL)
    assert info["population_means"][lo - 1] == pytest.approx(24 ** 0.25 * 10 ** 0.015, **TOL)
    assert info["population_means"][hi - 1] == pytest.approx(24 ** 0.25 * 10 ** 0.815, **TOL)
    assert info["population_D_stds"][lo - 1] == pytest.approx(np.std(10 ** vL), **TOL)
    assert info["population_D_stds"][hi - 1] == pytest.approx(np.std(10 ** vH), **TOL)
    assert info["population_stds"][lo - 1] == pytest.approx(np.std(q * 10 ** vL), **TOL)
    assert info["population_stds"][hi - 1] == pytest.approx(np.std(q * 10 ** vH), **TOL)
    st = info["population_stats"]
    assert st["n_gammas"].tolist() == [4, 4]
    assert st["n_files"].tolist() == [4, 4]
    df = d_o1()
    np.testing.assert_array_equal(clustered[f"gamma_pop{lo}"].to_numpy(float), df["gamma_1"].to_numpy())
    np.testing.assert_array_equal(clustered[f"gamma_pop{hi}"].to_numpy(float), df["gamma_2"].to_numpy())


def test_oracle_O01_two_groups_simple():
    clustered, info = o01()
    assert info["n_populations"] == 2
    assert info["reliable_populations"] == [1, 2]
    assert info["n_outlier_points"] == 0
    assert info["population_abundances"] == [1.0, 1.0]
    _check_o01_stats(clustered, info)
    for c in ("outlier_pop1", "outlier_pop2"):
        assert clustered[c].dtype == "boolean"
        assert clustered[c].tolist() == [False] * 4
    assert info["silhouette_score"] is None
    expected_keys = {"n_populations", "population_abundances", "population_means", "population_stds",
                     "silhouette_score", "reliable_populations", "population_stats",
                     "clustering_enabled", "clustering_strategy", "uncertainty_flags_enabled",
                     "n_outlier_points", "gammas_df", "cluster_id_to_pop", "clustering_plot",
                     "population_D_means", "population_D_stds",
                     "n_dropped_points", "population_n_dropped", "population_stats_pre_refinement",
                     "n_reassigned", "silhouette_before_refinement", "silhouette_after_refinement",
                     "populations_below_min_abundance"}
    assert set(info) == expected_keys
    assert info["populations_below_min_abundance"] == []
    assert info["n_reassigned"] == 0
    assert info["silhouette_before_refinement"] is None
    assert info["silhouette_after_refinement"] is None


def test_oracle_O02_silhouette_refined():
    clustered, info = o01(clustering_strategy="silhouette_refined", silhouette_threshold=0.3)
    assert info["n_populations"] == 2
    _check_o01_stats(clustered, info)
    assert S_O1 == pytest.approx(0.979162, abs=1e-6)
    assert info["silhouette_score"] == pytest.approx(S_O1, **TOL)
    assert info["n_reassigned"] == 0
    assert info["silhouette_before_refinement"] == pytest.approx(S_O1, **TOL)
    assert info["silhouette_after_refinement"] == pytest.approx(S_O1, **TOL)
    assert info["population_stats_pre_refinement"].equals(info["population_stats"])


def test_oracle_O03_gold_threshold_single_population():
    clustered, info = o01(clustering_strategy="silhouette_refined", distance_threshold=2.0)
    assert info["n_populations"] == 1
    assert info["population_D_means"] == pytest.approx([10 ** 0.415], **TOL)
    assert info["population_abundances"] == [1.0]
    assert info["population_stats"]["n_gammas"].tolist() == [8]
    assert info["silhouette_score"] is None
    assert info["n_reassigned"] == 0
    assert info["silhouette_before_refinement"] is None
    assert info["silhouette_after_refinement"] is None
    np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), d_o1()["gamma_1"].to_numpy())


@pytest.mark.parametrize("t", [0.3, 2.0])
def test_oracle_O04_single_population(t):
    df = d_single()
    clustered, info = run(df, ["gamma_1"], distance_threshold=t, min_abundance=0.3,
                          clustering_strategy="silhouette_refined")
    assert info["n_populations"] == 1
    assert info["population_D_means"] == pytest.approx([10 ** 0.022], **TOL)
    assert info["population_means"] == pytest.approx([120 ** 0.2 * 10 ** 0.022], **TOL)
    assert info["population_abundances"] == [1.0]
    assert info["cluster_id_to_pop"] == {0: 1}
    np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), df["gamma_1"].to_numpy())
    assert info["silhouette_score"] is None
    v = np.array(V_SINGLE)
    assert math.sqrt(10) * np.std(v) < 0.3


def test_oracle_O05_zscore_uncertainty_single():
    df = d_single()
    clustered, info = run(df, ["gamma_1"], distance_threshold=2.0, min_abundance=0.3,
                          clustering_strategy="silhouette_refined",
                          uncertainty_flags=True, uncertainty_threshold=0.5)
    v = np.array(V_SINGLE)
    expected = np.abs(v - 0.022) / math.sqrt(0.000296)
    assert expected == pytest.approx([1.278724, 0.116248, 1.627467, 0.464991, 0.697486], abs=1e-6)
    assert info["gammas_df"]["z_score"].to_numpy() == pytest.approx(expected, **TOL)
    assert clustered["uncertain_pop1"].astype(bool).tolist() == [True, False, True, False, True]
    np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), df["gamma_1"].to_numpy())


def test_oracle_D05_zscore_threshold_own_parameter_overrides_legacy():
    """D-05: z-score (single population) has its own threshold; legacy uncertainty_threshold=0.5
    is overridden by uncertainty_threshold_zscore."""
    kw = dict(distance_threshold=2.0, min_abundance=0.3, clustering_strategy="silhouette_refined",
              uncertainty_flags=True)
    clustered, info = run(d_single(), ["gamma_1"], uncertainty_threshold=0.5,
                          uncertainty_threshold_zscore=1.0, **kw)
    # |z| = [1.2787, 0.1162, 1.6275, 0.4650, 0.6975]  -> > 1.0: points 0 and 2 only
    assert clustered["uncertain_pop1"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    # the silhouette threshold is irrelevant for one population
    c2, _ = run(d_single(), ["gamma_1"], uncertainty_threshold_zscore=1.0,
                uncertainty_threshold_silhouette=99.0, **kw)
    assert c2["uncertain_pop1"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]
    # legacy parameter alone still drives both (backwards compatible)
    c3, _ = run(d_single(), ["gamma_1"], uncertainty_threshold=1.0, **kw)
    assert c3["uncertain_pop1"].tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_oracle_D05_silhouette_threshold_own_parameter_overrides_legacy():
    """D-05: silhouette (several populations) has its own threshold. d_ref after refinement has
    silhouettes [0.6186, 0.6591, 0.2241, 0.3000, 0.7941, 0.7941] (see C05)."""
    kw = dict(method="simple", distance_threshold=0.3, min_abundance=0.3,
              clustering_strategy="silhouette_refined", silhouette_threshold=0.3,
              uncertainty_flags=True)
    c_a, _ = run(d_ref(), ["gamma_1"], uncertainty_threshold=0.5,
                 uncertainty_threshold_silhouette=0.25, uncertainty_threshold_zscore=9.0, **kw)
    # JADE 3.1: pop1 = fastest (high D) population = rows 3-5, pop2 = rows 0-2
    assert c_a["uncertain_pop2"].dropna().tolist() == [0.0, 0.0, 1.0]
    assert c_a["uncertain_pop1"].dropna().tolist() == [0.0, 0.0, 0.0]
    c_b, _ = run(d_ref(), ["gamma_1"], uncertainty_threshold=0.25, **kw)
    assert c_b["uncertain_pop2"].dropna().tolist() == [0.0, 0.0, 1.0]
    c_c, _ = run(d_ref(), ["gamma_1"], uncertainty_threshold=0.5, **kw)
    assert c_c["uncertain_pop2"].dropna().tolist() == [0.0, 0.0, 1.0]
    assert c_c["uncertain_pop1"].dropna().tolist() == [1.0, 0.0, 0.0]   # silhouette 0.3 < 0.5


def test_oracle_D06_population_below_min_abundance_after_refinement_kept_and_flagged(monkeypatch):
    """D-06 (decided): a population that falls below min_abundance through refinement is kept
    but flagged in cluster_info['populations_below_min_abundance'] (numbers, 1-based)."""
    df = make_df([1.0] * 6, {"gamma_1": [0.0, 0.05, 0.1, 0.15, 1.0, 1.05]})

    def fake_refine(gdf, threshold, normalize_by_q2):
        # JADE 3.1: cluster 0 = fastest (high D, 2 points); move one of them to cluster 1
        idx = gdf.index[gdf["cluster"] == 0][0]
        gdf.loc[idx, "cluster"] = 1
        return 1

    monkeypatch.setattr(cl, "_refine_clusters_with_silhouette", fake_refine)
    clustered, info = run(df, ["gamma_1"], method="simple", distance_threshold=0.3,
                          min_abundance=0.3, clustering_strategy="silhouette_refined")
    assert info["n_reassigned"] == 1
    assert info["population_stats_pre_refinement"]["abundance"].tolist() == pytest.approx(
        [2 / 6, 4 / 6], **TOL)
    assert info["n_populations"] == 2                     # kept, not dropped
    assert info["population_abundances"] == pytest.approx([1 / 6, 5 / 6], **TOL)
    assert info["populations_below_min_abundance"] == [1]
    assert clustered["gamma_pop1"].notna().sum() == 1


def test_oracle_D06_no_flag_when_all_populations_stay_above_min_abundance():
    _, info = _ref_call()
    assert info["populations_below_min_abundance"] == []


def test_oracle_O06_outlier_population():
    clustered, info = run(d_out(), COLS2, distance_threshold=1.0, min_abundance=0.3)
    assert info["n_populations"] == 2
    assert info["n_outlier_points"] == 1
    assert info["population_abundances"] == pytest.approx([0.8, 0.8], **TOL)
    for p in (1, 2):
        assert np.isnan(clustered.loc[4, f"gamma_pop{p}"])
        assert clustered.loc[4, f"outlier_pop{p}"] == True  # noqa: E712
        assert clustered.loc[:3, f"outlier_pop{p}"].tolist() == [False] * 4
    assert len(info["gammas_df"]) == 8
    assert sorted(info["population_D_means"]) == pytest.approx(
        sorted([10 ** 0.015, 10 ** 0.815]), **TOL)


def test_oracle_O07_random_two_populations():
    q2, lg, v1, v2, v3 = r1_spec()
    df = d_r1()
    clustered, info = r1_run(df, clustering_strategy="simple")
    assert info["n_populations"] == 2 and info["n_outlier_points"] == 3
    assert info["population_abundances"] == pytest.approx([1.0, 1.0], **TOL)
    D = sorted(info["population_D_means"])
    assert D == pytest.approx(sorted([10 ** np.mean(v1), 10 ** np.mean(v2)]), **TOL)
    lo, hi = lo_hi(info)
    np.testing.assert_array_equal(clustered[f"gamma_pop{lo}"].to_numpy(float), df["gamma_1"].to_numpy())
    np.testing.assert_array_equal(clustered[f"gamma_pop{hi}"].to_numpy(float), df["gamma_2"].to_numpy())


@pytest.mark.parametrize("ma, npop, nout", [(0.0, 3, 0), (0.25, 3, 0), (0.26, 2, 3), (1.0, 2, 3), (1.01, 0, 27)])
def test_oracle_O08_min_abundance_sweep(ma, npop, nout):
    _, info = r1_run(d_r1(), clustering_strategy="simple", min_abundance=ma)
    assert info["n_populations"] == npop
    assert info["n_outlier_points"] == nout


def test_oracle_O09_n_clusters_override():
    clustered, info = o01(n_clusters=2, distance_threshold=0.001, clustering_strategy="silhouette_refined")
    assert info["n_populations"] == 2
    _check_o01_stats(clustered, info)
    clustered1, info1 = o01(n_clusters=1, distance_threshold=0.001, clustering_strategy="silhouette_refined")
    assert info1["n_populations"] == 1
    assert info1["population_D_means"] == pytest.approx([10 ** 0.415], **TOL)


def test_oracle_O10_n_clusters_random():
    q2, lg, v1, v2, v3 = r1_spec()
    df = d_r1()
    _, info3 = r1_run(df, clustering_strategy="simple", n_clusters=3)
    assert info3["n_populations"] == 2 and info3["n_outlier_points"] == 3
    c7, i7 = r1_run(df, clustering_strategy="simple")
    assert info3["population_D_means"] == pytest.approx(i7["population_D_means"], **TOL)
    c3, _ = r1_run(df, clustering_strategy="simple", n_clusters=3)
    for p in (1, 2):
        np.testing.assert_array_equal(c3[f"gamma_pop{p}"].to_numpy(float), c7[f"gamma_pop{p}"].to_numpy(float))
    clustered, info2 = r1_run(df, clustering_strategy="simple", n_clusters=2)
    assert info2["n_populations"] == 2
    assert info2["n_outlier_points"] == 0
    assert info2["population_abundances"] == pytest.approx([1.0, 1.0], **TOL)
    st = info2["population_stats"]
    k = int(st.index[st["n_gammas"] == 15][0])
    assert info2["population_D_means"][k] == pytest.approx(10 ** np.mean(np.concatenate([v2, v3])), **TOL)
    np.testing.assert_array_equal(clustered[f"gamma_pop{k + 1}"].to_numpy(float), df["gamma_2"].to_numpy())


def test_oracle_O11_simple_method_sorted_numbering():
    """JADE 3.1: descending mean D, pop1 = fastest."""
    clustered, info = run(d_ord(), COLS2, method="simple", distance_threshold=0.3, min_abundance=0.3)
    assert info["n_populations"] == 2
    assert info["population_D_means"] == pytest.approx([10 ** 0.8, 10 ** 0.015], **TOL)
    df = d_ord()
    np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), df["gamma_2"].to_numpy())
    np.testing.assert_array_equal(clustered["gamma_pop2"].to_numpy(float), df["gamma_1"].to_numpy())


def test_oracle_O12_peak_prefix_columns():
    df = d_o1_peaks()
    clustered, info = o01(df)
    new = set(clustered.columns) - set(df.columns)
    assert new == {"gamma_pop1", "gamma_pop2", "outlier_pop1", "outlier_pop2",
                   "tau_pop1", "tau_pop2", "intensity_pop1", "intensity_pop2",
                   "skewness_pop1", "skewness_pop2"}
    lo, hi = lo_hi(info)
    np.testing.assert_allclose(clustered[f"tau_pop{lo}"], df["tau_1"], rtol=1e-6)
    np.testing.assert_allclose(clustered[f"tau_pop{hi}"], df["tau_2"], rtol=1e-6)
    np.testing.assert_allclose(clustered[f"intensity_pop{lo}"], 0.6, rtol=1e-6)
    np.testing.assert_allclose(clustered[f"intensity_pop{hi}"], 0.4, rtol=1e-6)
    np.testing.assert_allclose(clustered[f"skewness_pop{lo}"], df["skewness_1"], rtol=1e-6)
    np.testing.assert_allclose(clustered[f"skewness_pop{hi}"], df["skewness_2"], rtol=1e-6)
    pd.testing.assert_frame_equal(clustered[df.columns], df)


def test_oracle_O13_quantile_sample_39_single_population():
    df, x = d_q(39)
    assert np.sqrt(2 * 39) * np.std(x) < 2.0
    _, info = run(df, ["gamma_1"], distance_threshold=2.0, min_abundance=0.0,
                  clustering_strategy="simple")
    assert info["n_populations"] == 1


def test_oracle_O14_min_abundance_equality():
    _, info = run(d_out(), COLS2, distance_threshold=1.0, min_abundance=0.2)
    assert info["n_populations"] == 3
    assert info["n_outlier_points"] == 0
    assert sorted(info["population_abundances"]) == pytest.approx([0.2, 0.8, 0.8], **TOL)


def test_oracle_O15_uncertainty_flags_multi():
    clustered, info = o01(clustering_strategy="silhouette_refined", uncertainty_flags=True)
    for p in (1, 2):
        assert not clustered[f"uncertain_pop{p}"].astype(bool).any()
    _check_o01_stats(clustered, info)


# ----------------------------------------------------------------------------
# Metamorphic
# ----------------------------------------------------------------------------
def test_meta_M01_row_permutation():
    df = d_r1()
    c0, i0 = r1_run(df)
    perm = df.sample(frac=1, random_state=7)
    c1, i1 = r1_run(perm)
    assert partition(i0) == partition(i1)
    assert i0["n_populations"] == i1["n_populations"]
    assert i0["n_outlier_points"] == i1["n_outlier_points"]
    assert i1["silhouette_score"] == pytest.approx(i0["silhouette_score"], **TOL)
    m0, m1 = multiset(i0), multiset(i1)
    assert np.array(m1) == pytest.approx(np.array(m0), **TOL)
    np.testing.assert_allclose(by_rank(c1.sort_index(), i1), by_rank(c0, i0), rtol=1e-6)
    assert i0["reliable_populations"] == i1["reliable_populations"]
    for p in i0["reliable_populations"]:
        np.testing.assert_allclose(c1.sort_index()[f"gamma_pop{p}"].to_numpy(float),
                                   c0[f"gamma_pop{p}"].to_numpy(float), rtol=1e-6)


def test_meta_M02_scale_gamma():
    c = 7.3
    df = d_r1()
    c0, i0 = r1_run(df)
    df2 = df.copy()
    for g in R1COLS:
        df2[g] = df[g] * c
    c1, i1 = r1_run(df2)
    assert partition(i0) == partition(i1)
    assert i0["cluster_id_to_pop"] == i1["cluster_id_to_pop"]
    assert i1["population_D_means"] == pytest.approx([c * v for v in i0["population_D_means"]], **TOL)
    assert i1["population_means"] == pytest.approx([c * v for v in i0["population_means"]], **TOL)
    assert i1["population_stds"] == pytest.approx([c * v for v in i0["population_stds"]], **TOL)
    assert i1["population_D_stds"] == pytest.approx([c * v for v in i0["population_D_stds"]], **TOL)
    assert i1["population_abundances"] == pytest.approx(i0["population_abundances"], **TOL)
    assert i1["silhouette_score"] == pytest.approx(i0["silhouette_score"], **TOL)
    for p in i0["reliable_populations"]:
        np.testing.assert_allclose(c1[f"gamma_pop{p}"].to_numpy(float),
                                   c * c0[f"gamma_pop{p}"].to_numpy(float), rtol=1e-6)


def test_meta_M03_scale_gamma_and_q2():
    c = 7.3
    df = d_r1()
    c0, i0 = r1_run(df)
    df2 = df.copy()
    for g in R1COLS:
        df2[g] = df[g] * c
    df2["q^2"] = df["q^2"] * c
    c1, i1 = r1_run(df2)
    assert partition(i0) == partition(i1)
    assert i0["cluster_id_to_pop"] == i1["cluster_id_to_pop"]
    assert i1["population_D_means"] == pytest.approx(i0["population_D_means"], **TOL)
    assert i1["population_D_stds"] == pytest.approx(i0["population_D_stds"], **TOL)
    assert i1["population_means"] == pytest.approx([c * v for v in i0["population_means"]], **TOL)
    for p in i0["reliable_populations"]:
        np.testing.assert_allclose(c1[f"gamma_pop{p}"].to_numpy(float),
                                   c * c0[f"gamma_pop{p}"].to_numpy(float), rtol=1e-6)
    assert i1["population_stds"] == pytest.approx([c * v for v in i0["population_stds"]], **TOL)


def test_meta_M04_scale_q2():
    c = 7.3
    df = d_r1()
    c0, i0 = r1_run(df)
    df2 = df.copy()
    df2["q^2"] = df["q^2"] * c
    c1, i1 = r1_run(df2)
    assert partition(i0) == partition(i1)
    assert i0["cluster_id_to_pop"] == i1["cluster_id_to_pop"]
    assert i1["population_D_means"] == pytest.approx([v / c for v in i0["population_D_means"]], **TOL)
    for p in i0["reliable_populations"]:
        np.testing.assert_allclose(c1[f"gamma_pop{p}"].to_numpy(float),
                                   c0[f"gamma_pop{p}"].to_numpy(float), rtol=1e-6)
    assert i1["population_means"] == pytest.approx(i0["population_means"], **TOL)


def test_meta_M05_power_transform():
    lam = 2.5
    q2, lg, _, _, _ = r1_spec()
    base = make_df(q2, lg)
    scaled = make_df(q2, {k: [None if x is None else lam * x for x in v] for k, v in lg.items()})
    c0, i0 = r1_run(base)
    c1, i1 = r1_run(scaled, distance_threshold=lam * 1.0)
    assert partition(i0) == partition(i1)
    assert i0["cluster_id_to_pop"] == i1["cluster_id_to_pop"]
    assert i1["silhouette_score"] == pytest.approx(i0["silhouette_score"], **TOL)
    assert i1["population_D_means"] == pytest.approx([v ** lam for v in i0["population_D_means"]], **TOL)
    for p in i0["reliable_populations"]:
        assert (c0[f"gamma_pop{p}"].notna().tolist() == c1[f"gamma_pop{p}"].notna().tolist())
        g0 = c0[f"gamma_pop{p}"].to_numpy(float)
        g1 = c1[f"gamma_pop{p}"].to_numpy(float)
        np.testing.assert_allclose(np.log10(g1 / q2), lam * np.log10(g0 / q2), rtol=1e-6, atol=1e-12)


def test_meta_M06_normalize_with_unit_q2():
    df = make_df([1.0] * 4, {"gamma_1": VL, "gamma_2": VH})
    c1, i1 = run(df, COLS2, distance_threshold=1.0, min_abundance=0.3,
                 clustering_strategy="silhouette_refined", normalize_by_q2=True)
    c0, i0 = run(df, COLS2, distance_threshold=1.0, min_abundance=0.3,
                 clustering_strategy="silhouette_refined", normalize_by_q2=False)
    assert partition(i0) == partition(i1)
    assert i0["cluster_id_to_pop"] == i1["cluster_id_to_pop"]
    assert i0["population_means"] == pytest.approx(i1["population_means"], **TOL)
    assert i0["silhouette_score"] == pytest.approx(i1["silhouette_score"], **TOL)
    assert "population_D_means" in i1 and "population_D_means" not in i0


def test_meta_M07_duplication_simple_method():
    _, i0 = run(d_o1(), COLS2, method="simple", distance_threshold=0.3, min_abundance=0.3)
    c2, i2 = run(d_o1x2(), COLS2, method="simple", distance_threshold=0.3, min_abundance=0.3)
    assert i0["n_populations"] == i2["n_populations"] == 2
    assert i0["population_D_means"] == pytest.approx(i2["population_D_means"], **TOL)
    assert i0["population_D_stds"] == pytest.approx(i2["population_D_stds"], **TOL)
    assert i0["population_abundances"] == pytest.approx(i2["population_abundances"], **TOL)


def test_meta_M08_append_empty_row():
    df = d_r1()
    _, i0 = r1_run(df)
    extra = pd.DataFrame({"filename": ["extra"], "q^2": [5.0], "gamma_1": [np.nan],
                          "gamma_2": [np.nan], "gamma_3": [np.nan]})
    _, i1 = r1_run(pd.concat([df, extra], ignore_index=True))
    assert partition(i0) == partition(i1)
    assert i1["n_populations"] == i0["n_populations"]
    assert i1["population_abundances"] == pytest.approx([a * 12 / 13 for a in i0["population_abundances"]], **TOL)


@pytest.mark.parametrize("bad_q2", [0.0, -1.0, np.nan])
def test_meta_M09_append_row_invalid_q2(bad_q2):
    df = d_r1()
    _, i0 = r1_run(df)
    extra = pd.DataFrame({"filename": ["extra"], "q^2": [bad_q2], "gamma_1": [1.0],
                          "gamma_2": [5.0], "gamma_3": [np.nan]})
    _, i1 = r1_run(pd.concat([df, extra], ignore_index=True))
    assert partition(i0) == partition(i1)
    assert i1["n_populations"] == i0["n_populations"]
    assert i1["population_abundances"] == pytest.approx([a * 12 / 13 for a in i0["population_abundances"]], **TOL)


def _nested(fine, coarse):
    lf, lc = labels_of(fine), labels_of(coarse)
    assert lf.keys() == lc.keys()
    for cid in set(lf.values()):
        assert len({lc[k] for k, v in lf.items() if v == cid}) == 1


@pytest.mark.parametrize("method", ["hierarchical", "simple"])
def test_meta_M10_threshold_monotone_nested(method):
    ts = [0.01, 0.05, 0.3, 1.0, 2.1, 5.0]
    infos = [r1_run(d_r1(), clustering_strategy="simple", method=method,
                    min_abundance=0.0, distance_threshold=t)[1] for t in ts]
    ns = [i["n_populations"] for i in infos]
    assert all(a >= b for a, b in zip(ns, ns[1:]))
    for a, b in zip(infos, infos[1:]):
        _nested(a, b)
    assert ns[-1] == 1
    if method == "hierarchical":
        assert ns[ts.index(1.0)] == 3


def test_meta_M11_min_abundance_monotone():
    mas = [0.0, 0.1, 0.25, 0.26, 0.5, 1.0, 1.01]
    infos = [r1_run(d_r1(), clustering_strategy="simple", min_abundance=m)[1] for m in mas]
    npop = [i["n_populations"] for i in infos]
    nout = [i["n_outlier_points"] for i in infos]
    assert all(a >= b for a, b in zip(npop, npop[1:]))
    assert all(a <= b for a, b in zip(nout, nout[1:]))


def test_meta_M12_reorder_gamma_cols():
    df = d_r1()
    c0, i0 = r1_run(df)
    c1, i1 = r1_run(df, cols=["gamma_3", "gamma_1", "gamma_2"])
    assert partition(i0) == partition(i1)
    assert i0["n_populations"] == i1["n_populations"]
    assert np.array(multiset(i1)) == pytest.approx(np.array(multiset(i0)), **TOL)
    np.testing.assert_allclose(by_rank(c1, i1), by_rank(c0, i0), rtol=1e-6)


def test_meta_M13_index_shift():
    df = d_r1()
    c0, i0 = r1_run(df)
    df2 = df.copy()
    df2.index = df2.index + 1
    c1, i1 = r1_run(df2)
    assert c1.index.tolist() == (df.index + 1).tolist()
    pd.testing.assert_frame_equal(c0.reset_index(drop=True), c1.reset_index(drop=True))
    assert i0["population_D_means"] == pytest.approx(i1["population_D_means"], **TOL)


def test_meta_M14_input_not_modified():
    df = d_r1()
    ref = copy.deepcopy(df)
    r1_run(df)
    assert df.equals(ref)
    df2 = d_o1()
    ref2 = copy.deepcopy(df2)
    cl.cluster_all_gammas(df2, COLS2, enable_clustering=False)
    assert df2.equals(ref2)


# ----------------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------------
NOVALID = "No valid gamma values found"


def test_edge_E01_all_nan():
    df = make_df(Q2_O1, {"gamma_1": [None] * 4, "gamma_2": [None] * 4})
    with pytest.raises(ValueError, match=NOVALID):
        run(df, COLS2, distance_threshold=1.0)


def test_edge_E02_all_nonpositive():
    df = d_o1()
    df["gamma_1"] = [0.0, -5.0, 0.0, -5.0]
    df["gamma_2"] = [-5.0, 0.0, -5.0, 0.0]
    with pytest.raises(ValueError, match=NOVALID):
        run(df, COLS2, distance_threshold=1.0)


def test_edge_E03_empty_gamma_cols():
    with pytest.raises(ValueError, match=NOVALID):
        run(d_o1(), [], distance_threshold=1.0)


def test_edge_E04_zero_rows():
    df = pd.DataFrame({"filename": [], "q^2": [], "gamma_1": []})
    with pytest.raises(ValueError, match=NOVALID):
        run(df, ["gamma_1"], distance_threshold=1.0)


def test_edge_E05_all_q2_invalid():
    df = d_o1()
    df["q^2"] = [0.0, -1.0, np.nan, 0.0]
    with pytest.raises(ValueError, match=NOVALID):
        run(df, COLS2, distance_threshold=1.0)


def test_edge_E06_missing_q_col_name():
    with pytest.raises(ValueError, match="q_squared_col must be provided"):
        run(d_o1(), COLS2, q_squared_col=None)


def test_edge_E07_unknown_q_col():
    with pytest.raises(ValueError, match="Column 'nope' not found"):
        run(d_o1(), COLS2, q_squared_col="nope")


def test_edge_E08_no_filename():
    with pytest.raises(ValueError, match="must have 'filename' column"):
        run(d_o1().drop(columns="filename"), COLS2)


def test_edge_E09_missing_gamma_col():
    with pytest.raises(ValueError, match="Column 'gamma_9' not found"):
        run(d_o1(), ["gamma_9"])


def test_edge_E10_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        run(d_o1(), COLS2, method="kmeans")


def test_edge_E11_no_reliable_population():
    df = d_o1()
    clustered, info = run(df, COLS2, distance_threshold=1.0, min_abundance=1.01)
    assert info["n_populations"] == 0
    assert info["reliable_populations"] == []
    assert info["population_abundances"] == []
    assert info["population_means"] == []
    assert info["cluster_id_to_pop"] == {}
    assert info["n_outlier_points"] == 8
    assert info["population_stats"].empty
    assert info["population_D_means"] == [] and info["population_D_stds"] == []   # F-17
    assert not any(c.startswith(("gamma_pop", "outlier_pop")) for c in clustered.columns)
    assert len(info["gammas_df"]) == 0
    assert info["population_n_dropped"] == [] and info["n_dropped_points"] == 0
    assert info["population_stats_pre_refinement"].empty
    assert info["n_reassigned"] == 0


def test_edge_E12_passthrough():
    df = pd.DataFrame({"a": [1, 2]})
    cols = ["gamma_1", "gamma_2"]
    out, info = cl.cluster_all_gammas(df, cols, enable_clustering=False)
    assert out is df
    assert info == {"n_populations": 2, "clustering_enabled": False,
                    "reliable_populations": [1, 2], "original_gamma_cols": cols}
    assert info["original_gamma_cols"] is cols


def test_edge_E13_single_point_simple_method():
    df = make_df([1.0], {"gamma_1": [1.0]})
    clustered, info = run(df, ["gamma_1"], method="simple", distance_threshold=0.3, min_abundance=0.3)
    assert info["n_populations"] == 1
    assert info["population_abundances"] == [1.0]
    assert clustered["gamma_pop1"].tolist() == pytest.approx([10.0], **TOL)


def test_edge_E14_mixed_invalid_values():
    df = d_o1()
    df.loc[0, "gamma_2"] = 0.0
    df.loc[1, "gamma_2"] = -3.0
    clustered, info = run(df, COLS2, distance_threshold=1.0, min_abundance=0.3)
    assert info["n_populations"] == 2
    lo, hi = lo_hi(info)
    assert info["population_stats"]["n_gammas"].tolist()[hi - 1] == 2
    assert info["population_abundances"][hi - 1] == pytest.approx(0.5, **TOL)
    col = clustered[f"gamma_pop{hi}"]
    assert col.isna().tolist() == [True, True, False, False]


def test_edge_E15_normalize_false():
    clustered, info = run(d_o1(), COLS2, normalize_by_q2=False, q_squared_col=None,
                          distance_threshold=1.0, min_abundance=0.3)
    assert "population_D_means" not in info
    assert info["gammas_df"]["D"].isna().all()


def test_edge_E16_unknown_strategy_behaves_like_simple():
    clustered, info = o01(clustering_strategy="foo")
    assert info["n_populations"] == 2
    assert info["silhouette_score"] is None


# ----------------------------------------------------------------------------
# Characterization
# ----------------------------------------------------------------------------
def test_char_C01_size_dependence_of_ward_cut_documents_current_behavior():
    """documents current behavior, not desired behavior (F-01)."""
    kw = dict(distance_threshold=2.0, min_abundance=0.3, clustering_strategy="silhouette_refined")
    _, i1 = run(d_o1(), COLS2, **kw)
    _, i2 = run(d_o1x2(), COLS2, **kw)
    assert i1["n_populations"] == 1
    assert i2["n_populations"] == 2
    h1 = linkage(i1["gammas_df"]["log_clustering_val"].to_numpy().reshape(-1, 1), "ward")[-1, 2]
    h2 = linkage(i2["gammas_df"]["log_clustering_val"].to_numpy().reshape(-1, 1), "ward")[-1, 2]
    assert h1 == pytest.approx(1.6, **TOL)
    assert h2 == pytest.approx(math.sqrt(8) * 0.8, **TOL)


def test_char_C02_size_dependence_quantile_sample_documents_current_behavior():
    """documents current behavior, not desired behavior (F-01)."""
    kw = dict(distance_threshold=2.0, min_abundance=0.0, clustering_strategy="simple")
    df39, _ = d_q(39)
    _, i39 = run(df39, ["gamma_1"], **kw)
    assert i39["n_populations"] == 1
    df400, _ = d_q(400)
    _, i400 = run(df400, ["gamma_1"], **kw)
    X = i400["gammas_df"]["log_clustering_val"].to_numpy().reshape(-1, 1)
    Z = linkage(X, "ward")
    assert Z[-1, 2] > 2.0
    assert i400["n_populations"] >= 2
    assert i400["n_populations"] == len(np.unique(fcluster(Z, 2.0, "distance")))


def test_oracle_C03_numbering_sorted_by_D():
    """JADE 3.1: populations are numbered by descending mean log10 D (pop1 = fastest = smallest particles)."""
    df = d_ord()
    for method, t in (("hierarchical", 1.0), ("simple", 0.3)):
        clustered, info = run(df, COLS2, method=method, distance_threshold=t, min_abundance=0.3)
        assert info["population_D_means"] == pytest.approx([10 ** 0.8, 10 ** 0.015], **TOL)
        np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), df["gamma_2"].to_numpy())
        np.testing.assert_array_equal(clustered["gamma_pop2"].to_numpy(float), df["gamma_1"].to_numpy())
        assert info["cluster_id_to_pop"] == {0: 1, 1: 2}


def _ascending_cases():
    return [
        ("O01", lambda: o01(), "population_D_means"),
        ("O01_refined", lambda: o01(clustering_strategy="silhouette_refined"), "population_D_means"),
        ("O07", lambda: r1_run(d_r1(), clustering_strategy="simple"), "population_D_means"),
        ("O10_n3", lambda: r1_run(d_r1(), clustering_strategy="simple", n_clusters=3), "population_D_means"),
        ("O10_n2", lambda: r1_run(d_r1(), clustering_strategy="simple", n_clusters=2), "population_D_means"),
        ("R1_all", lambda: r1_run(d_r1(), clustering_strategy="simple", min_abundance=0.0), "population_D_means"),
        ("OUT", lambda: run(d_out(), COLS2, distance_threshold=1.0, min_abundance=0.3), "population_D_means"),
        ("OUT_all", lambda: run(d_out(), COLS2, distance_threshold=1.0, min_abundance=0.0), "population_D_means"),
        ("ORD", lambda: run(d_ord(), COLS2, distance_threshold=1.0, min_abundance=0.3), "population_D_means"),
        ("REF", lambda: run(d_ref(), ["gamma_1"], distance_threshold=0.3, min_abundance=0.0), "population_D_means"),
        ("O01_noNorm", lambda: o01(normalize_by_q2=False, q_squared_col=None), "population_means"),
        ("ORD_noNorm", lambda: run(d_ord(), COLS2, normalize_by_q2=False, q_squared_col=None,
                                   distance_threshold=1.0, min_abundance=0.3), "population_means"),
    ]


@pytest.mark.parametrize("name, call, key", _ascending_cases(), ids=[c[0] for c in _ascending_cases()])
def test_oracle_numbering_ascending_in_D(name, call, key):
    """JADE 3.1: numbering is DESCENDING in mean D (Gamma if not normalized): pop1 = fastest."""
    _, info = call()
    vals = info[key]
    assert len(vals) >= 2
    assert all(a > b for a, b in zip(vals, vals[1:]))


def test_char_C04_first_come_documents_current_behavior():
    """documents current behavior, not desired behavior (F-05)."""
    df = d_o1_peaks()
    clustered, info = run(df, COLS2, distance_threshold=2.0, min_abundance=0.3)
    np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), df["gamma_1"].to_numpy())
    assert info["population_stats"]["n_gammas"].tolist() == [8]
    assert info["population_D_means"] == pytest.approx([10 ** 0.415], **TOL)
    assert gm(clustered["gamma_pop1"] / clustered["q^2"]) == pytest.approx(10 ** 0.015, **TOL)
    assert gm(clustered["gamma_pop1"] / clustered["q^2"]) != pytest.approx(info["population_D_means"][0], rel=1e-3)
    np.testing.assert_allclose(clustered["tau_pop1"], df["tau_1"], rtol=1e-6)
    np.testing.assert_allclose(clustered["skewness_pop1"], df["skewness_1"], rtol=1e-6)


def _ref_call():
    return run(d_ref(), ["gamma_1"], method="simple", distance_threshold=0.3, min_abundance=0.3,
               clustering_strategy="silhouette_refined", silhouette_threshold=0.3)


def test_oracle_C05_refinement_updates_statistics():
    """F-03 fixed: statistics are recomputed from the final assignment after refinement."""
    clustered, info = _ref_call()
    assert info["gammas_df"]["cluster"].tolist() == [1, 1, 1, 0, 0, 0]   # JADE 3.1: cluster 0 = fastest
    nan = np.nan
    # JADE 3.1: pop1 = fastest (B), pop2 = slowest (A)
    np.testing.assert_allclose(clustered["gamma_pop2"].to_numpy(float),
                               [1.0, 10 ** 0.25, 10 ** 0.5, nan, nan, nan], rtol=1e-6)
    np.testing.assert_allclose(clustered["gamma_pop1"].to_numpy(float),
                               [nan, nan, nan, 10 ** 0.75, 10 ** 1.1, 10 ** 1.1], rtol=1e-6)
    A = np.array([1.0, 10 ** 0.25, 10 ** 0.5])
    B = np.array([10 ** 0.75, 10 ** 1.1, 10 ** 1.1])
    assert info["population_abundances"] == pytest.approx([3 / 6, 3 / 6], **TOL)
    assert info["population_D_means"] == pytest.approx([10 ** ((0.75 + 2.2) / 3), 10 ** 0.25], **TOL)
    assert info["population_means"] == pytest.approx([gm(B), gm(A)], **TOL)
    assert info["population_D_stds"] == pytest.approx([np.std(B), np.std(A)], **TOL)
    assert info["population_stds"] == pytest.approx([np.std(B), np.std(A)], **TOL)
    st = info["population_stats"]
    assert st["n_gammas"].tolist() == [3, 3]
    assert st["n_files"].tolist() == [3, 3]
    assert info["reliable_populations"] == [1, 2]
    assert info["cluster_id_to_pop"] == {0: 1, 1: 2}
    expected = np.mean([1 - 0.375 / (2.95 / 3), 1 - 0.25 / (2.2 / 3), 1 - 0.375 / (1.45 / 3),
                        1 - 0.35 / 0.5, 1 - 0.175 / 0.85, 1 - 0.175 / 0.85])
    assert expected == pytest.approx(0.565018, abs=1e-6)
    assert info["silhouette_score"] == pytest.approx(expected, **TOL)
    assert info["silhouette_after_refinement"] == pytest.approx(expected, **TOL)
    before = np.mean([6 / 11, (0.85 - 1 / 3) / 0.85, (0.6 - 1 / 3) / 0.6, -0.3, 1, 1])
    assert info["silhouette_before_refinement"] == pytest.approx(before, **TOL)
    assert info["n_reassigned"] == 1
    pre = info["population_stats_pre_refinement"]
    assert pre["abundance"].tolist() == pytest.approx([2 / 6, 4 / 6], **TOL)
    assert pre["mean_D"].tolist() == pytest.approx([10 ** 1.1, 10 ** 0.375], **TOL)
    assert pre["n_gammas"].tolist() == [2, 4]


def test_char_C06_silhouette_only_when_needed_documents_current_behavior():
    """documents current behavior, not desired behavior (F-16)."""
    _, info = o01()
    assert info["n_populations"] == 2
    assert info["silhouette_score"] is None


def test_char_C07_signature_defaults_documents_current_behavior():
    """Parity guard for documented signature defaults (F-08)."""
    def defaults(f):
        return {k: v.default for k, v in inspect.signature(f).parameters.items()
                if v.default is not inspect.Parameter.empty}
    assert defaults(cl.cluster_all_gammas) == dict(
        q_squared_col=None, enable_clustering=True, normalize_by_q2=False, method="hierarchical",
        n_clusters="auto", distance_threshold=0.3, min_abundance=0.3, clustering_strategy="simple",
        silhouette_threshold=0.3, uncertainty_flags=False, uncertainty_threshold=0.5,
        plot=True, interactive=True, experiment_name="",
        uncertainty_threshold_zscore=None, uncertainty_threshold_silhouette=None)
    assert defaults(cl.clustering_parameter_sweep) == dict(
        q_squared_col="q^2", distance_thresholds=(0.3, 0.7, 1.0, 1.5, 2.0),
        min_abundances=(0.1, 0.2, 0.3, 0.4, 0.5), normalize_by_q2=True,
        clustering_strategy="silhouette_refined", method="hierarchical")
    assert defaults(cl.get_reliable_gamma_cols) == dict(prefix="gamma_pop")
    assert defaults(cl.aggregate_peak_stats) == dict(stat_prefixes=("skewness", "kurtosis"))


def test_char_C08_outlier_flag_semantics_documents_current_behavior():
    """documents current behavior, not desired behavior (F-20)."""
    clustered, info = run(d_out(), COLS2, distance_threshold=1.0, min_abundance=0.3)
    for p in (1, 2):
        out = clustered[f"outlier_pop{p}"]
        has = clustered[f"gamma_pop{p}"].notna()
        assert out[4] == True  # noqa: E712
        assert not (out[has] == True).any()  # noqa: E712
    # files without value and without outlier -> pd.NA
    df = d_o1()
    df.loc[0, "gamma_2"] = 0.0
    df.loc[1, "gamma_2"] = -3.0
    c2, i2 = run(df, COLS2, distance_threshold=1.0, min_abundance=0.3)
    lo, hi = lo_hi(i2)
    assert pd.isna(c2.loc[0, f"outlier_pop{hi}"]) and pd.isna(c2.loc[1, f"outlier_pop{hi}"])


def test_edge_C09_single_point_hierarchical_one_population():
    """F-10 fixed: one valid point with method='hierarchical' gives one population."""
    df = make_df([1.0], {"gamma_1": [1.0]})
    clustered, info = run(df, ["gamma_1"], method="hierarchical", distance_threshold=0.3)
    assert info["n_populations"] == 1
    assert info["population_abundances"] == [1.0]
    assert clustered["gamma_pop1"].tolist() == pytest.approx([10.0], **TOL)


def test_edge_H08_hierarchical_single_point_labels():
    labels, _ = cl._hierarchical_clustering(_gdf([0.5]), distance_threshold=0.3, plot=False)
    assert np.asarray(labels).tolist() == [0]


def test_char_C10_inf_gamma_raises_documents_current_behavior():
    """documents current behavior, not desired behavior (F-13)."""
    df = d_o1()
    df.loc[0, "gamma_1"] = np.inf
    with pytest.raises(ValueError):
        run(df, COLS2, distance_threshold=1.0)


def test_char_C11_non_unique_index_raises_documents_current_behavior():
    """documents current behavior, not desired behavior (F-11)."""
    df = d_o1()
    df.index = [0, 0, 1, 1]
    with pytest.raises(ValueError):
        run(df, COLS2, distance_threshold=1.0)


def test_char_C12_gamma_name_convention_documents_current_behavior():
    """documents current behavior, not desired behavior (F-12)."""
    df = d_o1().rename(columns={"gamma_1": "gamma_a", "gamma_2": "gamma_b"})
    with pytest.raises(ValueError):
        run(df, ["gamma_a", "gamma_b"], distance_threshold=1.0)


def test_char_C13_duplicate_filenames_lower_abundance_documents_current_behavior():
    """documents current behavior, not desired behavior (F-07)."""
    df = d_single()
    df["filename"] = ["same"] * 5
    _, info = run(df, ["gamma_1"], distance_threshold=2.0, min_abundance=0.3)
    assert info["n_populations"] == 0
    assert info["n_outlier_points"] == 5


@pytest.mark.parametrize("normalize", [True, False])
def test_char_C14_plot_returns_figure_without_pyplot_side_effects_documents_current_behavior(normalize):
    """documents current behavior, not desired behavior (F-18)."""
    df = make_df([1.0] * 4, {"gamma_1": VL, "gamma_2": VH}) if not normalize else d_o1()
    before = plt.get_fignums()
    _, info = run(df, COLS2, normalize_by_q2=normalize,
                  q_squared_col="q^2" if normalize else None,
                  distance_threshold=1.0, min_abundance=0.3, plot=True)
    assert isinstance(info["clustering_plot"], Figure)
    assert plt.get_fignums() == before
    _, info2 = run(df, COLS2, normalize_by_q2=normalize,
                   q_squared_col="q^2" if normalize else None,
                   distance_threshold=1.0, min_abundance=0.3, plot=False)
    assert info2["clustering_plot"] is None


def test_regression_F21_uncertainty_flags_no_crash():
    """Regression for F-21 (clustering.py:415, fixed): uncertain_popN is float."""
    clustered, _ = run(d_single(), ["gamma_1"], distance_threshold=2.0, min_abundance=0.3,
                       clustering_strategy="silhouette_refined",
                       uncertainty_flags=True, uncertainty_threshold=0.5)
    col = clustered["uncertain_pop1"]
    assert col.dtype == np.float64
    assert col.tolist() == [1.0, 0.0, 1.0, 0.0, 1.0]
    assert set(col.dropna().unique()) <= {0.0, 1.0}


def test_oracle_C15_D_keys_present_and_empty_without_populations():
    """F-17 fixed (D-10): with normalize_by_q2 the D keys always exist (empty lists without
    populations); without normalization they stay absent."""
    _, info = run(d_o1(), COLS2, distance_threshold=1.0, min_abundance=1.01)
    assert info["n_populations"] == 0
    assert info["population_D_means"] == [] and info["population_D_stds"] == []
    _, info2 = run(d_o1(), COLS2, normalize_by_q2=False, q_squared_col=None, distance_threshold=1.0)
    assert "population_D_means" not in info2


def test_oracle_F05_n_dropped_points():
    """F-05 stage 1: number of points not present in gamma_popN (first come)."""
    _, i2 = run(d_o1(), COLS2, distance_threshold=2.0, min_abundance=0.3)
    assert i2["population_n_dropped"] == [4]
    assert i2["n_dropped_points"] == 4
    _, i1 = run(d_o1(), COLS2, distance_threshold=1.0, min_abundance=0.3)
    assert i1["population_n_dropped"] == [0, 0]
    assert i1["n_dropped_points"] == 0
    _, ir = r1_run(d_r1())
    assert ir["population_n_dropped"] == [0, 0]
    assert ir["n_dropped_points"] == 0


def test_edge_F05_n_dropped_consistent_with_columns():
    clustered, info = run(d_o1(), COLS2, distance_threshold=2.0, min_abundance=0.3)
    n_pts = info["population_stats"]["n_gammas"].tolist()
    notna = [int(clustered[f"gamma_pop{p}"].notna().sum()) for p in info["reliable_populations"]]
    assert info["population_n_dropped"] == [a - b for a, b in zip(n_pts, notna)]
    assert isinstance(info["n_dropped_points"], int)


def test_edge_F05_passthrough_has_no_new_keys():
    _, info = cl.cluster_all_gammas(d_o1(), COLS2, enable_clustering=False)
    assert set(info) == {"n_populations", "clustering_enabled", "reliable_populations", "original_gamma_cols"}


def test_oracle_F05_warning_printed_when_dropped(capsys):
    run(d_o1(), COLS2, distance_threshold=2.0, min_abundance=0.3)
    lines = capsys.readouterr().out.splitlines()
    assert any(l.startswith("WARNING") and "dropped" in l for l in lines)
    run(d_o1(), COLS2, distance_threshold=1.0, min_abundance=0.3)
    lines = capsys.readouterr().out.splitlines()
    assert not any(l.startswith("WARNING") and "dropped" in l for l in lines)


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------
def _gdf(vals):
    return pd.DataFrame({"log_clustering_val": vals})


def test_oracle_H01_ward_heights_and_numbering():
    g = _gdf([0.0, 1.0, 3.0])
    labels, Z = cl._hierarchical_clustering(g, distance_threshold=2.0, plot=False)
    assert Z[:, 2] == pytest.approx([1.0, math.sqrt(25 / 3)], **TOL)
    assert labels.tolist() == [0, 0, 1]
    labels3, _ = cl._hierarchical_clustering(g, distance_threshold=3.0, plot=False)
    assert labels3.tolist() == [0, 0, 0]
    labels_n, _ = cl._hierarchical_clustering(g, n_clusters=2, plot=False)
    assert labels_n.tolist() == [0, 0, 1]


def test_edge_H02_threshold_equality_joins():
    g = _gdf([0.0, 0.5])
    l1, Z = cl._hierarchical_clustering(g, distance_threshold=0.5, plot=False)
    assert Z[0, 2] == pytest.approx(0.5, **TOL)
    assert len(np.unique(l1)) == 1
    l2, _ = cl._hierarchical_clustering(g, distance_threshold=0.4999, plot=False)
    assert len(np.unique(l2)) == 2


def test_oracle_H03_simple_threshold_labels():
    g = _gdf([0.5, 0.0, 1.2, 0.1, 0.45])
    assert cl._simple_threshold_clustering(g, distance_threshold=0.3).tolist() == [1, 0, 2, 0, 1]


def test_edge_H04_simple_gap_equal_threshold_not_split():
    g = _gdf([0.0, 0.25, 0.5])
    assert cl._simple_threshold_clustering(g, distance_threshold=0.25).tolist() == [0, 0, 0]
    assert cl._simple_threshold_clustering(g, distance_threshold=0.2499).tolist() == [0, 1, 2]


def test_oracle_H05_refine_reassigns():
    g = pd.DataFrame({"log_clustering_val": [0.0, 0.1, 0.9, 1.0, 1.1],
                      "cluster": [0, 0, 0, 1, 1],
                      "silhouette": [0.9, 0.9, -0.5, 0.9, 0.9]})
    n = cl._refine_clusters_with_silhouette(g, 0.3, True)
    assert n == 1
    assert g.loc[2, "cluster"] == 1
    assert g["cluster"].tolist() == [0, 0, 1, 1, 1]


def test_edge_H06_refine_single_cluster():
    g = pd.DataFrame({"log_clustering_val": [0.0, 0.1, 0.2], "cluster": [0, 0, 0],
                      "silhouette": [-1.0, -1.0, -1.0]})
    ref = g.copy()
    assert cl._refine_clusters_with_silhouette(g, 0.3, True) == 0
    pd.testing.assert_frame_equal(g, ref)


def test_oracle_H07_silhouette_scores():
    g3 = _gdf([0.0, 0.1, 0.2])
    np.testing.assert_array_equal(cl._calculate_silhouette_scores(g3, np.array([0, 0, 0])), np.zeros(3))
    np.testing.assert_array_equal(cl._calculate_silhouette_scores(g3, np.array([0, 1, 2])), np.zeros(3))
    g4 = _gdf([0.0, 0.01, 1.0, 1.01])
    s = cl._calculate_silhouette_scores(g4, np.array([0, 0, 1, 1]))
    a = 1 - 0.01 / 1.005
    b = 1 - 0.01 / 0.995
    assert s == pytest.approx([a, b, b, a], **TOL)


# ----------------------------------------------------------------------------
# get_reliable_gamma_cols
# ----------------------------------------------------------------------------
def test_oracle_G01_passthrough_returns_same_list():
    cols = ["gamma_1", "gamma_2"]
    _, info = cl.cluster_all_gammas(d_o1(), cols, enable_clustering=False)
    assert cl.get_reliable_gamma_cols(info) is cols


def test_oracle_G02_prefix():
    info = {"clustering_enabled": True, "reliable_populations": [1, 2]}
    assert cl.get_reliable_gamma_cols(info) == ["gamma_pop1", "gamma_pop2"]
    assert cl.get_reliable_gamma_cols(info, prefix="tau_pop") == ["tau_pop1", "tau_pop2"]


def test_oracle_G03_from_outlier_case():
    _, info = run(d_out(), COLS2, distance_threshold=1.0, min_abundance=0.3)
    assert cl.get_reliable_gamma_cols(info) == ["gamma_pop1", "gamma_pop2"]


def test_meta_G04_names_exist_and_match_count():
    cases = [
        (d_o1(), COLS2, dict(distance_threshold=1.0, min_abundance=0.3)),
        (d_o1(), COLS2, dict(distance_threshold=2.0, min_abundance=0.3)),
        (d_out(), COLS2, dict(distance_threshold=1.0, min_abundance=0.3)),
        (d_r1(), R1COLS, dict(distance_threshold=1.0, min_abundance=0.3)),
        (d_out(), COLS2, dict(distance_threshold=1.0, min_abundance=0.2)),
        (d_o1(), COLS2, dict(distance_threshold=1.0, min_abundance=1.01)),
    ]
    for df, cols, kw in cases:
        clustered, info = run(df, cols, **kw)
        names = cl.get_reliable_gamma_cols(info)
        assert all(n in clustered.columns for n in names)
        assert len(names) == info["n_populations"]


def test_edge_G05_fallback_dict_without_clustering_enabled():
    assert cl.get_reliable_gamma_cols({"n_populations": 0, "reliable_populations": []}) == []


def test_edge_G06_missing_reliable_populations_keyerror():
    with pytest.raises(KeyError):
        cl.get_reliable_gamma_cols({"clustering_enabled": True})


def test_edge_G07_disabled_without_cols():
    assert cl.get_reliable_gamma_cols({"clustering_enabled": False}) == []


def test_char_G08_alias_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    assert cl.get_gamma_cols_for_analysis is cl.get_reliable_gamma_cols


# ----------------------------------------------------------------------------
# aggregate_peak_stats
# ----------------------------------------------------------------------------
def _a01():
    df = d_o1_peaks()
    clustered, info = o01(df)
    return df, clustered, info


def test_oracle_A01_clustering_mode_stats():
    df, clustered, info = _a01()
    out = cl.aggregate_peak_stats(info, clustered)
    assert out is info
    lo, hi = lo_hi(info)
    m, s = info["population_skewness_mean"], info["population_skewness_std"]
    assert m[lo] == pytest.approx(0.25, **TOL)
    assert s[lo] == pytest.approx(np.std([0.1, 0.2, 0.3, 0.4]), **TOL)
    assert s[lo] == pytest.approx(0.111803, abs=1e-6)
    assert m[hi] == pytest.approx(1.5, **TOL)
    assert s[hi] == pytest.approx(0.5, **TOL)
    assert "population_kurtosis_mean" not in info and "population_kurtosis_std" not in info


def test_oracle_A02_passthrough_ddof0():
    """D-03 (decided): pass-through std uses np.std (ddof=0), like the clustering path."""
    df = d_o1_peaks()
    _, info = cl.cluster_all_gammas(df, COLS2, enable_clustering=False)
    cl.aggregate_peak_stats(info, df)
    assert info["population_skewness_mean"] == pytest.approx({1: 0.25, 2: 1.5}, **TOL)
    assert info["population_skewness_std"] == pytest.approx(
        {1: math.sqrt(0.05 / 4), 2: math.sqrt(1 / 4)}, **TOL)
    assert info["population_skewness_std"][2] == pytest.approx(0.5, **TOL)


def test_meta_A02b_passthrough_std_equals_clustering_path_std():
    """D-03: one convention. Single population holding exactly the same values in both modes."""
    df = d_o1_peaks()
    _, info_pt = cl.cluster_all_gammas(df, COLS2, enable_clustering=False)
    cl.aggregate_peak_stats(info_pt, df)
    vals = df["skewness_1"].dropna().to_numpy(float)
    assert info_pt["population_skewness_std"][1] == pytest.approx(np.std(vals), **TOL)


def test_char_A03_single_population_uses_all_points_documents_current_behavior():
    """documents current behavior, not desired behavior (F-05)."""
    df = d_o1_peaks()
    clustered, info = run(df, COLS2, distance_threshold=2.0, min_abundance=0.3)
    cl.aggregate_peak_stats(info, clustered)
    allv = [0.1, 0.2, 0.3, 0.4, 1.0, 1.0, 2.0, 2.0]
    assert info["population_skewness_mean"] == pytest.approx({1: 0.875}, **TOL)
    assert info["population_skewness_std"][1] == pytest.approx(math.sqrt(0.521875), **TOL)
    assert info["population_skewness_std"][1] == pytest.approx(np.std(allv), **TOL)
    assert clustered["skewness_pop1"].mean() == pytest.approx(0.25, **TOL)


def test_meta_A04_no_cluster_id_to_pop():
    df, clustered, info = _a01()
    ref = copy.deepcopy({k: info[k] for k in ("cluster_id_to_pop",)})
    cl.aggregate_peak_stats(info, clustered)
    m0, s0 = dict(info["population_skewness_mean"]), dict(info["population_skewness_std"])
    info["cluster_id_to_pop"] = None
    cl.aggregate_peak_stats(info, clustered)
    assert info["population_skewness_mean"] == pytest.approx(m0, **TOL)
    assert info["population_skewness_std"] == pytest.approx(s0, **TOL)
    assert ref["cluster_id_to_pop"]  # sanity: mapping existed before


def test_meta_A05_row_permutation():
    df = d_o1_peaks()
    c0, i0 = o01(df)
    cl.aggregate_peak_stats(i0, c0)
    c1, i1 = o01(df.sample(frac=1, random_state=7))
    cl.aggregate_peak_stats(i1, c1)

    def ms(i):
        return sorted((i["population_D_means"][p - 1], i["population_skewness_mean"][p],
                       i["population_skewness_std"][p]) for p in i["reliable_populations"])
    assert np.array(ms(i1)) == pytest.approx(np.array(ms(i0)), **TOL)


def test_edge_A06_no_stat_columns():
    clustered, info = o01()
    keys = set(info)
    assert cl.aggregate_peak_stats(info, clustered) is info
    assert set(info) == keys


def test_edge_A07_only_skewness_2():
    df = d_o1()
    df["skewness_2"] = [1.0, 1.0, 2.0, 2.0]
    clustered, info = o01(df)
    keys = set(info)
    cl.aggregate_peak_stats(info, clustered)
    assert set(info) == keys


def test_edge_A08_no_gammas_df():
    df, clustered, info = _a01()
    del info["gammas_df"]
    keys = set(info)
    assert cl.aggregate_peak_stats(info, clustered) is info
    assert set(info) == keys


def test_edge_A09_nan_skewness():
    df = d_o1_peaks()
    df["skewness_1"] = np.nan
    clustered, info = o01(df)
    cl.aggregate_peak_stats(info, clustered)
    lo, hi = lo_hi(info)
    assert np.isnan(info["population_skewness_mean"][lo])
    assert np.isnan(info["population_skewness_std"][lo])
    assert info["population_skewness_mean"][hi] == pytest.approx(1.5, **TOL)
    assert info["population_skewness_std"][hi] == pytest.approx(0.5, **TOL)
    # a single non-NaN value gives std NaN
    df2 = d_o1_peaks()
    df2["skewness_1"] = [np.nan, np.nan, np.nan, 0.4]
    c2, i2 = o01(df2)
    cl.aggregate_peak_stats(i2, c2)
    lo2, _ = lo_hi(i2)
    assert i2["population_skewness_mean"][lo2] == pytest.approx(0.4, **TOL)
    assert np.isnan(i2["population_skewness_std"][lo2])


def test_char_A10_passthrough_key_is_position_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    df = d_o1_peaks()
    _, info = cl.cluster_all_gammas(df, ["gamma_2"], enable_clustering=False)
    cl.aggregate_peak_stats(info, df)
    assert info["population_skewness_mean"] == pytest.approx({1: 1.5}, **TOL)
    _, info2 = cl.cluster_all_gammas(df, ["gamma_x"], enable_clustering=False)
    cl.aggregate_peak_stats(info2, df)
    assert info2["population_skewness_mean"] == {}


def test_char_A11_missing_clustering_enabled_is_passthrough_documents_current_behavior():
    """documents current behavior, not desired behavior (F-15)."""
    info = {"n_populations": 0, "reliable_populations": []}
    cl.aggregate_peak_stats(info, d_o1_peaks())
    assert info["population_skewness_mean"] == {}


# ----------------------------------------------------------------------------
# clustering_parameter_sweep
# ----------------------------------------------------------------------------
SUMMARY_COLS = (["distance_threshold", "min_abundance", "n_populations", "silhouette"]
                + [f"pop{i}_{s}" for i in range(1, 5) for s in ("D_mean", "D_std", "abund")]
                + ["error"])


def test_oracle_S01_sweep_two_thresholds():
    summ, scat = cl.clustering_parameter_sweep(d_o1(), COLS2, distance_thresholds=(1.0, 2.0),
                                               min_abundances=(0.3,))
    assert list(summ.columns) == SUMMARY_COLS and len(summ.columns) == 17
    assert summ["error"].isna().all()
    assert len(summ) == 2
    r1, r2 = summ.iloc[0], summ.iloc[1]
    assert (r1["distance_threshold"], r2["distance_threshold"]) == (1.0, 2.0)
    assert r1["n_populations"] == 2
    assert r1["silhouette"] == pytest.approx(S_O1, **TOL)
    pairs = sorted([(r1["pop1_D_mean"], r1["pop1_D_std"]), (r1["pop2_D_mean"], r1["pop2_D_std"])])
    assert pairs[0][0] == pytest.approx(10 ** 0.015, **TOL)
    assert pairs[1][0] == pytest.approx(10 ** 0.815, **TOL)
    assert pairs[0][1] == pytest.approx(np.std(10 ** np.array(VL)), **TOL)
    assert pairs[1][1] == pytest.approx(np.std(10 ** np.array(VH)), **TOL)
    assert np.isnan([r1["pop3_D_mean"], r1["pop4_D_mean"], r1["pop3_abund"], r1["pop4_abund"]]).all()
    assert (r1["pop1_abund"], r1["pop2_abund"]) == (1.0, 1.0)
    assert r2["n_populations"] == 1
    assert pd.isna(r2["silhouette"])
    assert r2["pop1_D_mean"] == pytest.approx(10 ** 0.415, **TOL)
    assert r2["pop1_abund"] == 1.0
    assert np.isnan([r2["pop2_D_mean"], r2["pop3_D_mean"], r2["pop4_D_mean"]]).all()

    assert list(scat.columns) == ["distance_threshold", "min_abundance", "population", "q^2", "D_t", "excluded"]
    assert len(scat) == 16
    s1 = scat[scat["distance_threshold"] == 1.0]
    assert len(s1) == 8 and not s1["excluded"].any()
    assert np.sort(s1["D_t"].to_numpy() * 1e18) == pytest.approx(np.sort(10 ** np.array(VL + VH)), **REL)
    s2 = scat[scat["distance_threshold"] == 2.0]
    inc, exc = s2[~s2["excluded"]], s2[s2["excluded"]]
    assert len(inc) == 4 and (inc["population"] == "gamma_pop1").all()
    assert np.sort(inc["D_t"].to_numpy() * 1e18) == pytest.approx(np.sort(10 ** np.array(VL)), **REL)
    assert len(exc) == 4 and (exc["population"] == "excluded").all()
    assert np.sort(exc["D_t"].to_numpy() * 1e18) == pytest.approx(np.sort(10 ** np.array(VH)), **REL)


def _s02():
    return cl.clustering_parameter_sweep(d_o1(), COLS2, distance_thresholds=(1.0, 2.0),
                                         min_abundances=(0.3, 1.01))


def test_oracle_S02_sweep_order_and_empty_populations():
    summ, scat = _s02()
    assert list(zip(summ["distance_threshold"], summ["min_abundance"])) == [
        (1.0, 0.3), (1.0, 1.01), (2.0, 0.3), (2.0, 1.01)]
    assert summ["n_populations"].tolist() == [2, 0, 1, 0]
    for i in (1, 3):
        row = summ.iloc[i]
        assert pd.isna(row["silhouette"])
        pop_cols = [c for c in SUMMARY_COLS if c.startswith("pop")]
        assert row[pop_cols].isna().all()
        sub = scat[(scat["distance_threshold"] == row["distance_threshold"]) & (scat["min_abundance"] == 1.01)]
        assert len(sub) == 8 and sub["excluded"].all()


def test_meta_S03_sweep_equals_direct_calls():
    summ, _ = _s02()
    for _, row in summ.iterrows():
        _, info = run(d_o1(), COLS2, distance_threshold=row["distance_threshold"],
                      min_abundance=row["min_abundance"], clustering_strategy="silhouette_refined",
                      method="hierarchical")
        assert row["n_populations"] == info["n_populations"]
        sil = info["silhouette_score"]
        if sil is None:
            assert pd.isna(row["silhouette"])
        else:
            assert row["silhouette"] == pytest.approx(sil, **TOL)
        for k, (d, a) in enumerate(zip(info.get("population_D_means", []), info["population_abundances"]), 1):
            assert row[f"pop{k}_D_mean"] == pytest.approx(d, **TOL)
            assert row[f"pop{k}_abund"] == pytest.approx(a, **TOL)


def test_meta_S04_sweep_reversed_thresholds():
    s_f, _ = cl.clustering_parameter_sweep(d_o1(), COLS2, distance_thresholds=(1.0, 2.0), min_abundances=(0.3,))
    s_r, _ = cl.clustering_parameter_sweep(d_o1(), COLS2, distance_thresholds=(2.0, 1.0), min_abundances=(0.3,))
    pd.testing.assert_frame_equal(s_f.iloc[::-1].reset_index(drop=True), s_r)


@pytest.mark.parametrize("was_interactive", [False, True])
def test_oracle_S05_sweep_restores_pyplot_state(was_interactive):
    """D-08 (decided): the sweep restores the interactive flag and leaves foreign figures open."""
    plt.interactive(was_interactive)
    fig = plt.figure()
    cl.clustering_parameter_sweep(d_o1(), COLS2, distance_thresholds=(1.0,), min_abundances=(0.3,))
    assert plt.isinteractive() is was_interactive
    assert fig.number in plt.get_fignums()


def test_oracle_S06_sweep_collects_exceptions_in_error_column():
    """D-08 (decided): a failing combination yields a row with the error text, not a silent skip."""
    summ, scat = cl.clustering_parameter_sweep(d_o1(), COLS2, q_squared_col="nope",
                                               distance_thresholds=(1.0, 2.0), min_abundances=(0.3,))
    assert list(summ.columns) == SUMMARY_COLS
    assert len(summ) == 2
    assert list(zip(summ["distance_threshold"], summ["min_abundance"])) == [(1.0, 0.3), (2.0, 0.3)]
    for msg in summ["error"]:
        assert isinstance(msg, str) and "ValueError" in msg and "nope" in msg
    assert summ["n_populations"].isna().all()
    assert summ["silhouette"].isna().all()
    assert scat.empty


def test_char_S07_sweep_truncates_to_four_populations_documents_current_behavior():
    """documents current behavior, not desired behavior (F-19)."""
    summ, _ = cl.clustering_parameter_sweep(d_r1(), R1COLS, distance_thresholds=(0.001,),
                                            min_abundances=(0.0,))
    assert summ.iloc[0]["n_populations"] > 4
    assert list(summ.columns) == SUMMARY_COLS
    assert not any(c.startswith("pop5") for c in summ.columns)


def test_char_S08_sweep_without_normalization_documents_current_behavior():
    """documents current behavior, not desired behavior."""
    summ, _ = cl.clustering_parameter_sweep(d_o1(), COLS2, normalize_by_q2=False,
                                            distance_thresholds=(1.0,), min_abundances=(0.3,))
    row = summ.iloc[0]
    assert row["n_populations"] >= 1
    assert np.isnan([row[f"pop{i}_D_mean"] for i in range(1, 5)]).all()
    assert np.isnan([row[f"pop{i}_D_std"] for i in range(1, 5)]).all()
    assert pd.notna(row["pop1_abund"])


# ----------------------------------------------------------------------------
# Findings (xfail strict): desired behavior, currently violated
# ----------------------------------------------------------------------------
def test_oracle_X01_stats_consistent_after_refinement():
    clustered, info = _ref_call()
    # JADE 3.1: pop1 (index 0) = fastest = cluster B
    assert info["population_abundances"][0] == pytest.approx(3 / 6, **TOL)
    assert info["population_D_means"][0] == pytest.approx(10 ** ((0.75 + 2.2) / 3), **TOL)


@pytest.mark.xfail(raises=AssertionError, strict=True, reason="F-05: Auswahl-Regel offen, siehe _dev/notes/open_discussion_points.md")
def test_finding_X02_n_gammas_matches_gamma_pop_column():
    clustered, info = run(d_o1(), COLS2, distance_threshold=2.0, min_abundance=0.3)
    assert info["population_stats"]["n_gammas"].tolist()[0] == int(clustered["gamma_pop1"].notna().sum())


# ----------------------------------------------------------------------------
# Internal helpers: _relabel_by_value, _compute_population_stats
# ----------------------------------------------------------------------------
def test_oracle_relabel_by_value_ascending_noncontiguous():
    out = cl._relabel_by_value([5, 5, 2, 9], [3.0, 5.0, 10.0, 1.0])
    # means: 5 -> 4.0, 2 -> 10.0, 9 -> 1.0  => 9:0, 5:1, 2:2
    assert out.tolist() == [1, 1, 2, 0]
    assert np.issubdtype(out.dtype, np.integer)


def test_edge_relabel_by_value_tie_is_stable():
    out = cl._relabel_by_value([3, 1, 3, 1], [1.0, 1.0, 1.0, 1.0])
    assert out.tolist() == [1, 0, 1, 0]  # smaller original label first
    assert np.issubdtype(out.dtype, np.integer)


def test_edge_relabel_by_value_empty():
    out = cl._relabel_by_value([], [])
    assert out.shape == (0,)
    assert np.issubdtype(out.dtype, np.integer)


def test_edge_relabel_by_value_nan_mean_sorts_last():
    out = cl._relabel_by_value([0, 1, 2], [np.nan, 5.0, -1.0])
    assert out.tolist() == [2, 1, 0]
    assert np.issubdtype(out.dtype, np.integer)


def test_oracle_compute_population_stats_hand_values():
    pts = pd.DataFrame({
        "cluster": [0, 0, 0, 1, 1, 2],
        "filename": ["a", "a", "b", "c", "d", "e"],
        "gamma": [1.0, 4.0, 2.0, 10.0, 40.0, 7.0],
        "D": [0.5, 2.0, 1.0, 5.0, 20.0, 3.5],
    })
    out = cl._compute_population_stats(pts, [1, 0, 7], 5, True)  # 7 has no points
    assert out["cluster_id"].tolist() == [1, 0]
    assert out["n_gammas"].tolist() == [2, 3]
    assert out["n_files"].tolist() == [2, 2]
    assert out["abundance"].tolist() == pytest.approx([2 / 5, 2 / 5], **TOL)
    assert out["mean_gamma"].tolist() == pytest.approx([20.0, 2.0], **TOL)  # sqrt(400), cbrt(8)
    assert out["std_gamma"].tolist() == pytest.approx([15.0, np.std([1.0, 4.0, 2.0])], **TOL)
    assert out["mean_D"].tolist() == pytest.approx([10.0, 1.0], **TOL)
    assert out["std_D"].tolist() == pytest.approx([7.5, np.std([0.5, 2.0, 1.0])], **TOL)
    assert out["reliable"].tolist() == [True, True]
    assert out["status"].tolist() == ["RELIABLE", "RELIABLE"]


def test_edge_compute_population_stats_no_D_columns_when_not_normalized():
    pts = pd.DataFrame({"cluster": [0, 0], "filename": ["a", "a"], "gamma": [1.0, 4.0], "D": [np.nan] * 2})
    out = cl._compute_population_stats(pts, [0], 4, False)
    assert "mean_D" not in out.columns and "std_D" not in out.columns
    assert out["n_files"].tolist() == [1]
    assert out["abundance"].tolist() == pytest.approx([0.25], **TOL)


@pytest.mark.parametrize("normalize", [True, False])
def test_meta_compute_population_stats_columns_match_cluster_info(normalize):
    _, info = run(d_o1(), COLS2, normalize_by_q2=normalize,
                  q_squared_col="q^2" if normalize else None,
                  distance_threshold=1.0, min_abundance=0.3)
    g = info["gammas_df"]
    out = cl._compute_population_stats(g, sorted(g["cluster"].unique()), 4, normalize)
    assert list(out.columns) == list(info["population_stats"].columns)


# ----------------------------------------------------------------------------
# JADE 3.1: descending numbering (pop1 = fastest = smallest particles)
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("method, t", [("hierarchical", 1.0), ("simple", 0.3)])
@pytest.mark.parametrize("normalize", [True, False])
def test_oracle_jade31_numbering_descending_both_methods(method, t, normalize):
    """JADE 3.1: pop1 = highest mean log10 D (log10 Gamma if not normalized); lists follow pop order."""
    df = make_df([1.0] * 4, {"gamma_1": VL, "gamma_2": [0.8] * 4})  # q^2 = 1: separable with and without q^2
    kw = dict(method=method, distance_threshold=t, min_abundance=0.3)
    if not normalize:
        kw.update(normalize_by_q2=False, q_squared_col=None)
    clustered, info = run(df, COLS2, **kw)
    np.testing.assert_array_equal(clustered["gamma_pop1"].to_numpy(float), df["gamma_2"].to_numpy())
    np.testing.assert_array_equal(clustered["gamma_pop2"].to_numpy(float), df["gamma_1"].to_numpy())
    key = "population_D_means" if normalize else "population_means"
    assert info[key][0] > info[key][1]
    st = info["population_stats"]
    assert st["mean_gamma"].tolist() == pytest.approx(info["population_means"], **TOL)         if "mean_gamma" in st.columns else True
    assert info["population_abundances"] == pytest.approx([1.0, 1.0], **TOL)
    assert info["reliable_populations"] == [1, 2]


def test_oracle_jade31_numbering_descending_recluster_path():
    """JADE 3.1: the uncertainty/silhouette re-cluster path also numbers fastest first."""
    clustered, info = run(d_ord(), COLS2, method="hierarchical", distance_threshold=1.0,
                          min_abundance=0.3, uncertainty_flags=True,
                          clustering_strategy="silhouette_refined")
    D = info["population_D_means"]
    assert D[0] > D[1]
    assert np.nanmean(clustered["gamma_pop1"] / clustered["q^2"]) >         np.nanmean(clustered["gamma_pop2"] / clustered["q^2"])
