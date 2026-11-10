# -*- coding: utf-8 -*-
"""Tests of the synthetic-data simulator ade_dls.benchmark.synthetic (ported from the synthetic-data study)."""
import numpy as np
import pandas as pd
import pytest

from ade_dls.benchmark import synthetic as synth
from ade_dls.core import preprocessing as pp

BI3 = [(50 / 3 ** 0.5, 0.05, 0.5), (50 * 3 ** 0.5, 0.05, 0.5)]


def test_lag_grid_is_the_alv_grid():
    assert len(synth.LAG_GRID_S) == 207
    assert synth.LAG_GRID_MS[0] == pytest.approx(2.5e-5)
    assert synth.LAG_GRID_MS[-1] == pytest.approx(6291.46)
    assert np.all(np.diff(synth.LAG_GRID_S) > 0)


def test_physics_roundtrip_and_q():
    assert synth.D_to_rh(synth.rh_to_D(57.0)) == pytest.approx(57.0)
    q90 = 4 * np.pi * 1.332 / 632.8 * np.sin(np.pi / 4)
    assert synth.q_nm(90) == pytest.approx(q90)


@pytest.mark.parametrize("pdi", [0.01, 0.05, 0.2])
def test_population_moments(pdi):
    """g1(tau) of a population has mean Gamma = D q^2 and relative variance = PDI."""
    q = synth.q_nm(90)
    pop = synth.Population(50.0, pdi, 1.0)
    tau = np.array([0.0, 1e-9, 2e-9])                    # tiny taus: finite differences at tau -> 0
    g = synth.population_g1(pop, q, tau)
    gam_mean = synth.rh_to_D(50.0) * (q * 1e9) ** 2
    assert g[0] == pytest.approx(1.0)
    m1 = -(g[1] - g[0]) / 1e-9
    m2 = (g[2] - 2 * g[1] + g[0]) / 1e-18
    assert m1 == pytest.approx(gam_mean, rel=2e-3)
    assert m2 / m1 ** 2 - 1 == pytest.approx(pdi, rel=0.02, abs=2e-3)


def test_dataset_structure_matches_pipeline():
    ds = synth.simulate([(50, 0.05, 1.0)], noise_level=1, seed=3)
    assert len(ds.all_correlations) == 13 * 3
    assert set(ds.df_basedata["filename"]) == set(ds.all_correlations)
    assert ds.df_basedata.index[0] == 1
    for col in ["angle [°]", "temperature [K]", "wavelength [nm]", "refractive_index", "viscosity [cp]",
                "duration [s]", "meancr0 [kHz]", "meancr1 [kHz]", "q", "q^2", "filename"]:
        assert col in ds.df_basedata.columns
    assert sorted(ds.df_basedata["angle [°]"].unique()) == list(range(30, 151, 10))
    df = next(iter(ds.all_correlations.values()))
    assert list(df.columns) == ["time [ms]", "correlation 1", "correlation 2", "correlation 3", "correlation 4"]
    pc = ds.processed_correlations()
    assert set(pc) == set(ds.all_correlations)
    assert list(next(iter(pc.values())).columns) == ["t [s]", "g(2)-1"]


def test_seed_determinism_and_common_random_numbers():
    a = synth.simulate(BI3, noise_level=1, seed=5)
    b = synth.simulate(BI3, noise_level=1, seed=5)
    c = synth.simulate(BI3, noise_level=2, seed=5)
    d = synth.simulate(BI3, noise_level=1, seed=6)
    k = list(a.all_correlations)[10]
    ya, yb = a.all_correlations[k]["correlation 1"].values, b.all_correlations[k]["correlation 1"].values
    assert np.array_equal(ya, yb)
    kc = list(c.all_correlations)[10]
    noise1 = ya - a.clean[k]
    noise2 = c.all_correlations[kc]["correlation 1"].values - c.clean[kc]
    assert np.allclose(noise2, 2 * noise1, atol=1e-12)          # same xi, doubled level
    assert not np.array_equal(ya, d.all_correlations[list(d.all_correlations)[10]]["correlation 1"].values)


def test_noise_free_has_no_noise():
    ds = synth.simulate(BI3, noise_level=0, seed=1)
    for k, df in ds.all_correlations.items():
        assert np.array_equal(df["correlation 1"].values, ds.clean[k])
    assert 0.8 < next(iter(ds.clean.values()))[0] < 1.0         # beta < 1


def test_noise_amplitude_matches_biganzoli_ferri():
    ds = synth.simulate(BI3, noise_level=1, seed=11)
    z = np.concatenate([(ds.all_correlations[k]["correlation 1"].values - ds.clean[k]) / ds.sigma[k]
                        for k in ds.clean])
    assert z.std() == pytest.approx(1.0, abs=0.05)
    assert abs(z.mean()) < 0.05
    ds4 = synth.simulate(BI3, noise_level=4, seed=11)
    z4 = np.concatenate([(ds4.all_correlations[k]["correlation 1"].values - ds4.clean[k]) / ds4.sigma[k]
                         for k in ds4.clean])
    assert z4.std() == pytest.approx(1.0, abs=0.05)


def test_correlated_noise_has_neighbour_correlation():
    def lag1(mode):
        ds = synth.simulate(BI3, noise_level=1, noise_mode=mode, seed=2)
        zs = [(ds.all_correlations[k]["correlation 1"].values - ds.clean[k]) / ds.sigma[k] for k in ds.clean]
        sl = slice(60, 140)                                     # around the decay for these Gammas
        a = np.concatenate([z[sl][:-1] for z in zs])
        b = np.concatenate([z[sl][1:] for z in zs])
        return np.corrcoef(a, b)[0, 1]
    assert abs(lag1("white")) < 0.08
    assert lag1("correlated") > 0.7
    zs = synth.correlated_unit_noise(np.random.default_rng(0).standard_normal(200000), np.full(200000, 1.0))
    assert zs.std() == pytest.approx(1.0, abs=0.02)             # marginal variance preserved


def test_grid_sizes():
    g = synth.geometries()
    assert len(g) == 3 + 7 * 3 * 2                      # 3 monomodal + ratios x PDIs x intensity splits
    assert "bi_r5_pdi0.1" in g and "bi_r5_pdi0.1_i80" in g
    sc = synth.make_scenarios(levels=(0, 1, 2), modes=("correlated",))
    assert len(sc) == 45 * 3
    sc = synth.make_scenarios(levels=(0, 1, 2), modes=("white", "correlated"))
    assert len(sc) == 45 * (1 + 2 * 2)
    keys = [s.key for s in sc]
    assert len(set(keys)) == len(keys)
    assert len(synth.cumulant_geometries()) == 3 + 3 * 7
    assert all(x in g for x in synth.cumulant_geometries())


def test_bimodal_populations_in_grid():
    g = synth.geometries()
    lo, hi = g["bi_r5_pdi0.1_i80"]
    assert hi.rh_nm / lo.rh_nm == pytest.approx(5.0)
    assert (lo.rh_nm * hi.rh_nm) ** 0.5 == pytest.approx(50.0)
    assert lo.pdi == hi.pdi == 0.1
    assert (lo.fraction, hi.fraction) == (0.8, pytest.approx(0.2))


def test_truth_values():
    t = synth.simulate(BI3, noise_level=0, seed=1).truth
    assert t["n_pop"] == 2
    assert t["rh_pops"] == pytest.approx([50 / 3 ** 0.5, 50 * 3 ** 0.5])
    D = [synth.rh_to_D(r) for r in t["rh_pops"]]
    assert synth.rh_to_D(t["rh_z"]) == pytest.approx(0.5 * D[0] + 0.5 * D[1])
    assert D[0] / D[1] == pytest.approx(3.0)


def test_asc_roundtrip_through_pipeline_readers(tmp_path):
    ds = synth.simulate(BI3, noise_level=1, seed=4, angles=(30, 90), runs=2)
    paths = synth.write_asc(ds, tmp_path)
    assert len(paths) == 4
    for p in paths:
        base = ds.df_basedata.set_index("filename").loc[p.name]
        meta = pp.extract_data(str(p)).iloc[0]
        assert meta["angle [°]"] == pytest.approx(base["angle [°]"])
        assert meta["temperature [K]"] == pytest.approx(base["temperature [K]"], rel=1e-6)
        assert meta["wavelength [nm]"] == pytest.approx(632.8)
        assert meta["duration [s]"] == 12
        corr = pp.extract_correlation(str(p))
        ref = ds.all_correlations[p.name]
        assert np.allclose(corr[0].values, ref["time [ms]"].values, rtol=1e-5)
        assert np.allclose(corr[1].values, ref["correlation 1"].values, rtol=1e-5, atol=1e-8)
        assert pp.extract_countrate(str(p)) is not None


def test_size_geometries_are_separate_from_main_grid():
    main, extra = synth.geometries(), synth.size_geometries()
    assert len(extra) == 2 * 3 and not set(main) & set(extra)
    assert "mono_rh25_pdi0.1" in extra and "mono_rh100_pdi0.2" in extra
    pop = extra["mono_rh25_pdi0.1"][0]
    assert (pop.rh_nm, pop.pdi, pop.fraction) == (25.0, 0.1, 1.0)
    scn = synth.make_scenarios(["mono_rh100_pdi0.05"], levels=(1,), modes=("correlated",))
    assert len(scn) == 1 and scn[0].populations[0].rh_nm == 100.0
    assert len(synth.make_scenarios(levels=(1,), modes=("correlated",))) == 45     # default grid unchanged


def test_sphere_form_factor_limits():
    assert synth.sphere_form_factor(0.0) == pytest.approx(1.0)
    assert synth.sphere_form_factor(1e-3) == pytest.approx(1.0, abs=1e-5)
    assert synth.sphere_form_factor(4.4934) < 1e-3                  # first zero of the sphere form factor
    x = np.array([0.5, 1.5, 3.0])
    assert np.all(np.diff(synth.sphere_form_factor(x)) < 0)


def test_intensity_fractions_follow_form_factor():
    pops = (synth.Population(22.0, 0.05, 0.5), synth.Population(112.0, 0.05, 0.5))
    ref = synth.q_nm(90)
    f_none = synth.intensity_fractions(pops, synth.q_nm(30), ref, "none")
    assert np.allclose(f_none, [0.5, 0.5])                          # angle independent
    assert np.allclose(synth.intensity_fractions(pops, ref, ref, "sphere"), [0.5, 0.5])   # nominal at the reference angle
    large = [synth.intensity_fractions(pops, synth.q_nm(a), ref, "sphere")[1] for a in (30, 70, 90, 110, 150)]
    assert all(np.diff(large) < 0) and large[0] > 0.6 and large[-1] < 0.35   # large particle fades with angle
    assert synth.intensity_fractions(pops, synth.q_nm(150), ref, "sphere").sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        synth.intensity_fractions(pops, ref, ref, "mie")


def test_form_factor_off_reproduces_old_data_and_truth():
    pops = [(22.0, 0.05, 0.5), (112.0, 0.05, 0.5)]
    a = synth.simulate(pops, noise_level=1, seed=3)
    b = synth.simulate(pops, noise_level=1, seed=3, form_factor="none")
    k = list(a.clean)[5]
    assert np.array_equal(a.clean[k], b.clean[k])
    tau, q = synth.LAG_GRID_S, synth.q_nm(60)
    old = synth.mixture_g1(tuple(synth.Population(*p) for p in pops), q, tau)      # old call signature
    ds = synth.simulate(pops, noise_level=0, seed=3, angles=(60,), runs=1)
    m = old > 1e-3                                                                    # before the curve has decayed
    ratio = next(iter(ds.clean.values()))[m] / old[m] ** 2                            # = beta, constant over tau
    assert np.allclose(ratio, ratio[0])
    t = a.truth
    assert t["rh_z_reg_origin"] == pytest.approx(t["rh_z"]) and t["rh_z_reg_free"] == pytest.approx(t["rh_z"])
    assert t["fractions_by_angle"] is None


def test_truth_regression_value_with_form_factor():
    pops = [(22.0, 0.05, 0.5), (112.0, 0.05, 0.5)]
    t = synth.simulate(pops, noise_level=0, seed=3, form_factor="sphere").truth
    assert set(t["fractions_by_angle"]) == set(synth.ANGLES_DEG)
    assert t["fractions_by_angle"][30][1] > 0.6 > 0.4 > t["fractions_by_angle"][150][1]
    assert t["rh_z_reg_origin"] != pytest.approx(t["rh_z"], rel=1e-3)           # effective value differs from the z-average at 90 deg
    assert min(22.0, 112.0) < t["rh_z_reg_free"] < max(22.0, 112.0)


def test_form_factor_geometries_are_separate_and_use_the_sphere_model():
    ff = synth.ff_geometries()
    assert not set(ff) & set(synth.geometries()) and len(ff) == 2 * (4 + 2)
    assert "ff_r5_pdi0.05_gm25" in ff and "ff_r5_pdi0.1_gm100" in ff
    lo, hi = ff["ff_r5_pdi0.1_gm100"]
    assert hi.rh_nm / lo.rh_nm == pytest.approx(5.0) and (lo.rh_nm * hi.rh_nm) ** 0.5 == pytest.approx(100.0)
    scn = synth.make_scenarios(["ff_r5_pdi0.05_gm50", "bi_r5_pdi0.05"], levels=(1,), modes=("correlated",))
    assert [s.form_factor for s in scn] == ["sphere", "none"]
    assert len(synth.make_scenarios(levels=(1,), modes=("correlated",))) == 45          # main grid unchanged


def test_i20_geometries_are_separate_and_reversed_intensity():
    g = synth.i20_geometries()
    assert len(g) == 7 * 3 and not set(g) & set(synth.geometries())
    assert "bi_r5_pdi0.1_i20" in g
    lo, hi = g["bi_r5_pdi0.1_i20"]
    assert (lo.fraction, hi.fraction) == (0.2, pytest.approx(0.8))                 # large particle dominant
    assert hi.rh_nm / lo.rh_nm == pytest.approx(5.0) and (lo.rh_nm * hi.rh_nm) ** 0.5 == pytest.approx(50.0)
    scn = synth.make_scenarios(["bi_r5_pdi0.1_i20"], levels=(1,), modes=("correlated",))
    assert scn[0].form_factor == "none" and scn[0].populations == g["bi_r5_pdi0.1_i20"]
    assert len(synth.make_scenarios(levels=(1,), modes=("correlated",))) == 45     # main grid unchanged


def test_trimodal_geometries():
    g = synth.trimodal_geometries()
    assert len(g) == 27
    for pops in g.values():
        assert len(pops) == 3
        assert abs(sum(p.fraction for p in pops) - 1.0) < 1e-9
        rh = [p.rh_nm for p in pops]
        assert rh == sorted(rh) and abs(rh[1] - synth.BI_RH_GEOMEAN_NM) < 1e-9
    sc = synth.make_scenarios(["tri_r3_pdi0.1_i17"], (0.0,))[0]
    t = sc.truth()
    assert t["n_pop"] == 3 and t["fractions"][2] > t["fractions"][0]
