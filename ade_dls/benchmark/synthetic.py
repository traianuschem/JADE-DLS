# -*- coding: utf-8 -*-
"""
ade_dls.benchmark.synthetic
===========================
Synthetic multi-angle DLS data with known ground truth, in the format the
ADE-DLS pipeline consumes.

Provenance: adapted from ``synth.py`` of the ADE-DLS synthetic-data study (reference results computed with
the JADE-DLS modules). Licence: this program is part of ADE-DLS (GPL-3.0-or-later); the datasets it
generates are distributed under CC BY-SA 4.0 (see ``data/NOTICE.md``). Changes against the original: the
``weighting`` / ``preprocessing`` imports point to ``ade_dls``; the lag grid is read from the package data;
the physics, the noise model and the seeds are unchanged.

What is simulated
-----------------
* g2(tau)-1 = beta * g1(tau)^2 with g1 = sum_i f_i * <exp(-Gamma tau)>_i, where population i
  has a lognormal Gamma distribution (mean Gamma_i = D_i q^2, relative variance = PDI_i)
  and intensity fraction f_i (angle independent; no form factor yet).
* The real ALV-7004 multi-tau lag grid (207 channels, 25 ns ... 6.3 s), angles 30-150 deg in
  10 deg steps, three runs per angle (39 curves per dataset).
* lambda = 632.8 nm, n = 1.332, water at 25 degC (eta = 0.890 mPa s), beta ~ 0.88 with a small
  per-file jitter (beta < 1).
* Noise: per-channel variance of Biganzoli & Ferri (2018) from weighting.py, scaled by a
  noise level s (sigma = s * sigma_BF, measuring time and count rate stay fixed). Either
  independent per channel ('white') or with a neighbour correlation along the lag axis
  ('correlated', a unit-variance AR(1) with a lag-dependent coefficient, see
  correlation_profile()).

Ground truth
------------
The Rh of a population is the Rh belonging to its *mean* D (z-average within the population),
Rh = kT / (6 pi eta D_mean). The Stokes-Einstein constants are the ones the pipeline uses.

Output
------
Dataset.all_correlations : {filename: DataFrame[time [ms], correlation 1..4]}  (as extract_correlation)
Dataset.df_basedata      : DataFrame as built from extract_data(), plus q and q^2, index from 1
write_asc(...)           : optional ALV-style .ASC files readable by the unchanged pipeline.

Dependencies: numpy, pandas, scipy, ade_dls.analysis.weighting (no Qt).
"""
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.constants import k as K_B

from ade_dls.analysis import weighting as wt

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- constants
WAVELENGTH_NM = 632.8
REFRACTIVE_INDEX = 1.332
TEMPERATURE_K = 298.15            # 25 degC
VISCOSITY_CP = 0.890              # water, 25 degC
ANGLES_DEG = tuple(range(30, 151, 10))
RUNS_PER_ANGLE = 3
DURATION_S = 12.0                 # per run, as in the real measurements
COUNTRATE_KHZ = 330.0             # both detector channels
BETA_MEAN = 0.88
BETA_SD = 0.005
BASE_SEED = 20260101

LAG_GRID_MS = np.loadtxt(HERE / "data" / "alv_lag_grid_ms.csv")
LAG_GRID_S = LAG_GRID_MS * 1e-3


# --------------------------------------------------------------------------- physics
def q_nm(angle_deg, wavelength_nm=WAVELENGTH_NM, n=REFRACTIVE_INDEX):
    """Scattering vector in nm^-1, same formula as the pipeline."""
    return abs(4 * np.pi * n / wavelength_nm * np.sin(np.radians(angle_deg) / 2))


def rh_to_D(rh_nm, T=TEMPERATURE_K, eta_cP=VISCOSITY_CP):
    """Stokes-Einstein: Rh [nm] -> D [m^2/s]."""
    return K_B * T / (6 * np.pi * eta_cP * 1e-3 * rh_nm * 1e-9)


def D_to_rh(D, T=TEMPERATURE_K, eta_cP=VISCOSITY_CP):
    """D [m^2/s] -> Rh [nm]."""
    return K_B * T / (6 * np.pi * eta_cP * 1e-3 * D) * 1e9


@dataclass(frozen=True)
class Population:
    rh_nm: float        # Rh belonging to the mean D of the population
    pdi: float          # relative variance of Gamma within the population
    fraction: float     # intensity fraction (at the reference angle if a form factor is used)
    scatter_radius_nm: float = 0.0   # radius entering the form factor; 0 = use rh_nm


def population_g1(pop, q, tau, n_quad=201, n_sigma=5.0):
    """<exp(-Gamma tau)> for a lognormal Gamma distribution with mean D_mean q^2 and relative
    variance pdi (trapezoid-free: normalised weights on a regular grid in ln Gamma)."""
    gamma_mean = rh_to_D(pop.rh_nm) * (q * 1e9) ** 2
    if pop.pdi <= 1e-8:
        return np.exp(-gamma_mean * tau)
    s2 = np.log1p(pop.pdi)
    s = np.sqrt(s2)
    z = np.linspace(-n_sigma, n_sigma, n_quad)
    w = np.exp(-0.5 * z ** 2)
    w /= w.sum()
    gam = np.exp(np.log(gamma_mean) - 0.5 * s2 + s * z)
    return w @ np.exp(-np.outer(gam, tau))


def sphere_form_factor(x):
    """Rayleigh-Gans-Debye form factor P(x) of a homogeneous sphere, x = q R (P -> 1 for x -> 0)."""
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = x > 1e-4
    xm = x[m]
    out[m] = (3 * (np.sin(xm) - xm * np.cos(xm)) / xm ** 3) ** 2
    return out


def intensity_fractions(pops, q, ref_q, form_factor="none"):
    """Intensity fractions of the populations at scattering vector q [nm^-1].

    'none'   angle independent: the nominal fractions of the populations.
    'sphere' the nominal fractions hold at ref_q; at other q they follow the scattered intensity per
             particle I_k(q) ~ R_k^6 P(q R_k) (RGD sphere, R_k = scatter radius) with fixed number weights."""
    f_ref = np.array([p.fraction for p in pops], dtype=float)
    f_ref = f_ref / f_ref.sum()
    if form_factor in (None, "none"):
        return f_ref
    if form_factor != "sphere":
        raise ValueError(f"unknown form factor {form_factor!r}")
    R = np.array([p.scatter_radius_nm or p.rh_nm for p in pops], dtype=float)
    def I(qq):
        return R ** 6 * sphere_form_factor(qq * R)
    f = f_ref / I(ref_q) * I(q)
    return f / f.sum()


def mixture_g1(pops, q, tau, fractions=None):
    f = (np.array([p.fraction for p in pops], dtype=float) if fractions is None
         else np.asarray(fractions, dtype=float))
    f = f / f.sum()
    return sum(fi * population_g1(p, q, tau) for fi, p in zip(f, pops))


# --------------------------------------------------------------------------- noise
# Neighbour correlation of the standardised noise along the lag axis versus x = tau / tau_c.
# Generic, rounded profile motivated by the lag-to-lag correlation of triplicate residuals of
# real ALV data (~0.8 well before the decay, 0.9-1.0 around it, falling to 0 beyond ~100 tau_c).
_PROFILE_LOG10X = np.array([-3.0, -2.0, -1.0, 0.25, 0.7, 1.5, 2.5])
_PROFILE_RHO = np.array([0.80, 0.88, 0.97, 0.92, 0.65, 0.28, 0.0])


def correlation_profile(x):
    return np.interp(np.log10(np.maximum(x, 1e-12)), _PROFILE_LOG10X, _PROFILE_RHO)


def correlated_unit_noise(eps, x):
    """Unit-variance noise with neighbour correlation rho(x_k): xi_k = rho_k xi_{k-1} + sqrt(1-rho_k^2) eps_k."""
    rho = correlation_profile(x)
    xi = np.empty_like(eps)
    xi[0] = eps[0]
    for k in range(1, len(eps)):
        xi[k] = rho[k] * xi[k - 1] + np.sqrt(1 - rho[k] ** 2) * eps[k]
    return xi


def bf_sigma(tau, clean, beta, T=DURATION_S, cr_khz=COUNTRATE_KHZ):
    """Per-channel standard deviation of Biganzoli & Ferri (via weighting.estimate_weights), using
    the single-exponential characteristic time of the noise-free curve, plus that time."""
    _, tau_c, _ = wt.prefit_single_exponential(tau, clean)
    _, diag = wt.estimate_weights(tau, clean, cr_khz, cr_khz, T, beta=beta, tau_c=tau_c,
                                  return_diagnostics=True, verbose=False)
    return np.sqrt(diag["sigma2"]), tau_c


# --------------------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    geometry: str                 # e.g. 'bi_r3_pdi0.05'
    populations: tuple
    noise_level: float = 1.0      # s: sigma = s * sigma_BF
    noise_mode: str = "white"     # 'white' | 'correlated'
    beta_mean: float = BETA_MEAN
    form_factor: str = "none"     # 'none' (fractions independent of angle) | 'sphere' (RGD sphere, see intensity_fractions)
    ref_angle_deg: float = 90.0   # angle at which the nominal fractions hold if a form factor is used

    @property
    def key(self):
        mode = "white" if self.noise_level == 0 else self.noise_mode
        return f"{self.geometry}__s{self.noise_level:g}_{mode}"

    def truth(self, angles=ANGLES_DEG):
        pops = sorted(self.populations, key=lambda p: p.rh_nm)
        f = np.array([p.fraction for p in pops], float)
        f /= f.sum()
        D = np.array([rh_to_D(p.rh_nm) for p in pops])
        pdi = np.array([p.pdi for p in pops])
        D_z = float(np.sum(f * D))
        # With an angle-dependent form factor the z-average D_z(theta) varies with the angle, so the
        # Gamma-q^2 regression of the cumulant methods returns an effective value: the slope of the exact
        # mean decay rates (B: through the origin, C: free intercept, as in the pipeline settings).
        ref_q = q_nm(self.ref_angle_deg)
        qs = np.array([q_nm(a) for a in angles])
        Dz_ang = np.array([np.sum(intensity_fractions(pops, qq, ref_q, self.form_factor) * D) for qq in qs])
        x = (qs * 1e9) ** 2
        y = Dz_ang * x
        slope_origin = float(np.sum(x * y) / np.sum(x * x))
        slope_free = float(np.polyfit(x / x.max(), y, 1)[0] / x.max())
        return {
            "n_pop": len(pops),
            "rh_pops": [p.rh_nm for p in pops],
            "fractions": f.tolist(),
            "rh_z": float(D_to_rh(D_z)),                              # z-average at the reference angle
            "pdi_mix": float(np.sum(f * D ** 2 * (1 + pdi)) / D_z ** 2 - 1),
            "rh_z_reg_origin": float(D_to_rh(slope_origin)),          # what B should return
            "rh_z_reg_free": float(D_to_rh(slope_free)),              # what C2 / C4 should return
            "fractions_by_angle": {int(a): intensity_fractions(pops, qq, ref_q, self.form_factor).tolist()
                                   for a, qq in zip(angles, qs)} if self.form_factor != "none" else None,
        }


MONO_RH_NM = 50.0
MONO_PDIS = (0.05, 0.10, 0.20)
BI_RATIOS = (1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0)
BI_PDIS = (0.05, 0.10, 0.20)
BI_FRACTIONS = (0.5, 0.8)          # intensity fraction of the SMALLER particle (0.5 = 50/50, 0.8 = 80/20)
BI_RH_GEOMEAN_NM = 50.0
NOISE_LEVELS = (0.0, 1.0, 2.0)
CUMULANT_PDIS = (0.05, 0.10, 0.20)   # geometries on which the cumulant methods B, C2, C4 are run (all PDIs, 50/50 only)


MONO_SIZE_RH_NM = (25.0, 100.0)      # extra monomodal sizes (size dependence of the cumulant / D / NNLS / Reg results)


def geometry_id(kind, pdi, ratio=None, fraction=0.5, rh=None):
    if kind == "mono":
        return f"mono_pdi{pdi:g}" if rh is None else f"mono_rh{rh:g}_pdi{pdi:g}"
    return f"bi_r{ratio:g}_pdi{pdi:g}" + ("" if fraction == 0.5 else f"_i{round(100 * fraction)}")


def geometries():
    """{geometry id: populations}: monomodal + bimodal grid. Both populations of a bimodal case
    have the same PDI; Rh = 50 nm / sqrt(ratio) and 50 nm * sqrt(ratio); the D ratio equals the Rh ratio."""
    g = {}
    for pdi in MONO_PDIS:
        g[geometry_id("mono", pdi)] = (Population(MONO_RH_NM, pdi, 1.0),)
    for f in BI_FRACTIONS:
        for r in BI_RATIOS:
            for pdi in BI_PDIS:
                lo, hi = BI_RH_GEOMEAN_NM / np.sqrt(r), BI_RH_GEOMEAN_NM * np.sqrt(r)
                g[geometry_id("bi", pdi, r, f)] = (Population(lo, pdi, f), Population(hi, pdi, round(1 - f, 10)))
    return g


def size_geometries():
    """Monomodal samples at the extra sizes MONO_SIZE_RH_NM (kept apart from geometries(), which defines the main grid)."""
    return {geometry_id("mono", pdi, rh=rh): (Population(rh, pdi, 1.0),)
            for rh in MONO_SIZE_RH_NM for pdi in MONO_PDIS}


FF_PDIS = (0.05, 0.10)
FF_RATIOS = (3.0, 5.0, 7.0, 10.0)
FF_SIZE_RATIO = 5.0                     # ratio at which the absolute size (geometric-mean Rh) is varied
FF_GEOMEANS_NM = (25.0, 50.0, 100.0)    # the form-factor effect depends on q R, i.e. on the absolute sizes


def ff_geometry_id(ratio, pdi, geomean_nm=BI_RH_GEOMEAN_NM):
    return f"ff_r{ratio:g}_pdi{pdi:g}_gm{geomean_nm:g}"


def ff_geometries():
    """Bimodal samples with the sphere form factor (RGD): 50/50 at the reference angle (90 deg), scatter radius =
    Rh, populations of equal PDI. Kept apart from geometries(), the main grid of angle-independent fractions."""
    g = {}
    for pdi in FF_PDIS:
        for r in FF_RATIOS:
            g[ff_geometry_id(r, pdi)] = _bimodal(r, pdi, 0.5, BI_RH_GEOMEAN_NM)
        for gm in FF_GEOMEANS_NM:
            g[ff_geometry_id(FF_SIZE_RATIO, pdi, gm)] = _bimodal(FF_SIZE_RATIO, pdi, 0.5, gm)
    return g


def _bimodal(ratio, pdi, f_small, geomean_nm):
    lo, hi = geomean_nm / np.sqrt(ratio), geomean_nm * np.sqrt(ratio)
    return (Population(lo, pdi, f_small), Population(hi, pdi, round(1 - f_small, 10)))


I20_FRACTION = 0.2      # intensity fraction of the SMALLER particle: 20/80, large particle dominant


def i20_geometries():
    """Bimodal 20/80 samples (20 % of the intensity in the smaller particle), same ratios and PDIs as the main grid,
    angle-independent fractions. Kept apart from geometries() (the main grid has 50/50 and 80/20 only)."""
    return {geometry_id("bi", pdi, r, I20_FRACTION): _bimodal(r, pdi, I20_FRACTION, BI_RH_GEOMEAN_NM)
            for r in BI_RATIOS for pdi in BI_PDIS}


TRI_RATIOS = (2.0, 3.0, 5.0)      # ratio of neighbouring populations: Rh = 50 / r, 50, 50 * r nm (outer ratio r^2), geometric mean 50 nm
TRI_PDIS = (0.05, 0.10, 0.20)     # the same PDI for all three populations
TRI_FRACTIONS = {"": (1 / 3, 1 / 3, 1 / 3),          # equal intensities
                 "_i50": (3 / 6, 2 / 6, 1 / 6),      # 50/33/17: small particle dominant (intensity 3 : 2 : 1, small -> large)
                 "_i17": (1 / 6, 2 / 6, 3 / 6)}      # 17/33/50: large particle dominant (1 : 2 : 3)


def tri_geometry_id(ratio, pdi, suffix=""):
    return f"tri_r{ratio:g}_pdi{pdi:g}{suffix}"


def trimodal_geometries():
    """Trimodal samples (3 x 3 x 3 = 27): neighbour ratio 2, 3, 5, PDI 0.05 / 0.1 / 0.2, intensity splits equal, 50/33/17, 17/33/50
    (smallest -> largest particle), angle-independent fractions. Kept apart from geometries() (main grid: monomodal and bimodal)."""
    g = {}
    for suf, fr in TRI_FRACTIONS.items():
        for r in TRI_RATIOS:
            for pdi in TRI_PDIS:
                rh = (BI_RH_GEOMEAN_NM / r, BI_RH_GEOMEAN_NM, BI_RH_GEOMEAN_NM * r)
                g[tri_geometry_id(r, pdi, suf)] = tuple(Population(x, pdi, f) for x, f in zip(rh, fr))
    return g


def cumulant_geometries():
    """Reduced geometry set for B, C2, C4 (they depend on noise and PDI, hardly on separation)."""
    ids = [geometry_id("mono", p) for p in CUMULANT_PDIS]
    ids += [geometry_id("bi", p, r, 0.5) for p in CUMULANT_PDIS for r in BI_RATIOS]
    return ids


def make_scenarios(geometry_ids=None, levels=NOISE_LEVELS, modes=("white",)):
    geoms = {**geometries(), **size_geometries(), **ff_geometries(), **i20_geometries(), **trimodal_geometries()}
    ids = list(geometries()) if geometry_ids is None else list(geometry_ids)
    out = []
    for gid in ids:
        ff = "sphere" if gid in ff_geometries() else "none"
        for s in levels:
            for m in (("white",) if s == 0 else modes):
                out.append(Scenario(gid, geoms[gid], float(s), m, form_factor=ff))
    return out


def seed_sequence(geometry, rep, base=BASE_SEED):
    """Seed depends on (geometry, replicate) only: the same random numbers are reused for every
    noise level and mode (common random numbers) and adding replicates never changes old ones."""
    return np.random.SeedSequence([base, zlib.crc32(geometry.encode()), int(rep)])


# --------------------------------------------------------------------------- dataset
@dataclass
class Dataset:
    scenario: Scenario
    rep: int
    all_correlations: dict
    df_basedata: pd.DataFrame
    clean: dict = field(default_factory=dict)       # filename -> noise-free g2-1
    sigma: dict = field(default_factory=dict)       # filename -> sigma used (incl. noise level)
    truth: dict = field(default_factory=dict)


def simulate_dataset(scn, rep, angles=ANGLES_DEG, runs=RUNS_PER_ANGLE, base_seed=BASE_SEED):
    rng = np.random.default_rng(seed_sequence(scn.geometry, rep, base_seed))
    tau = LAG_GRID_S
    all_corr, clean_d, sigma_d, rows = {}, {}, {}, []
    for ang in angles:
        q = q_nm(ang)
        fr = intensity_fractions(scn.populations, q, q_nm(scn.ref_angle_deg), scn.form_factor)
        g1 = mixture_g1(scn.populations, q, tau, fr)
        for run in range(1, runs + 1):
            beta = float(np.clip(rng.normal(scn.beta_mean, BETA_SD), 0.5, 1.0))
            eps = rng.standard_normal(tau.size)
            clean = beta * g1 ** 2
            if scn.noise_level > 0:
                sig, tau_c = bf_sigma(tau, clean, beta)
                xi = eps if scn.noise_mode == "white" else correlated_unit_noise(eps, tau / tau_c)
                y = clean + scn.noise_level * sig * xi
                sig = scn.noise_level * sig
            else:
                y, sig = clean.copy(), np.zeros_like(clean)
            name = f"{scn.key}_r{rep:03d}_a{ang:03d}_{run:04d}.ASC"
            all_corr[name] = pd.DataFrame({0: LAG_GRID_MS, 1: y, 2: y,
                                           3: np.zeros_like(y), 4: np.zeros_like(y)}
                                          ).rename(columns={0: "time [ms]", 1: "correlation 1",
                                                            2: "correlation 2", 3: "correlation 3",
                                                            4: "correlation 4"})
            clean_d[name], sigma_d[name] = clean, sig
            rows.append({"angle [°]": float(ang), "temperature [K]": TEMPERATURE_K,
                         "wavelength [nm]": WAVELENGTH_NM, "refractive_index": REFRACTIVE_INDEX,
                         "viscosity [cp]": VISCOSITY_CP, "duration [s]": DURATION_S,
                         "meancr0 [kHz]": COUNTRATE_KHZ, "meancr1 [kHz]": COUNTRATE_KHZ,
                         "monitordiode [cps]": np.nan, "filename": name})
    base = pd.DataFrame(rows)
    base["q"] = abs(4 * np.pi * base["refractive_index"] / base["wavelength [nm]"]
                    * np.sin(np.radians(base["angle [°]"]) / 2))
    base["q^2"] = base["q"] ** 2
    base.index = base.index + 1
    return Dataset(scn, rep, all_corr, base, clean_d, sigma_d, scn.truth(angles))


# --------------------------------------------------------------------------- ASC export
def write_asc(ds, folder):
    """Write one ALV-style .ASC file per curve (readable by preprocessing.extract_data /
    extract_correlation / extract_countrate). Returns the list of paths."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    base = ds.df_basedata.set_index("filename")
    for name, df in ds.all_correlations.items():
        b = base.loc[name]
        lines = ['ALV-7004/USB', 'Date :\t"01.01.2026"', 'Time :\t"00:00:00"',
                 f'Samplename : \t"{ds.scenario.key}"']
        lines += [f'SampMemo({i}) : \t""' for i in range(10)]
        lines += [f'Temperature [K] :\t{b["temperature [K]"]:14.5f}',
                  f'Viscosity [cp]  :\t{b["viscosity [cp]"]:14.5f}',
                  f'Refractive Index:\t{b["refractive_index"]:14.5f}',
                  f'Wavelength [nm] :\t{b["wavelength [nm]"]:14.5f}',
                  f'Angle [°]       :\t{b["angle [°]"]:14.5f}',
                  f'Duration [s]    :\t{int(b["duration [s]"]):10d}',
                  'Mode            :\t"C-CH0/1+1/0"',
                  f'MeanCR0 [kHz]   :\t{b["meancr0 [kHz]"]:14.5f}',
                  f'MeanCR1 [kHz]   :\t{b["meancr1 [kHz]"]:14.5f}',
                  'MeanCR2 [kHz]   :\t       0.00000', 'MeanCR3 [kHz]   :\t       0.00000',
                  'Monitor Diode   :\t   1000000', '', '"Correlation"']
        for r in df.itertuples(index=False):
            lines.append(f'{r[0]:15.5E}\t{r[1]:15.5E}\t{r[2]:15.5E}\t{0:15.5E}\t{0:15.5E}')
        lines += ['', '"Count Rate"']
        for i in range(1, 11):
            cr = b["meancr0 [kHz]"]
            lines.append(f'{i * 0.046875:14.5f}\t{cr:14.5f}\t{cr:14.5f}\t{cr:14.5f}\t{cr:14.5f}')
        p = folder / name
        p.write_text("\n".join(lines) + "\n", encoding="cp1252")
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- reuse API
def _as_population(p):
    return p if isinstance(p, Population) else Population(*p)


def simulate(populations, noise_level=1.0, noise_mode="white", seed=0, angles=ANGLES_DEG,
             runs=RUNS_PER_ANGLE, beta_mean=BETA_MEAN, name="custom", form_factor="none", ref_angle_deg=90.0):
    """Free-parameter entry point, independent of the benchmark grid and runner.

    populations : iterable of Population or (rh_nm, pdi, intensity_fraction[, scatter_radius_nm]) tuples
    form_factor : 'none' (angle-independent fractions) | 'sphere' (RGD sphere; fractions hold at ref_angle_deg)
    noise_level : s, sigma = s * sigma_BF (0 = noise-free), noise_mode: 'white' | 'correlated'
    seed        : integer; same (name, seed) -> identical data, also across noise levels

    Example (e.g. in a marimo cell replacing the file-loading cells):
        ds = synth.simulate([(30, 0.05, 0.5), (90, 0.05, 0.5)], noise_level=2, seed=7)
        all_correlations_mod = ds.all_correlations
        df_basedata_mod      = ds.df_basedata
        processed_correlations_1 = ds.processed_correlations()
    Everything downstream of those three objects runs unchanged.
    """
    pops = tuple(_as_population(p) for p in populations)
    scn = Scenario(name, pops, float(noise_level), noise_mode, beta_mean, form_factor, float(ref_angle_deg))
    return simulate_dataset(scn, seed, angles=angles, runs=runs)


def _processed_correlations(self):
    """{filename: DataFrame['t [s]', 'g(2)-1']}, built by the pipeline's own
    preprocessing.process_correlation_data (what the notebooks call processed_correlations_1)."""
    from ade_dls.core import preprocessing as pp
    drop = ["time [ms]", "correlation 1", "correlation 2", "correlation 3", "correlation 4"]
    return pp.process_correlation_data(self.all_correlations, drop)


Dataset.processed_correlations = _processed_correlations
