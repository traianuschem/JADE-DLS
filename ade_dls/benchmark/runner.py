"""Run ADE-DLS methods on simulated (or any loaded) series and return results in the format of the
synthetic-data study (``n_found``, sorted ``rh`` list, ``abundance``, ``pdi``), so that they can be compared
with the stored reference results and evaluated against the ground truth (``ade_dls.benchmark.evaluate``).

The settings dictionary has the names of the study (``b_fit_limits``, ``c_base``, ``d_gap_threshold``,
``nnls_points``, ``reg_alpha`` ...); see ``tests/data/synthetic_golden/golden.json`` for examples.
"""
import contextlib
import io
import tempfile
from typing import Any, Dict, Optional

import numpy as np

from ade_dls import headless
from ade_dls.benchmark import synthetic as syn

METHODS = ("B", "C2", "C4", "D", "NNLS", "Reg")


def all_geometries() -> Dict[str, tuple]:
    """Every sample geometry of the study (monomodal sizes, bimodal grid, 20/80, trimodal, form factor)."""
    return {**syn.geometries(), **syn.size_geometries(), **syn.ff_geometries(),
            **syn.i20_geometries(), **syn.trimodal_geometries()}


def simulate_series(geometry: str, noise_level: float, noise_mode: str = "white", rep: int = 0,
                    folder: Optional[str] = None, angles=None, runs: Optional[int] = None):
    """Simulate one sample, write it as ALV ``.ASC`` files into ``folder`` (temporary if None) and load it
    with the normal loader. Returns ``(data, dataset)``; ``dataset.truth`` holds the ground truth.
    ``angles`` / ``runs`` default to the study (30-150 deg in 10 deg steps, 3 runs = 39 files)."""
    form_factor = "sphere" if geometry in syn.ff_geometries() else "none"
    scn = syn.Scenario(geometry, all_geometries()[geometry], float(noise_level), noise_mode, form_factor=form_factor)
    kw = {}
    if angles is not None:
        kw["angles"] = tuple(angles)
    if runs is not None:
        kw["runs"] = int(runs)
    ds = syn.simulate_dataset(scn, rep, **kw)
    folder = folder or tempfile.mkdtemp(prefix="ade_synth_")
    syn.write_asc(ds, folder)
    with contextlib.redirect_stdout(io.StringIO()):
        data = headless.load_series(folder)
    return data, ds


def _rows(frame, prefix):
    sub = frame[frame["Fit"].astype(str).str.startswith(prefix)]
    return sub.sort_values("Rh [nm]")


def run_method(data, method: str, settings: Dict[str, Any], weighted: bool) -> Dict[str, Any]:
    """Run one method with study-style ``settings``; returns ``{"method", "n_found", "rh", "abundance",
    "pdi", "table"}`` (``table``: per-file DataFrame of the fit, if the method has one)."""
    s = settings
    out = {"method": method, "n_found": None, "rh": [], "abundance": [], "pdi": None, "table": None}
    with contextlib.redirect_stdout(io.StringIO()):
        if method == "B":
            an, res = headless.run_cumulant_b(
                data, tuple(s["b_fit_limits"]), q_range=tuple(s["b_q2_range"]),
                fit_through_origin=s["b_through_origin"], use_weighting=weighted)
            out.update(n_found=1, rh=[float(res["Rh [nm]"].iloc[0])], abundance=[1.0])
        elif method in ("C2", "C4"):
            params = {"fit_function": "fit_function2" if method == "C2" else "fit_function4",
                      "adaptive_initial_guesses": True, "adaptation_strategy": "individual",
                      "optimizer": s["c_method"], "fit_limits": tuple(s["c_fit_limits"]),
                      "initial_parameters": list(s["c_base"])}
            an, res = headless.run_cumulant_c(data, params, use_weighting=weighted)
            out.update(n_found=1, rh=[float(res["Rh [nm]"].iloc[0])], abundance=[1.0], pdi=float(res["PDI"].iloc[0]))
        elif method == "D":
            an, res = headless.run_cumulant_d(data, {"gap_threshold": s["d_gap_threshold"]}, use_weighting=weighted)
            pops = _rows(res, "Rh from Method D (Population")
            out.update(n_found=len(pops), rh=[float(x) for x in pops["Rh [nm]"]], table=an.method_d_fit)
        elif method == "NNLS":
            p = {"decay_times": np.logspace(-8, 1, int(s["nnls_points"])), "prominence": s["nnls_prominence"],
                 "distance": s["nnls_distance"], "distance_threshold": s["cluster_distance"],
                 "clustering_strategy": s["cluster_strategy"], "use_weighting": weighted}
            an, _ = headless.run_nnls(data, p)
            pops = _rows(an.nnls_final_results, "NNLS Peak")
            out.update(n_found=len(pops), rh=[float(x) for x in pops["Rh [nm]"]], table=an.nnls_data)
            if "Abundance [%]" in pops:
                out["abundance"] = [float(x) / 100 for x in pops["Abundance [%]"]]
        elif method == "Reg":
            p = {"decay_times": np.logspace(-8, 1, int(s["reg_points"])), "alpha": s["reg_alpha"],
                 "prominence": s["reg_prominence"], "distance": s["reg_distance"], "normalize": s["reg_normalize"],
                 "fit_beta": s["reg_fit_beta"], "sparsity_penalty": s["reg_sparsity"],
                 "peak_method": s["reg_peak_method"], "use_centroid": s["reg_peak_method"] == "centroid",
                 "distance_threshold": s["cluster_distance"], "clustering_strategy": s["cluster_strategy"],
                 "use_weighting": weighted}
            an, _ = headless.run_regularized(data, p)
            pops = _rows(an.regularized_final_results, "Regularized Peak")
            out.update(n_found=len(pops), rh=[float(x) for x in pops["Rh [nm]"]], table=an.regularized_data)
        else:
            raise ValueError(f"unknown method {method!r}")
    return out
