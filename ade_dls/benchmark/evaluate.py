"""Evaluation of a method result against the ground truth (rule of the synthetic-data study)."""


def evaluate(result, truth, tol=0.20):
    """Compare one method result with the ground truth.

    result : {"method", "n_found", "rh" (sorted ascending)}
    count_ok  : number of populations found == true number
    success   : count_ok and every population's Rh within +-tol of its true Rh (sorted matching)
    spurious  : more populations found than present
    rel_err   : per-population relative Rh error (found/true - 1), sorted matching, if the count fits
    rel_err_z : relative error of the single apparent Rh against the true z-average Rh (B, C2, C4)
    """
    rh, n_true = result["rh"], truth["n_pop"]
    n = result["n_found"]
    ev = {"count_ok": None, "success": None, "spurious": None, "rel_err": [], "rel_err_z": None}
    if n is None:
        return ev
    ev["count_ok"] = bool(n == n_true)
    ev["spurious"] = bool(n > n_true)
    if len(rh) == n_true:
        ev["rel_err"] = [float(r / t - 1) for r, t in zip(rh, truth["rh_pops"])]
    ev["success"] = bool(ev["count_ok"] and len(ev["rel_err"]) == n_true
                         and all(abs(x) <= tol for x in ev["rel_err"]))
    if result["method"] in ("B", "C2", "C4") and rh:
        # reference of the cumulant methods (free-intercept regression): the z-average (angle-independent
        # fractions) or, for angle-dependent fractions, the slope of the exact mean decay rates
        ref = truth.get("rh_z_reg_free", truth["rh_z"])
        ev["rel_err_z"] = float(rh[0] / ref - 1)
    return ev
