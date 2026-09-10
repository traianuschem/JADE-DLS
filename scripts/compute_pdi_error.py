"""
Re-runs Method C for all 32 temperatures (same loading/filtering as
batch_method_c.py) to additionally extract the per-file PDI values that
JADE-DLS itself only ever averages into a single number. From those we get
a standard deviation and standard error of the mean for the PDI, which
JADE-DLS does not currently report on its own.

Writes an updated CSV next to the existing summary CSV/plot.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from batch_method_c import (
    CYCLES, METHOD_C_PARAMS, FILTER_THRESHOLDS, k_to_label,
    load_folder_headless, auto_filter_countrates, auto_filter_correlations,
)
from ade_dls.gui.analysis.cumulant_analyzer import CumulantAnalyzer

OUT_BASE = Path(
    r"C:\Users\traja\TUBAF-Nextcloud\RiNe_Supra-muG_PNIPAM-PMAPTAC_6"
    r"\3_ProcessedData\DLS"
)


def pdi_stats_for_temperature(raw_folder: Path, label: str) -> dict:
    all_data = load_folder_headless(raw_folder)
    kept_cr, _ = auto_filter_countrates(all_data["countrates"], FILTER_THRESHOLDS)
    corr_subset = {k: v for k, v in all_data["correlations"].items() if k in kept_cr}
    kept_corr, _ = auto_filter_correlations(
        corr_subset, FILTER_THRESHOLDS, METHOD_C_PARAMS["fit_limits"]
    )

    filtered_loaded_data = {
        "basedata": all_data["basedata"],
        "correlations": kept_corr,
        "countrates": {k: all_data["countrates"][k] for k in kept_corr if k in all_data["countrates"]},
    }

    analyzer = CumulantAnalyzer(filtered_loaded_data, str(raw_folder))
    analyzer.prepare_basedata()
    analyzer.run_method_c(METHOD_C_PARAMS, q_range=None)

    merged = pd.merge(analyzer.df_basedata, analyzer.method_c_fit, on="filename", how="inner")
    pdi_per_file = (merged["best_c"] / merged["best_b"] ** 2).dropna()

    n = len(pdi_per_file)
    mean = float(pdi_per_file.mean())
    std = float(pdi_per_file.std(ddof=1)) if n > 1 else float("nan")
    sem = std / np.sqrt(n) if n > 1 else float("nan")

    mean_reported = float(analyzer.method_c_results["PDI"].iloc[0])
    assert abs(mean - mean_reported) < 1e-6, (
        f"{label}: recomputed PDI mean {mean} != reported {mean_reported}"
    )

    print(f"  {label}: n={n}, PDI={mean:.4f}, std={std:.4f}, sem={sem:.4f}")
    return {"n_files": n, "pdi_mean": mean, "pdi_std": std, "pdi_sem": sem}


def main():
    rows = []
    for cycle in CYCLES:
        for k in cycle["temps_k"]:
            label = k_to_label(k)
            print(f"{cycle['name']} {label}")
            stats = pdi_stats_for_temperature(cycle["raw_dir"] / f"{k}K", label)
            rows.append({
                "Zyklus": cycle["name"],
                "Temperatur_C": int(label.replace("degC", "")),
                **stats,
            })

    df = pd.DataFrame(rows).sort_values(["Zyklus", "Temperatur_C"])
    out_csv = OUT_BASE / "TemperaturSerie_PDI_error.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
