"""
Batch runner for JADE-DLS Method C (standard settings) across all temperatures
of a temperature-dependent measurement series.

Reuses the app's own internal, headless-capable building blocks
(CumulantAnalyzer, ProvenanceRecord, AnalysisView, ResultDetailBlock,
PlotBlock) so that the generated ``<label>degC.md`` / ``<label>degC_prov.json``
/ ``<label>degC_figures/plot_01_1_Method_C_Summary.png`` trio is
structurally identical to what the GUI itself produces -- just without any
manual clicking, and with an automatic (threshold-based) outlier filter in
place of the GUI's visual-inspection filter dialogs.

Run:
    python scripts/batch_method_c.py            # full batch, both cycles
    python scripts/batch_method_c.py --dry-run  # single temperature into a scratch dir
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

# Headless Qt: never actually shows a window, but keep it off-screen-safe.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import ade_dls
from ade_dls.core.parsers import detect_parser
from ade_dls.core.preprocessing import process_correlation_data
from ade_dls.gui.analysis.cumulant_analyzer import CumulantAnalyzer
from ade_dls.gui.core.provenance import ProvenanceRecord

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RAW_BASE = Path(
    r"C:\Users\traja\TUBAF-Nextcloud\RiNe_Supra-muG_PNIPAM-PMAPTAC_6"
    r"\2_RawData\231101_RN_Com-muG_4\sortiert_nach_temperatur"
)
OUT_BASE = Path(
    r"C:\Users\traja\TUBAF-Nextcloud\RiNe_Supra-muG_PNIPAM-PMAPTAC_6"
    r"\3_ProcessedData\DLS"
)

CYCLES = [
    {
        "name": "Zyklus_1_Aufwärmzyklus",
        "raw_dir": RAW_BASE / "Zyklus_1_Aufwärmzyklus",
        "out_dir": OUT_BASE / "TemperatureResult_Dialysed",
        "temps_k": [293, 295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315,
                    319, 323, 327, 331, 335, 339, 343],
    },
    {
        "name": "Zyklus_2_Abkühlzyklus",
        "raw_dir": RAW_BASE / "Zyklus_2_Abkühlzyklus",
        "out_dir": OUT_BASE / "TemperatureResult_Dialysed_Abkuehlzyklus",
        "temps_k": [293, 295, 299, 303, 307, 311, 315, 319, 323, 327, 331, 335, 339],
    },
]

METHOD_C_PARAMS = {
    "fit_limits": (1e-9, 10.0),
    "fit_function": "fit_function4",
    "adaptive_initial_guesses": True,
    "optimizer": "lm",
    "use_multiprocessing": True,
    "initial_parameters": [0.8, 10000.0, 0.0, 0, 0, 0],
    "fit_through_origin": False,
}

FILTER_THRESHOLDS = {
    "cv_threshold": 0.10,             # count-rate coefficient of variation
    "cross_channel_threshold": 0.50,  # relative difference between detector channel means
    "spike_multiple": 2.0,            # single-point spike vs. channel mean
    "baseline_threshold": 0.10,       # |g(2)-1 baseline| at long lag times
    "r2_threshold": 0.99,             # single-exponential pre-screen fit quality
}


def k_to_label(k: float) -> str:
    return f"{round(k - 273.15)}degC"


# ---------------------------------------------------------------------------
# Headless data loading (mirrors DataLoadWorker.run(), without Qt/threads)
# ---------------------------------------------------------------------------

def load_folder_headless(folder: Path) -> dict:
    parser = detect_parser(str(folder))
    if parser is None:
        raise RuntimeError(f"No parser recognised the folder: {folder}")

    files = parser.get_file_list(str(folder))
    if not files:
        raise RuntimeError(f"No data files found in {folder}")

    basedata_frames = []
    countrates = {}
    correlations = {}
    for f in files:
        bd = parser.extract_basedata(f)
        if bd is None or bd.empty:
            print(f"  [WARN] skipping {f}: no basedata")
            continue
        label = parser.get_label(f)
        basedata_frames.append(bd)
        cr = parser.extract_countrates(f)
        if cr is not None:
            countrates[label] = cr
        corr = parser.extract_correlations(f)
        if corr is not None:
            correlations[label] = corr

    basedata = pd.concat(basedata_frames, ignore_index=True)
    return {
        "files": files,                       # absolute paths
        "file_by_label": {parser.get_label(f): f for f in files},
        "basedata": basedata,
        "countrates": countrates,
        "correlations": correlations,
    }


# ---------------------------------------------------------------------------
# Automatic outlier filtering (replaces CountrateFilterDialog / CorrelationFilterDialog)
# ---------------------------------------------------------------------------

def auto_filter_countrates(countrates: dict, thresholds: dict) -> tuple[dict, list]:
    kept, excluded = {}, []
    channel_cols = ["detectorslot 1", "detectorslot 2", "detectorslot 3", "detectorslot 4"]

    for fname, df in countrates.items():
        active_cols = [c for c in channel_cols if c in df.columns and df[c].abs().mean() > 1e-9]
        if not active_cols:
            excluded.append(fname)
            continue

        bad = False
        means = []
        for c in active_cols:
            vals = df[c].to_numpy(dtype=float)
            mean = vals.mean()
            means.append(mean)
            if mean <= 0:
                bad = True
                continue
            cv = vals.std() / mean
            if cv > thresholds["cv_threshold"]:
                bad = True
            if vals.max() > thresholds["spike_multiple"] * mean:
                bad = True

        if len(means) >= 2 and min(means) > 0:
            if (max(means) - min(means)) / min(means) > thresholds["cross_channel_threshold"]:
                bad = True

        if bad:
            excluded.append(fname)
        else:
            kept[fname] = df

    return kept, excluded


def _fit_function1(x, a, b, f):
    return f + a * np.exp(-2 * b * x)


def _prescreen_r2(t: np.ndarray, g: np.ndarray, fit_limits: tuple) -> float:
    lo, hi = fit_limits
    mask = (t >= lo) & (t <= hi) & np.isfinite(g)
    t_fit, g_fit = t[mask], g[mask]
    if len(t_fit) < 8:
        return -np.inf

    tail_n = max(1, int(0.1 * len(g_fit)))
    baseline0 = float(np.mean(g_fit[-tail_n:]))
    a0 = max(float(g_fit[0] - baseline0), 1e-4)

    best_r2 = -np.inf
    for b0 in (10.0, 1e2, 1e3, 1e4, 1e5):
        try:
            popt, _ = curve_fit(
                _fit_function1, t_fit, g_fit,
                p0=[a0, b0, baseline0], maxfev=20000,
            )
            pred = _fit_function1(t_fit, *popt)
            ss_res = float(np.sum((g_fit - pred) ** 2))
            ss_tot = float(np.sum((g_fit - g_fit.mean()) ** 2))
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf
            best_r2 = max(best_r2, r2)
        except Exception:
            continue
    return best_r2


def auto_filter_correlations(correlations: dict, thresholds: dict, fit_limits: tuple) -> tuple[dict, list]:
    kept, excluded = {}, []
    processed = process_correlation_data(correlations, columns_to_remove=None)

    for fname, df in processed.items():
        t = df["t [s]"].to_numpy(dtype=float)
        g = df["g(2)-1"].to_numpy(dtype=float)

        n_tail = max(1, int(0.1 * len(g)))
        baseline = float(np.mean(g[-n_tail:]))

        bad = abs(baseline) > thresholds["baseline_threshold"]
        if not bad:
            r2 = _prescreen_r2(t, g, fit_limits)
            if r2 < thresholds["r2_threshold"]:
                bad = True

        if bad:
            excluded.append(fname)
        else:
            kept[fname] = correlations[fname]  # keep the original raw frame

    return kept, excluded


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def build_provenance_record(
    raw_folder: Path,
    all_data: dict,
    excluded_cr: list,
    excluded_corr: list,
    n_after_cr: int,
    method_c_params: dict,
) -> ProvenanceRecord:
    record = ProvenanceRecord(version=ade_dls.__version__)
    record.set_input_folder(str(raw_folder))

    basedata = all_data["basedata"]
    file_by_label = all_data["file_by_label"]
    for _, row in basedata.iterrows():
        label = row["filename"]
        fpath = file_by_label.get(label)
        if fpath is None:
            continue
        record.add_input_entity(
            fpath,
            metadata={
                "temperature_K": float(row["temperature [K]"]),
                "viscosity_mPas": float(row["viscosity [cp]"]),
            },
        )
    for f in excluded_cr + excluded_corr:
        record.mark_file_excluded(f)

    now = datetime.now().isoformat()
    n_total = len(basedata)

    record.add_activity_from_step({
        "name": "Load Data", "step_type": "custom", "timestamp": now,
        "params": {"data_folder": str(raw_folder), "num_files": n_total},
    })
    record.add_activity_from_step({
        "name": "Extract Data", "step_type": "custom", "timestamp": now,
        "params": {"num_files": n_total},
    })
    record.add_activity_from_step({
        "name": "Filter Countrates", "step_type": "filter", "timestamp": now,
        "params": {
            "filter_type": "countrates",
            "excluded_files": excluded_cr,
            "original_count": n_total,
            "remaining_count": n_after_cr,
            "filter_method": "automatic_threshold",
            "filter_criteria": {
                "cv_threshold": FILTER_THRESHOLDS["cv_threshold"],
                "cross_channel_threshold": FILTER_THRESHOLDS["cross_channel_threshold"],
                "spike_multiple": FILTER_THRESHOLDS["spike_multiple"],
            },
        },
    })
    record.add_activity_from_step({
        "name": "Filter Correlations", "step_type": "filter", "timestamp": now,
        "params": {
            "filter_type": "correlations",
            "excluded_files": excluded_corr,
            "original_count": n_after_cr,
            "remaining_count": n_after_cr - len(excluded_corr),
            "filter_method": "automatic_threshold",
            "filter_criteria": {
                "baseline_threshold": FILTER_THRESHOLDS["baseline_threshold"],
                "r2_threshold": FILTER_THRESHOLDS["r2_threshold"],
                "fit_limits": list(method_c_params["fit_limits"]),
            },
        },
    })
    record.add_activity_from_step({
        "name": "Cumulant Analysis Method C", "step_type": "custom", "timestamp": now,
        "params": {
            "q_range": None,
            "methods": ["C"],
            "method_a_params": {},
            "method_b_params": {},
            "method_c_params": {
                "fit_limits": list(method_c_params["fit_limits"]),
                "fit_function": method_c_params["fit_function"],
                "adaptive_initial_guesses": method_c_params["adaptive_initial_guesses"],
                "optimizer": method_c_params["optimizer"],
                "use_multiprocessing": method_c_params["use_multiprocessing"],
                "initial_parameters": method_c_params["initial_parameters"],
                "fit_through_origin": method_c_params["fit_through_origin"],
            },
        },
    })
    return record


# ---------------------------------------------------------------------------
# Report / plot generation (reuses the real GUI widgets, headlessly)
# ---------------------------------------------------------------------------

def build_report_and_figure(analyzer: CumulantAnalyzer, out_dir: Path, label: str) -> None:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from ade_dls.gui.widgets.analysis_view import AnalysisView
    from ade_dls.gui.widgets.report_panel import ResultDetailBlock, PlotBlock
    import types

    app = QApplication.instance() or QApplication([])

    pipeline_stub = types.SimpleNamespace(data={})
    view = AnalysisView(pipeline_stub)
    view.display_cumulant_results(
        "Method C",
        analyzer.method_c_results,
        plots_dict={"Method C Summary": (analyzer.method_c_summary_plot, {})},
        fit_quality=analyzer.method_c_fit_quality,
        switch_tab=False,
        regression_stats=analyzer.method_c_regression_stats,
        analyzer=analyzer,
    )

    row = view.results_table.rowCount() - 1
    item = view.results_table.item(row, 0)
    html = item.data(Qt.UserRole)
    method_name = item.data(Qt.UserRole + 1)

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    detail_block = ResultDetailBlock({
        "method_name": method_name, "timestamp": now_str, "html": html,
    })

    plot_title = view.plot_list.item(0).text()
    import io
    png_buf = io.BytesIO()
    view.figure.savefig(png_buf, format="png", dpi=96, bbox_inches="tight")
    pdf_buf = io.BytesIO()
    view.figure.savefig(pdf_buf, format="pdf", bbox_inches="tight")
    plot_block = PlotBlock({
        "title": plot_title, "timestamp": now_str,
        "image_bytes": png_buf.getvalue(), "pdf_bytes": pdf_buf.getvalue(),
    })

    figures_dir = out_dir / f"{label}_figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    md_text = (
        detail_block.to_markdown()
        + "\n\n"
        + plot_block.to_markdown(figures_dir=figures_dir, fig_index=1)
        + "\n"
    )
    (out_dir / f"{label}.md").write_text(md_text, encoding="utf-8")

    view.deleteLater()


# ---------------------------------------------------------------------------
# Per-temperature run
# ---------------------------------------------------------------------------

def run_one_temperature(raw_folder: Path, label: str, out_dir: Path) -> None:
    print(f"\n{'=' * 70}\n{label}  <-  {raw_folder}\n{'=' * 70}")

    all_data = load_folder_headless(raw_folder)
    n_total = len(all_data["basedata"])
    print(f"  loaded {n_total} files")

    kept_cr, excluded_cr = auto_filter_countrates(all_data["countrates"], FILTER_THRESHOLDS)
    print(f"  countrate filter: excluded {len(excluded_cr)}/{n_total} -> {excluded_cr}")

    corr_subset = {k: v for k, v in all_data["correlations"].items() if k in kept_cr}
    kept_corr, excluded_corr = auto_filter_correlations(
        corr_subset, FILTER_THRESHOLDS, METHOD_C_PARAMS["fit_limits"]
    )
    print(f"  correlation filter: excluded {len(excluded_corr)}/{len(kept_cr)} -> {excluded_corr}")

    if len(kept_corr) < 3:
        raise RuntimeError(
            f"{label}: only {len(kept_corr)} files survived filtering -- aborting, check data quality"
        )

    filtered_loaded_data = {
        "basedata": all_data["basedata"],
        "correlations": kept_corr,
        "countrates": {k: all_data["countrates"][k] for k in kept_corr if k in all_data["countrates"]},
    }

    analyzer = CumulantAnalyzer(filtered_loaded_data, str(raw_folder))
    analyzer.prepare_basedata()
    analyzer.run_method_c(METHOD_C_PARAMS, q_range=None)
    print(f"  Rh = {analyzer.method_c_results['Rh [nm]'].iloc[0]:.3f} nm, "
          f"R2 = {analyzer.method_c_results['R_squared'].iloc[0]:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    record = build_provenance_record(
        raw_folder, all_data, excluded_cr, excluded_corr, len(kept_cr), METHOD_C_PARAMS
    )
    record.export_prov_json_to_file(str(out_dir / f"{label}_prov.json"))

    build_report_and_figure(analyzer, out_dir, label)
    print(f"  wrote {label}.md / {label}_prov.json / {label}_figures/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                     help="Run only 297K (24degC) into a scratch dir for verification.")
    args = ap.parse_args()

    if args.dry_run:
        scratch = REPO_ROOT / "scripts" / "_dry_run_output"
        run_one_temperature(
            CYCLES[0]["raw_dir"] / "297K", "24degC", scratch
        )
        print(f"\nDry run complete. Inspect: {scratch}")
        return

    for cycle in CYCLES:
        for k in cycle["temps_k"]:
            label = k_to_label(k)
            run_one_temperature(cycle["raw_dir"] / f"{k}K", label, cycle["out_dir"])


if __name__ == "__main__":
    main()
