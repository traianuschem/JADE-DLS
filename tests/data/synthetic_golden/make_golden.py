"""Extract the golden subset from the synthetic-data study (documentation of provenance; run manually).

    python tests/data/synthetic_golden/make_golden.py <study>/data/raw

The study computed the reference results with the unmodified JADE-DLS modules (notebook defaults of
JADE-DLS_workingvariant.ipynb). Only compact, per-cell results are kept: settings, ground truth, population-level
results of every method, and the per-file table of Method D. Licence of the data: CC BY-SA 4.0 (see NOTICE.md).
"""
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent

# (sweep, cell key) -- replicate 0 of each; "white" = noise-free, "correlated" = noisy (s = 1 / 2)
CELLS = [
    ("main", "mono_pdi0.05__s0_white"),
    ("main", "mono_pdi0.1__s1_correlated"),
    ("main", "mono_pdi0.2__s2_correlated"),
    ("sizes", "mono_rh25_pdi0.1__s1_correlated"),
    ("sizes", "mono_rh100_pdi0.1__s1_correlated"),
    ("main", "bi_r3_pdi0.1__s1_correlated"),
    ("main", "bi_r5_pdi0.05__s0_white"),
    ("main", "bi_r3_pdi0.1_i80__s1_correlated"),
    ("i20", "bi_r5_pdi0.05_i20__s1_correlated"),
    ("tri", "tri_r3_pdi0.05__s1_correlated"),
]
GROUPS = {"B": ("B",), "cum": ("C2", "C4"), "D": ("D",), "NNLS": ("NNLS",), "Reg": ("Reg",)}
D_KEEP = ("filename", "q^2", "n_modes", "beta", "n_populations", "gamma_mean")


def compact(res, with_table):
    out = {k: res[k] for k in ("n_found", "rh", "abundance", "pdi")}
    if with_table:
        keep = [c for c in res["table"][0] if c in D_KEEP or c.startswith(("gamma_", "tau_"))]
        out["table"] = [{c: row.get(c) for c in keep if c in row} for row in res["table"]]
    return out


def main(raw):
    raw = Path(raw)
    cells = {}
    for sweep, key in CELLS:
        cell = None
        for group, names in GROUPS.items():
            j = json.load(open(raw / sweep / f"{key}_r000_{group}_w.json", encoding="utf-8"))
            if cell is None:
                cell = {k: j[k] for k in ("geometry", "noise_level", "noise_mode", "rep", "truth")}
                cell["sweep"] = sweep
                cell["methods"] = {}
            for name in names:
                cell["methods"][name] = {"weighted": j["weighted"], "settings": j["settings"],
                                         "result": compact(j["results"][name], with_table=(name in ("D", "NNLS", "Reg"))),
                                         "eval": j["eval"][name]}
        cells[key + "_r000"] = cell
    doc = {"_meta": {"source": "ADE-DLS synthetic-data study, reference results by the JADE-DLS modules",
                     "license": "CC BY-SA 4.0", "cells": len(cells)}, "cells": cells}
    text = json.dumps(doc, separators=(",", ":"), allow_nan=True)
    (OUT / "golden.json").write_text(text, encoding="utf-8", newline="\n")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (OUT / "golden.json.sha256").write_text(f"{digest}  golden.json\n", encoding="utf-8", newline="\n")
    print(len(cells), "cells,", len(text) // 1024, "KB, sha256", digest[:12])


if __name__ == "__main__":
    main(sys.argv[1])
