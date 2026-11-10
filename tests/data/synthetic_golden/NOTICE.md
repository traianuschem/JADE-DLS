# Golden subset of the synthetic-data study

`golden.json` holds reference results of the ADE-DLS synthetic-data study for 10 simulated samples
(replicate 0): monomodal (25/50/100 nm, PDI 0.05–0.2), bimodal (50/50, 20/80, 80/20), trimodal; noise-free and
with noise level s = 1 / 2. Per sample: the ground truth, the settings, the population-level results of the
methods B, C2, C4, D, NNLS and Regularized, and the per-file tables of D, NNLS and Regularized.

* **Reference:** computed with the unmodified JADE-DLS modules (notebook defaults of
  `JADE-DLS_workingvariant.ipynb`, e.g. Method D `gap_threshold` 2.0, noise weighting on, Regularized with
  centroid peaks). The input data are not stored: `ade_dls.benchmark.synthetic` regenerates them from the seeds.
* **Integrity:** `golden.json.sha256`; `make_golden.py` documents how the file was extracted.
* **Read-only:** do not edit the file by hand; a new extraction needs the approval of the project authors.

## Licence

Golden data and the simulated datasets of the benchmark: **CC BY-SA 4.0** —
https://creativecommons.org/licenses/by-sa/4.0/ (attribution, share alike).
The simulator program (`ade_dls.benchmark`) is part of ADE-DLS under GPL-3.0-or-later.
Author line for attribution: Vincent Schildknecht, ORCID https://orcid.org/0009-0007-1856-7784 (simulated datasets and golden data).
