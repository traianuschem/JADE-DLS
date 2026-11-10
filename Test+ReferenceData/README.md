# Reference datasets of (J)ADE-DLS

Experimental reference data of the gold-nanoparticle reference samples described in the (J)ADE-DLS paper.
**All datasets in this directory are licensed under CC BY-SA 4.0, see [LICENSE](LICENSE).**

## DLS: `ALV_gold-nanoparticle-reference-data.zip`

Raw data of the angle-resolved DLS measurements on the ALV-7004/USB (wavelength 632.8 nm, 12 s per measurement,
about 298 K, solvent Milli-Q water, viscosity 0.89 cP, refractive index 1.332). Each sample was measured at 13 angles
(30°–150° in 10° steps) with 3 repetitions per angle. The files are the unmodified ALV exports (do not convert line endings).

| Folder | Paper label | Sample (nominal diameter) | Measured |
|---|---|---|---|
| `30nm` | Au1 | 30 nm | 12.03.2026 |
| `100nm` | Au2 | 100 nm | 16.03.2026 |
| `200nm` | Au3 | 200 nm | 20.02.2026 |
| `30nm+100nm` | Au1+Au2 | binary mixture | 17.02.2026 |
| `30nm+200nm` | Au1+Au3 | binary mixture | 11.02.2026 |
| `100nm+200nm` | Au2+Au3 | binary mixture | 17.02.2026 |
| `30nm+100nm+200nm` | Au1+Au2+Au3 | ternary mixture | 20.02.2026 |

Mixing ratios and concentrations: see Table "composition" of the paper.

Per folder (39 measurements = 13 angles × 3 repetitions):

* `<sample>_<angle index>_<repetition>.ASC` — raw correlation function and count rates of a single measurement
  (angle index 0000 = 30° … 0012 = 150°);
* `<sample>_<angle index>_averaged.ASC` — average of the 3 repetitions (ignored by the (J)ADE-DLS parser);
* `*.sta` — ALV statistics file of the series;
* single-component samples only: `Cumulant_Method_A_data_*.txt` and `Cumulant_Method_C_data_*.txt` — results of the
  cumulant methods A and C of the JADE-DLS analysis (reference values for the comparison).

Use: open one folder (e.g. `100nm`) in (J)ADE-DLS; the `averaged` files are skipped automatically.

## SAXS: `SAXSdata/`

SAXS patterns of the three gold standards (30, 100 and 200 nm) and of the citrate buffer and water references,
with the sphere form-factor fits used for the quality control of the reference material (paper, section "Quality control of
gold-reference material: SAXS patterns").

## LS Instruments cross-validation data: `LSInstruments_gold-nanoparticle-reference-data.zip`

The same samples measured on an LS Spectrometer (561 nm, pseudo cross-correlation), including the CORENN results of the
instrument software; see the `README.md` inside the archive. For the mixtures, CORENN was manually optimised only at
50°, 90° and 130°.

## Authors

SAXS data: Richard Neubert (ORCID: https://orcid.org/0009-0006-0594-4379). DLS gold-nanoparticle data: Vincent
Schildknecht (ORCID: https://orcid.org/0009-0007-1856-7784). LS Instruments data: Florian Trummer (ORCID:
https://orcid.org/0000-0002-3831-3825), TU Freiberg. (The simulated datasets of the synthetic benchmark, Vincent
Schildknecht, are in `tests/data/synthetic_golden/`.)

## Relation to the test fixtures

`tests/data/alv_mono/` (byte-identical ALV files of an earlier example set used for the unit tests) is independent of
these data and is described in `tests/data/README.md`.
