# Analysis Methods

ADE-DLS implements seven analysis methods for DLS data. All methods operate on the field autocorrelation function g₁(τ), extracted from the measured intensity autocorrelation g₂(τ) via the Siegert relation:

```
g₂(τ) = 1 + β · |g₁(τ)|²
```

where β is the intercept (coherence factor, ideally ≈ 1 for ideal optics).

The scattering vector q is computed from the scattering angle θ, laser wavelength λ (in vacuo), and solvent refractive index n:

```
q = (4π n / λ) · sin(θ/2)
```

The diffusion coefficient D is obtained from the measured decay rate Γ:

```
D = Γ / q²
```

The hydrodynamic radius Rh follows from the Stokes-Einstein equation:

```
Rh = kB T / (6π η D)
```

where kB is Boltzmann's constant, T is temperature (K), and η is solvent viscosity.

---

## Method A — ALV Software Cumulants

**Model:** Uses the cumulant fit results (1st, 2nd, 3rd order decay rates Γ) already computed by the instrument software (ALV correlator `.asc` files, or the per-order Rh values from LS Instruments exports).

No correlation-function fitting is performed in ADE-DLS; instead the pre-computed per-file Γ values are regressed against q² to obtain D (and Rh via Stokes-Einstein). For LS Instruments "radius" exports, the per-order Rh values are averaged directly.

**When to use:** Quick cross-check when the instrument software's own cumulant fits are trusted.

**Output:** Γ (per order), D, Rh, PDI (2nd/3rd order), skewness (3rd order), R²

---

## Method B — Linear Cumulant

**Model:**

```
ln[g₁(τ)] = −Γ τ + (μ₂/2!) τ² − (μ₃/3!) τ³ + …
```

A linear least squares fit of ln[g₁(τ)] vs. τ over a user-defined range [τ_min, τ_max]. The slope gives −Γ; higher-order terms give cumulants μ₂ (PDI = μ₂/Γ²), μ₃, …

**Limitations:** Requires a good baseline correction. Sensitive to noise at long lag times. Best used as a quick sanity check or for monodisperse samples.

**Output:** Γ, D, Rh, PDI (= μ₂/Γ²), R²

---

## Method C — Iterative Non-Linear Cumulant

**Model:** Same cumulant expansion as Method B, but fitted non-linearly to g₁(τ) itself (not the log). Supports 2nd, 3rd, or 4th order fits with adaptive initial parameter estimates and iterative outlier rejection.

**Algorithm:**
1. Initial fit with scipy `curve_fit` (Levenberg-Marquardt)
2. Residual-based outlier detection
3. Re-fit on cleaned data
4. Repeat until convergence or max iterations

**Advantages over Method B:** More robust to noise, better uncertainty estimation, works with asymmetric distributions.

**Output:** Γ, D, Rh, PDI, cumulant order, R², fitted curve

---

## Method D — Multi-Exponential Decomposition

**Model:**

```
g₁(τ) = Σᵢ Aᵢ · exp(−Γᵢ τ)
```

Fits g₁(τ) to a sum of two or more exponentials. Each exponential corresponds to one particle population with its own Γᵢ, Dᵢ, and Rhᵢ.

**Clustering:** After fitting all angles, Ward hierarchical clustering groups populations across angles by their log₁₀(D) values. The cluster distance threshold can be tuned in the post-fit dialog.

**When to use:** When PDI > 0.2 or when you expect two distinct populations (e.g., monomer + aggregate).

**Output per population:** Γᵢ, Dᵢ, Rhᵢ, amplitude Aᵢ, relative intensity fraction; cluster assignment per angle

---

## NNLS — Non-Negative Least Squares Inverse Laplace

**Model:**

```
g₁(τ) = ∫ G(Γ) exp(−Γ τ) dΓ
```

Discretized to a logarithmic grid of N relaxation times. The coefficients G(Γᵢ) are constrained to be non-negative. Solved via the NNLS algorithm (Lawson & Hanson).

**No regularization** is applied; the solution tends to be sparse (few non-zero components).

**Peak detection:** Local maxima in G(Γ) are identified as populations. Each peak gives Γ, D, and Rh.

**Clustering:** Same Ward hierarchical clustering as Method D for multi-angle data.

**Output:** Full discrete distribution G(Γ), detected peak positions with D and Rh, cluster assignments

---

## Regularized NNLS — Tikhonov-Phillips Regularization

**Model:** Same as NNLS, but with an added smoothness penalty:

```
minimize ‖A·x − b‖² + α · ‖L·x‖²
subject to x ≥ 0
```

where L is a second-difference operator (smoothness constraint) and α is the regularization parameter.

**Selecting α:**
- **L-curve**: Plot log‖residual‖ vs. log‖smoothness‖; optimal α is at the corner
- **GCV (Generalized Cross-Validation)**: Minimizes prediction error; automated suggestion

Both are accessible in the **alpha analysis dialog** (post-fit refinement).

**Additional constraints** (configurable in Parameters):
- Normalization (∑G = 1)
- Sparsity (L1 penalty)
- Unimodality

**Output:** Smooth distribution G(Γ), populations with D and Rh, α used, R², L-curve data

---

## Static Light Scattering (SLS)

Available as a post-processing step after Regularized NNLS.

**Intensity decomposition:** The total scattered intensity at each angle is apportioned to populations identified by the Regularized NNLS run, weighted by their amplitude fractions. Monitor-diode correction is applied to normalize for laser power drift.

**Guinier analysis** per population and for total intensity:

```
ln[I(q)] ≈ ln[I₀] − (Rg² / 3) · q²
```

Fit over the range qRg < 1.3 (Guinier regime). Returns I₀, Rg, qRg_max, R².

**Number-weighting correction:** The Guinier fit itself always runs on the real, intensity-weighted per-angle intensities. Afterwards, the extrapolated I₀ values are converted to number/concentration fractions using a configurable Rh exponent (c ∝ I₀/Rh^exponent; 6 = Rayleigh/compact spheres, 5 = Daoud-Cotton/star polymers), always computed alongside the intensity-weighted result — applying this correction to the raw per-angle intensities before the fit (as earlier versions did) would corrupt the q-dependence the Guinier fit relies on.

**Output:** Guinier plots per population, I₀, Rg, qRg_max, R² for each population and total

---

## Noise Weighting (Biganzoli & Ferri)

Optional, off by default. Real multi-tau DLS data is heteroscedastic: the
statistical uncertainty of each lag-time channel varies strongly with τ —
shot-noise dominated at the shortest gate times, decreasing as longer bins
accumulate more photon counts, and rising again near the measuring-time
limit. Enabling "Use noise weighting" in the Cumulant B/C/D, NNLS and
Regularized dialogs weights each fit residual by the channel's true noise
variance, computed from the Biganzoli & Ferri (2018) correction to the
Schätzel (1990) formula (a correction for "triangular averaging", the
finite-sampling-time effect of multi-tau correlators).

Requires the Duration, MeanCR0 and MeanCR1 metadata from the ALV file
header (extracted automatically alongside angle/temperature/etc.); the
checkbox is disabled with an explanatory tooltip when this metadata is
unavailable (e.g. LS Instruments data, or files with an incomplete header).
Weights are computed once per (filtered) dataset and shared across all
analyses; whether a given fit actually uses them is a separate, per-run
choice. Method A (instrument-software cumulants) is unaffected — it reads
pre-computed values and does not refit the correlation function.

Weighted fits report an additional `weighted_R_squared` alongside the
usual (unweighted) `R_squared`, and a `weighted` flag column.

Reference: Biganzoli & Ferri, "Statistical analysis of dynamic light
scattering data: revisiting and beyond the Schätzel formulas", *Optics
Express* **26**(22), 29375–29395 (2018).

---

## Choosing a Method

| Sample type | Recommended method |
|-------------|-------------------|
| Monodisperse, clean | Cumulant C |
| Monodisperse, quick check | Cumulant B |
| Bimodal (two populations) | Cumulant D or NNLS |
| Polydisperse (broad distribution) | Regularized NNLS |
| Size distribution needed | Regularized NNLS |
| Aggregate detection | NNLS + Cumulant D comparison |
| Static scattering / Rg | SLS (requires Regularized NNLS) |
