# DIPY vs ANTs SyN: Fair Comparison Guide

## Purpose

This document summarizes the current working understanding of the parameter correspondences and implementation differences between DIPY's `SymmetricDiffeomorphicRegistration` and ANTs/ANTsPy SyN registration. It is intended as a guide for designing fair benchmarks and for discussing remaining uncertainties.

The focus here is **SyN-only deformable registration**, not affine, rigid, or full multi-stage registration pipelines.

---

## Scope and Working Assumptions

DIPY and ANTs expose SyN registration through substantially different interfaces:

* **DIPY** uses a dedicated class for SyN-like registration: `SymmetricDiffeomorphicRegistration`.
* **ANTsPy** uses a single high-level `registration()` function whose behavior depends on `type_of_transform`.

For fair SyN-only comparisons, the closest ANTs mode is generally:

```python
ants.registration(..., type_of_transform="SyNOnly")
```

rather than:

```python
ants.registration(..., type_of_transform="SyN")
```

because ANTsPy `"SyN"` includes an affine stage before the deformable SyN stage, whereas DIPY's `SymmetricDiffeomorphicRegistration` only performs the nonlinear symmetric diffeomorphic optimization.

---

## Parameter Equivalences

| DIPY                      | ANTs / ANTsPy                                               | Notes                                                                                                                      |
| ------------------------- | ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `level_iters`             | `reg_iterations`                                            | Both control the number of nonlinear SyN iterations per pyramid level.                                                     |
| `step_length`             | `grad_step`                                                 | Both control the update magnitude, but the update normalization/composition details may differ.                                |
| `static` in `optimize()`  | `fixed`                                                     | Reference image.                                                                                                           |
| `moving` in `optimize()`  | `moving`                                                    | Image to be registered to the reference.                                                                                   |
| `prealign`                | `initial_transform`                                         | Both used for prealignment, can be set to identity for fair comparison |
| `CCMetric(radius=...)`    | `syn_sampling` when `syn_metric="CC"`                       | In CC, `syn_sampling` corresponds to the local neighborhood radius.                                                        |
| `opt_tol`                 | ANTs convergence threshold hardcoded to `1e-7`                     | Need to inspect ITK/ANTs convergence formula before claiming direct equivalence.                                           |
| DIPY `energy_window = 12` | ANTs convergence window hardcoded to 8. | Both relate to convergence monitoring, but formulas and interpretation need confirmation.                                  |


---

## Pyramid Construction: Smoothing and Scaling

### ANTs / ANTsPy

For the SyN stage, ANTsPy computes scalar smoothing sigmas and nominal shrink factors from the number of pyramid levels:

```text
iterations_i      = reg_iterations[i]
smoothing_sigma_i = L - 1 - i
shrink_factor_i   = 2 ** (L - 1 - i)
```

Example:

```text
reg_iterations = [40, 20, 20]
L = 3
smoothing sigmas = [2, 1, 0]
shrink factors   = [4, 2, 1]
```

Later, ANTs converts the nominal shrink factor into per-dimension integer shrink factors, taking image spacing into account. The goal is to keep the downsampled image spacing as close as possible to isotropic while using integer shrink factors.

### DIPY

DIPY constructs a `ScaleSpace` object. For each pyramid level, it starts from the same conceptual scale progression:

```text
scale_i = 2 ** i
```

but then computes per-dimension scaling using the minimum input spacing:

```text
scaling[d] = 2**i * min_spacing / input_spacing[d]
```

This gives:

```text
output_spacing[d] = input_spacing[d] * scaling[d]
                  = 2**i * min_spacing
```

Therefore, DIPY allows non-integer per-dimension scaling values and effectively forces the target spacing at each level to be isotropic in physical units.

For smoothing, DIPY computes voxel-unit Gaussian sigmas as:

```text
sigma[d] = sigma_factor * (output_spacing[d] / input_spacing[d] - 1)
```

For isotropic data, this simplifies to:

```text
sigma_i = sigma_factor * (2**i - 1)
```

Thus, DIPY's default smoothing can be much weaker than ANTs' default smoothing. For three isotropic levels and `sigma_factor = 0.2`, DIPY gives fine-to-coarse sigmas:

```text
[0, 0.2, 0.6]
```

whereas ANTs uses:

```text
[0, 1, 2]
```

fine-to-coarse.

---

## Update Smoothing and Deformation Regularization

### ANTs

ANTs separates deformation regularization into two explicit SyN parameters:

```text
SyN[grad_step, flow_sigma, total_sigma]
```

* `flow_sigma` smooths the **update field**, i.e. the incremental displacement estimated at the current iteration before it is composed into the running deformation.
* `total_sigma` smooths the **accumulated total field**, i.e. the running deformation after update composition.

### DIPY

DIPY's `SymmetricDiffeomorphicRegistration` does not expose direct equivalents of either `flow_sigma` or `total_sigma`.

Instead, update smoothing is implemented inside metric classes, before the update field is normalized and composed by the SyN optimizer. These parameters are therefore at most conceptually similar to ANTs' `flow_sigma`, not `total_sigma`.

There is currently no direct DIPY equivalent of ANTs' `total_sigma`.

---

## Masks

ANTs supports explicit metric masks:

```text
mask
moving_mask
mask_all_stages
```

These can restrict where the metric is evaluated during optimization.

DIPY `SymmetricDiffeomorphicRegistration` does not expose equivalent fixed/moving metric masks. The closest related mechanism is `metric.mask0`, used by `ScaleSpace` to keep zero-valued regions zero after smoothing. This is not equivalent to ANTs' fixed/moving registration masks.

---

## Other ANTs Features Without Direct DIPY Equivalents

| ANTs parameter                  | Relevance to SyN                                           | DIPY equivalent                                              |
| ------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------ |
| `restrict_transformation`       | Can restrict deformation components along selected axes    | No built-in equivalent                                       |
| `multivariate_extras`           | Adds weighted extra metrics during deformable optimization | No built-in equivalent; would require custom combined metric |
| `use_legacy_histogram_matching` | Legacy intensity preprocessing option; not recommended     | No direct equivalent                                         |

---

## Intensity Preprocessing / Normalization

Current working summary:

* ANTs/ITK appears to build a 256-bin histogram, estimate intensity bounds using lower and upper quantiles, and use those bounds for min/max-like normalization before constructing coarser smoothed levels.
* DIPY performs direct min/max normalization to `[0, 1]` in `ScaleSpace`, and does this again after smoothing at each coarse level.

---

## Inverse Field Computation

Both DIPY and ANTs/ITK SyN maintain inverse displacement fields during optimization.

### DIPY

DIPY exposes inverse-field inversion parameters directly:

```text
inv_iter = 20
inv_tol  = 1e-3
```

These are passed to the displacement-field inversion routine during optimization.

### ANTs / ITK SyN

ANTs/ITK SyN also uses iterative inverse displacement field estimation. The apparent internal settings are:

```text
maximum inverse iterations = 20
mean error tolerance       = 0.001
max error tolerance        = 0.1
```

ANTsPy `registration()` does not expose these inverse-field inversion parameters.

---

## Potential Implementation Detail to Verify

There may be a possible issue in DIPY's `CCMetric.compute_backward()`:

```text
compute_backward() assigns the returned energy to a local variable `energy`, whereas compute_forward() stores it as `self.energy`.
```

If true, `get_energy()` after the backward computation may not reflect the backward energy.

---

## Recommended Benchmarking Strategy

1. **Start with SyN-only comparisons**

   * ANTs: `type_of_transform="SyNOnly"`
   * DIPY: `SymmetricDiffeomorphicRegistration`

2. **Use the same metric family where possible**

   * Best first comparison: ANTs `syn_metric="CC"` vs DIPY `CCMetric`.
   * Map `syn_sampling` to `CCMetric.radius`.

3. **Control pyramid levels explicitly**

   * Match `reg_iterations` and `level_iters`.
   * Be aware that smoothing and scaling schedules are still not identical.

4. **Minimize convergence stopping differences initially**

   * Consider forcing full iteration execution by setting very strict or very permissive convergence thresholds, depending on interface possibilities.
   * Revisit convergence once other differences are controlled.

5. **Set ANTs `total_sigma=0` for fairer comparison**

   * DIPY does not expose a corresponding total-field smoothing parameter.

6. **Treat update smoothing carefully**

   * For CC, compare ANTs `flow_sigma` with DIPY `CCMetric.sigma_diff`, but remember this is only conceptual.

7. **Avoid masks, multivariate metrics, and restricted transforms in the first benchmark**

   * These have no direct DIPY equivalents.

---

## Open Questions

* Why exactly does the SyN formulation update forward-to-middle fields and maintain inverse fields for warping, instead of directly optimizing pull fields?
