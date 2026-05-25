# Registration Benchmark

## Purpose

This folder contains a small benchmarking framework for comparing DIPY and
ANTsPy SyN registration on the same fixed/moving image pairs.

The benchmark is designed to keep dataset-specific logic separate from the
registration pipeline. Dataset adapters should produce a generic pair CSV, and the
main benchmark script consumes only:

```text
fixed_path,moving_path
```

An optional `pair_id` column can be provided to control output folder names.

---

## Pipeline

For each pair, `run_benchmark.py` applies the same preprocessing before running
either backend:

1. Reslice both images by the same downsampling factor.
2. Skull-strip both images with SynthSeg.
3. Fill holes and keep the largest connected mask component.
4. Rigidly prealign the moving image to the fixed image.
5. Run DIPY SyN and ANTsPy SyN from the same prealigned inputs.
6. Evaluate both warped outputs with framework-independent metrics.

The current evaluator computes:

```text
ncc
mse
mi
nmi
```

Metrics are written per pair and progressively aggregated into:

```text
benchmark_results.json
```

This file contains run metadata, per-sample metrics, and overall summary
statistics.

---

## Configuration

Benchmark parameters are stored in YAML files under `configs/`.

Each configuration is split into:

```text
registration  shared benchmark intent and common parameter values
dipy          DIPY-specific parameters
ants          ANTsPy-specific parameters
```

The parameter mapping follows the notes in
[DIPY vs ANTs SyN: Fair Comparison Guide](dipy_ants_syn_comparison.md).

---

## Dataset Adapters

Dataset-specific code should live under `datasets/` and produce the generic
pair CSV consumed by `run_benchmark.py`.

An OASIS-2 example adapter is available at
[datasets/oasis2](datasets/oasis2/README.md).

---

## Usage

Run a small benchmark from `benchmarks/registration`:

```powershell
python run_benchmark.py `
  --pairs data/oasis2_pairs.csv `
  --config configs/syn_cc_default.yaml `
  --out-dir outputs/oasis2_syn_cc_ds2 `
  --downsample-factor 2 `
  --n 5
```

Use `--use-cuda` to run SynthSeg with CUDA.
