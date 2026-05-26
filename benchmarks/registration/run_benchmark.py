"""Run the registration benchmark pipeline for a CSV of fixed/moving pairs.

Pipeline:

1. Reslice both images by the same downsampling factor.
2. Skull-strip both images with SynthSeg.
3. Fill holes and keep the largest connected mask component.
4. Rigidly prealign the moving image to the fixed image.
5. Run DIPY SyN and ANTs SyN from the same prealigned inputs.
6. Evaluate both warped outputs with framework-independent metrics.

Example, from benchmarks/registration:

    python run_benchmark.py \
        --pairs data/oasis2_pairs.csv \
        --config configs/syn_cc_default.yaml \
        --out-dir outputs/oasis2_syn_cc_ds2 \
        --downsample-factor 2 \
        --n 5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import statistics

from evaluate import evaluate_registration
import nibabel as nib
import numpy as np
from runners.ants_syn import run_ants_syn
from runners.dipy_syn import run_dipy_syn
from scipy.ndimage import binary_fill_holes, label
import yaml

from dipy.align.imaffine import (
    AffineRegistration,
    MutualInformationMetric,
    transform_centers_of_mass,
)
from dipy.align.reslice import reslice
from dipy.align.transforms import RigidTransform3D, TranslationTransform3D

_SYNTHSEG_MODEL = None
METRIC_NAMES = ("ncc", "mse", "nmi")


def load_yaml(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def read_pairs(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"fixed_path", "moving_path"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise ValueError(f"Pair CSV is missing columns: {sorted(missing)}")
    return rows


def sample_pairs(
    rows: list[dict[str, str]], n: int | None, seed: int
) -> list[dict[str, str]]:
    if n is None:
        return rows
    if n > len(rows):
        raise ValueError(f"Requested n={n}, but only found {len(rows)} pairs.")
    return random.Random(seed).sample(rows, n)


def get_pair_id(row: dict[str, str], index: int) -> str:
    pair_id = row.get("pair_id", "").strip()
    return pair_id or f"pair_{index:04d}"


def reslice_by_factor(in_path: str | Path, out_path: str | Path, factor: float) -> Path:
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if factor == 1:
        return in_path
    if out_path.exists():
        print(f"Reusing resliced image: {out_path}")
        return out_path

    print(f"Reslicing image: {in_path}")
    img = nib.load(str(in_path))
    data = np.squeeze(np.asarray(img.dataobj, dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {data.shape}: {in_path}")

    zooms = img.header.get_zooms()[:3]
    new_zooms = tuple(float(zoom) * factor for zoom in zooms)
    data_rs, affine_rs = reslice(data, img.affine, zooms, new_zooms)

    out_img = nib.Nifti1Image(data_rs.astype(np.float32), affine_rs)
    nib.save(out_img, str(out_path))
    return out_path


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    components, n_components = label(mask)
    if n_components <= 1:
        return mask.astype(bool)

    counts = np.bincount(components.ravel())
    counts[0] = 0
    return components == int(counts.argmax())


def get_synthseg_model(use_cuda: bool):
    global _SYNTHSEG_MODEL
    if _SYNTHSEG_MODEL is None:
        from dipy.nn.torch.synthseg import SynthSeg

        _SYNTHSEG_MODEL = SynthSeg(verbose=False, use_cuda=use_cuda)
    return _SYNTHSEG_MODEL


def skull_strip(
    in_path: str | Path,
    out_img_path: str | Path,
    out_mask_path: str | Path,
    *,
    use_cuda: bool,
) -> Path:
    out_img_path = Path(out_img_path)
    out_mask_path = Path(out_mask_path)
    if out_img_path.exists() and out_mask_path.exists():
        print(f"Reusing skull-stripped image: {out_img_path}")
        return out_img_path

    print(f"Skull stripping image: {in_path}")
    out_img_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.load(str(in_path))
    data = np.squeeze(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"SynthSeg expects a 3D image, got {data.shape}: {in_path}")

    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    _, _, mask = get_synthseg_model(use_cuda).predict(data, img.affine)
    mask = binary_fill_holes(mask.astype(bool))
    mask = keep_largest_component(mask)
    brain = data * mask

    brain_img = nib.Nifti1Image(brain.astype(np.float32), img.affine)
    nib.save(brain_img, str(out_img_path))

    mask_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine)
    nib.save(mask_img, str(out_mask_path))

    return out_img_path


def rigid_prealign(
    fixed_path: str | Path, moving_path: str | Path, out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    if out_path.exists():
        print(f"Reusing rigid prealignment: {out_path}")
        return out_path

    print("Running rigid prealignment")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))
    fixed = fixed_img.get_fdata(dtype=np.float32)
    moving = moving_img.get_fdata(dtype=np.float32)

    center = transform_centers_of_mass(
        fixed, fixed_img.affine, moving, moving_img.affine
    )

    metric = MutualInformationMetric(nbins=32, sampling_proportion=None)
    affreg = AffineRegistration(
        metric=metric,
        level_iters=[10000, 1000, 100],
        sigmas=[3.0, 1.0, 0.0],
        factors=[4, 2, 1],
    )

    translation = affreg.optimize(
        fixed,
        moving,
        TranslationTransform3D(),
        params0=None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=center.affine,
    )

    rigid = affreg.optimize(
        fixed,
        moving,
        RigidTransform3D(),
        params0=None,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        starting_affine=translation.affine,
    )

    prealigned = rigid.transform(moving)

    nib.save(
        nib.Nifti1Image(
            prealigned.astype(np.float32), fixed_img.affine, fixed_img.header
        ),
        str(out_path),
    )

    return out_path


def prepare_pair(
    row: dict[str, str],
    pair_out: Path,
    *,
    downsample_factor: float,
    use_cuda: bool,
) -> tuple[Path, Path]:
    prepared_out = pair_out / "prepared"
    inputs_out = prepared_out / "inputs"
    skullstrip_out = prepared_out / "skullstrip"

    print("Preparing inputs")
    fixed_resliced = reslice_by_factor(
        row["fixed_path"],
        inputs_out / "fixed_resliced.nii.gz",
        downsample_factor,
    )
    moving_resliced = reslice_by_factor(
        row["moving_path"],
        inputs_out / "moving_resliced.nii.gz",
        downsample_factor,
    )

    fixed_brain = skull_strip(
        fixed_resliced,
        skullstrip_out / "fixed_brain.nii.gz",
        skullstrip_out / "fixed_mask.nii.gz",
        use_cuda=use_cuda,
    )
    moving_brain = skull_strip(
        moving_resliced,
        skullstrip_out / "moving_brain.nii.gz",
        skullstrip_out / "moving_mask.nii.gz",
        use_cuda=use_cuda,
    )

    moving_prealigned = rigid_prealign(
        fixed_brain,
        moving_brain,
        pair_out / "prealign" / "moving_rigid_to_fixed.nii.gz",
    )
    return fixed_brain, moving_prealigned


def gains_vs_baseline(metrics: dict) -> dict:
    baseline = metrics["baseline"]
    return {
        method: {metric: values[metric] - baseline[metric] for metric in METRIC_NAMES}
        for method, values in metrics.items()
        if method != "baseline"
    }


def summarize(samples: list[dict]) -> dict:
    methods = sorted({method for sample in samples for method in sample["metrics"]})
    summary = {}

    for method in methods:
        method_summary = {"n": len(samples)}
        for metric in METRIC_NAMES:
            values = [
                sample["metrics"][method][metric]
                for sample in samples
                if method in sample["metrics"]
            ]
            method_summary[metric] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }

            if method == "baseline":
                continue

            gain_values = [
                sample["gains_vs_baseline"][method][metric]
                for sample in samples
                if method in sample["gains_vs_baseline"]
            ]
            method_summary[f"{metric}_gain_vs_baseline"] = {
                "mean": statistics.mean(gain_values),
                "std": statistics.stdev(gain_values) if len(gain_values) > 1 else 0.0,
            }
        summary[method] = method_summary

    return summary


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def run_pair(
    pair_id: str,
    row: dict[str, str],
    out_dir: Path,
    config: dict,
    args: argparse.Namespace,
) -> dict:
    pair_out = out_dir / pair_id
    fixed, moving = prepare_pair(
        row,
        pair_out,
        downsample_factor=args.downsample_factor,
        use_cuda=args.use_cuda,
    )

    print("Running DIPY SyN")
    dipy_result = run_dipy_syn(fixed, moving, pair_out / "dipy", config)

    print("Running ANTs SyN")
    ants_result = run_ants_syn(fixed, moving, pair_out / "ants", config)

    print("Evaluating registration outputs")
    return evaluate_registration(
        pair_id=pair_id,
        fixed_path=fixed,
        moving_path=moving,
        fixed_mask_path=pair_out / "prepared" / "skullstrip" / "fixed_mask.nii.gz",
        warped_ants_path=ants_result["warped_image"],
        warped_dipy_path=dipy_result["warped_image"],
        out_json=pair_out / "evaluation.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DIPY vs ANTs registration benchmark."
    )
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out-dir", default=Path("outputs/registration_benchmark"), type=Path
    )
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--downsample-factor", type=float, default=1.0)
    parser.add_argument("--use-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    pairs = sample_pairs(read_pairs(args.pairs), args.n, args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "pairs_file": str(args.pairs),
            "config_file": str(args.config),
            "config": config,
            "n_pairs": len(pairs),
            "seed": args.seed,
            "downsample_factor": args.downsample_factor,
        },
        "samples": [],
        "summary": {},
    }

    for index, row in enumerate(pairs, start=1):
        pair_id = get_pair_id(row, index)
        print(f"\n=== {pair_id} ===")
        if not row.get("pair_id", "").strip():
            print(f"fixed:  {row['fixed_path']}")
            print(f"moving: {row['moving_path']}")

        evaluation = run_pair(pair_id, row, args.out_dir, config, args)
        results["samples"].append(
            {
                "pair_id": pair_id,
                "fixed_path": row["fixed_path"],
                "moving_path": row["moving_path"],
                "metrics": evaluation["metrics"],
                "gains_vs_baseline": gains_vs_baseline(evaluation["metrics"]),
            }
        )
        results["summary"] = summarize(results["samples"])
        write_json(args.out_dir / "benchmark_results.json", results)

    print(f"\nDone. Results saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
