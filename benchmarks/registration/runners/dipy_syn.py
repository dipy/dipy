"""Run DIPY SyN registration for one fixed/moving image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

from dipy.align.imwarp import SymmetricDiffeomorphicRegistration
from dipy.align.metrics import CCMetric


def load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def as_3d(data: np.ndarray, path: str | Path) -> np.ndarray:
    data = np.squeeze(data)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got {data.shape}: {path}")
    return data


def run_dipy_syn(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    config: dict,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))

    fixed = as_3d(fixed_img.get_fdata(dtype=np.float32), fixed_path)
    moving = as_3d(moving_img.get_fdata(dtype=np.float32), moving_path)

    registration_cfg = config["registration"]
    dipy_cfg = config["dipy"]

    metric = CCMetric(
        3,
        radius=registration_cfg["cc_radius"],
        sigma_diff=dipy_cfg["update_field_sigma"],
    )

    sdr = SymmetricDiffeomorphicRegistration(
        metric,
        level_iters=registration_cfg["level_iters"],
        step_length=registration_cfg["grad_step"],
        ss_sigma_factor=dipy_cfg["ss_sigma_factor"],
        opt_tol=registration_cfg["convergence_tol"],
        inv_iter=dipy_cfg["inv_iter"],
        inv_tol=dipy_cfg["inv_tol"],
    )
    sdr.energy_window = registration_cfg["convergence_window"]

    mapping = sdr.optimize(
        fixed,
        moving,
        static_grid2world=fixed_img.affine,
        moving_grid2world=moving_img.affine,
        prealign=None,
    )

    warped = mapping.transform(moving)

    warped_path = out_dir / "warped_dipy.nii.gz"

    nib.save(nib.Nifti1Image(warped.astype(np.float32), fixed_img.affine), warped_path)

    return {"warped_image": str(warped_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DIPY SyN registration.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dipy_syn(
        fixed_path=args.fixed,
        moving_path=args.moving,
        out_dir=args.out_dir,
        config=load_config(args.config),
    )


if __name__ == "__main__":
    main()
