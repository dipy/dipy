"""Run ANTsPy SyN registration for one fixed/moving image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import ants
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def run_ants_syn(
    fixed_path: str | Path,
    moving_path: str | Path,
    out_dir: str | Path,
    config: dict,
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))

    registration_cfg = config["registration"]
    ants_cfg = config["ants"]

    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform="SyNOnly",
        initial_transform=registration_cfg["initial_transform"],
        syn_metric=ants_cfg["syn_metric"],
        syn_sampling=registration_cfg["cc_radius"],
        reg_iterations=registration_cfg["level_iters"],
        grad_step=registration_cfg["grad_step"],
        flow_sigma=ants_cfg["flow_sigma"],
        total_sigma=ants_cfg["total_sigma"],
        singleprecision=True,
        use_legacy_histogram_matching=False,
        outprefix=str(out_dir / "ants_"),
        verbose=True,
    )

    warped_path = out_dir / "warped_ants.nii.gz"
    ants.image_write(reg["warpedmovout"], str(warped_path))

    return {"warped_image": str(warped_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ANTsPy SyN registration.")
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_ants_syn(
        fixed_path=args.fixed,
        moving_path=args.moving,
        out_dir=args.out_dir,
        config=load_config(args.config),
    )


if __name__ == "__main__":
    main()
