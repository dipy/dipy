"""Evaluate registration outputs with framework-independent metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
from nibabel.processing import resample_from_to
import numpy as np

EPS = 1e-8


def load_image(path: str | Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    img = nib.load(str(path))
    data = np.squeeze(img.get_fdata(dtype=np.float32))
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D image, got shape {data.shape}: {path}")
    return img, data


def same_grid(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4)


def normalize_in_mask(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(data, dtype=np.float32)
    values = data[mask].astype(np.float64)
    out[mask] = (
        (values - values.min()) / max(values.max() - values.min(), EPS)
    ).astype(np.float32)
    return out


def ncc(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    xv = x[mask].astype(np.float64)
    yv = y[mask].astype(np.float64)
    xv -= xv.mean()
    yv -= yv.mean()
    return float(np.dot(xv, yv) / max(np.linalg.norm(xv) * np.linalg.norm(yv), EPS))


def mse(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    diff = x[mask].astype(np.float64) - y[mask].astype(np.float64)
    return float(np.mean(diff * diff))


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    bins: int = 64,
) -> float:
    hist, _, _ = np.histogram2d(x[mask].ravel(), y[mask].ravel(), bins=bins)
    pxy = hist / max(hist.sum(), EPS)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nz = pxy > 0
    independent = np.maximum(px[:, None] * py[None, :], EPS)
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / independent[nz])))


def normalized_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    bins: int = 64,
) -> float:
    hist, _, _ = np.histogram2d(x[mask].ravel(), y[mask].ravel(), bins=bins)
    pxy = hist / max(hist.sum(), EPS)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    hxy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
    return float((hx + hy) / max(hxy, EPS))


def evaluate_candidate(
    fixed: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    fixed_n = normalize_in_mask(fixed, mask)
    candidate_n = normalize_in_mask(candidate, mask)
    return {
        "ncc": ncc(fixed_n, candidate_n, mask),
        "mse": mse(fixed_n, candidate_n, mask),
        "mi": mutual_information(fixed_n, candidate_n, mask),
        "nmi": normalized_mutual_information(fixed_n, candidate_n, mask),
    }


def evaluate_registration(
    pair_id: str,
    fixed_path: str | Path,
    moving_path: str | Path,
    warped_ants_path: str | Path,
    warped_dipy_path: str | Path,
    out_json: str | Path,
) -> dict:
    fixed_img, fixed = load_image(fixed_path)
    moving_img, moving = load_image(moving_path)
    ants_img, ants_warped = load_image(warped_ants_path)
    dipy_img, dipy_warped = load_image(warped_dipy_path)

    if not same_grid(fixed_img, ants_img):
        raise ValueError("ANTs warped image is not on the fixed image grid.")
    if not same_grid(fixed_img, dipy_img):
        raise ValueError("DIPY warped image is not on the fixed image grid.")

    if not same_grid(fixed_img, moving_img):
        moving = np.squeeze(
            resample_from_to(moving_img, fixed_img, order=1).get_fdata(dtype=np.float32)
        )

    candidates = {
        "baseline": moving,
        "ants": ants_warped,
        "dipy": dipy_warped,
    }
    mask = np.isfinite(fixed) & (fixed > 0)
    for candidate in candidates.values():
        mask &= np.isfinite(candidate)
    if not mask.any():
        raise ValueError("Evaluation mask is empty.")

    output = {
        "pair_id": pair_id,
        "metrics": {
            name: evaluate_candidate(fixed, candidate, mask)
            for name, candidate in candidates.items()
        },
    }

    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(output, f, indent=2)

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate registration outputs.")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--warped-ants", required=True)
    parser.add_argument("--warped-dipy", required=True)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_registration(
        pair_id=args.pair_id,
        fixed_path=args.fixed,
        moving_path=args.moving,
        warped_ants_path=args.warped_ants,
        warped_dipy_path=args.warped_dipy,
        out_json=args.out_json,
    )


if __name__ == "__main__":
    main()
