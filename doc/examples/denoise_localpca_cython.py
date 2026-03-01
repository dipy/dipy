from __future__ import annotations

from time import time

import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.denoise._localpca_fast import genpca_core
import dipy.denoise.localpca as localpca_mod
from dipy.denoise.localpca import localpca
from dipy.denoise.pca_noise_estimate import pca_noise_estimate
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti


def compare_arrays(a: np.ndarray, b: np.ndarray, rtol: float, atol: float) -> None:
    diff = a - b
    absd = np.abs(diff)
    print(
        f"\nallclose={np.allclose(a, b, rtol=rtol, atol=atol)} "
        f"rtol={rtol:g} atol={atol:g}"
    )
    print(f"max_abs={absd.max():.6e}  mean_abs={absd.mean():.6e}")


def make_genpca_fast():
    def genpca_fast(
        arr,
        *,
        sigma=None,
        mask=None,
        patch_radius=2,
        tau_factor=None,
        return_sigma=False,
        out_dtype=None,
        pca_method="eig",
        suppress_warning=False,
    ):
        if isinstance(patch_radius, (int, np.integer)):
            prx = pry = prz = int(patch_radius)
        else:
            pr = np.asarray(patch_radius).astype(int)
            if pr.shape != (3,):
                raise ValueError("patch_radius must be int or length-3 array")
            prx, pry, prz = int(pr[0]), int(pr[1]), int(pr[2])

        return genpca_core(
            arr,
            mask=mask,
            sigma=sigma,
            patch_radius_arr_x=prx,
            patch_radius_arr_y=pry,
            patch_radius_arr_z=prz,
            tau_factor=tau_factor,
            return_sigma=return_sigma,
            out_dtype=out_dtype,
        )

    return genpca_fast


def main():
    dwi_fname, dwi_bval_fname, dwi_bvec_fname = get_fnames(name="isbi2013_2shell")
    data, _aff = load_nifti(dwi_fname)
    bvals, bvecs = read_bvals_bvecs(dwi_bval_fname, dwi_bvec_fname)
    gtab = gradient_table(bvals, bvecs=bvecs)

    print("Input", data.shape, data.dtype)

    t0 = time()
    sigma = pca_noise_estimate(data, gtab, correct_bias=True, smooth=3)
    t_sigma = time() - t0
    print(f"Sigma estimation time: {t_sigma:.3f}s")

    # Reference run
    t0 = time()
    den_ref = localpca(data, sigma=sigma, tau_factor=2.3, patch_radius=2)
    t_ref = time() - t0
    print(f"Reference localpca time: {t_ref:.3f}s")

    # Patched run
    old_genpca = localpca_mod.genpca
    localpca_mod.genpca = make_genpca_fast()
    try:
        t0 = time()
        den_fast = localpca(data, sigma=sigma, tau_factor=2.3, patch_radius=2)
        t_fast = time() - t0
        print(f"Patched localpca time: {t_fast:.3f}s")
    finally:
        localpca_mod.genpca = old_genpca

    # tolerances similar to what you used before
    if den_ref.dtype == np.float64:
        rtol = atol = 1e-10
    else:
        rtol = atol = 2e-5

    compare_arrays(den_fast, den_ref, rtol=rtol, atol=atol)

    if t_fast > 0:
        print(f"\nspeedup: {t_ref / t_fast:.2f}x")
    else:
        print("\nspeedup: (t_fast=0?)")


if __name__ == "__main__":
    main()
