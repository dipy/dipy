#!/usr/bin/env python
"""Quick launcher for the Skyline-inception viewer with HCP data.

Loads the HCP CSD fit and an FA map and launches the integrated
Skyline viewer (dipy.viz.skyline.app) with the SH glyph slicer
using the skyline-inception UI (UIWindow + imgui sidebar).
"""

import os
import sys
BUILD = os.path.join(os.path.dirname(__file__), "build", "cp312")
if os.path.isdir(BUILD):
    sys.path.insert(0, BUILD)

import numpy as np
from pathlib import Path

HCP_DIR = Path.home() / "hermite" / "hcp_data"
CSD_FIT = HCP_DIR / "hcp_csd_fit.npz"

print("Loading CSD fit …")
csd = np.load(CSD_FIT, allow_pickle=True)
sh_coeffs   = csd["sh_coeffs"]
mask        = csd["mask"]
fa_data     = csd["fa"]
voxel_sizes = csd["voxel_sizes"]
sh_order    = int(csd["sh_order"])

sh_crop = sh_coeffs
fa_crop = fa_data
mask_crop = mask

affine = np.diag(list(voxel_sizes) + [1.0])
print(f"  SH coeffs shape : {sh_crop.shape}")
print(f"  FA shape         : {fa_crop.shape}")
print(f"  Mask voxels      : {mask_crop.sum()}")
print(f"  Voxel sizes      : {voxel_sizes}")
print(f"  SH order         : {sh_order}")
print(f"  Affine:\n{affine}")

print("\nLaunching Skyline (skyline-inception UI) …")

from dipy.viz.skyline.app import skyline

skyline(
    images=[(fa_crop, affine, "FA (HCP)")],
    sh_coeffs=[(sh_crop, affine, "SH Glyphs (HCP)")],
)
