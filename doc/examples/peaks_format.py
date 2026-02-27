"""
==================================
Understanding the PAM5 File Format
==================================

DIPY stores peak directions and associated metrics from diffusion MRI
reconstruction in PAM5 files (``.pam5``), which are HDF5 under the hood.

"""

import h5py
import numpy as np

from dipy.core.gradients import gradient_table
from dipy.data import default_sphere, get_fnames
from dipy.direction import peaks_from_model
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti
from dipy.io.peaks import load_pam, save_pam
from dipy.reconst.shm import CsaOdfModel

###############################################################################
# PAM5 File Structure
# ===================
#
# A ``.pam5`` file is an HDF5 file with the following layout:
#
# .. code-block:: text
#
#     file.pam5  (HDF5)
#     |-- @version = "0.0.1"              <- file-level attribute
#     +-- pam/                            <- HDF5 group
#         |-- peak_dirs       (X,Y,Z,N,3)   float64  REQUIRED
#         |-- peak_values     (X,Y,Z,N)     float64  REQUIRED
#         |-- peak_indices    (X,Y,Z,N)     int32    REQUIRED
#         |-- affine          (4,4)          float64
#         |-- sphere_vertices (M,3)         float64
#         |-- shm_coeff       (X,Y,Z,K)     float64
#         |-- B               (K,M)         float64
#         |-- gfa             (X,Y,Z)       float64
#         |-- qa              (X,Y,Z,N)     float64
#         |-- odf             (X,Y,Z,M)     float64
#         |-- total_weight    (1,)           float64
#         +-- ang_thr         (1,)           float64
#
# X, Y, Z are spatial dimensions, N is the number of peaks per voxel
# (default 5), M is the number of sphere vertices, and K is the number
# of spherical harmonics coefficients.
#
# Only ``peak_dirs``, ``peak_values``, and ``peak_indices`` are
# required. Everything else is optional and will be ``None`` when not
# present.

###############################################################################
# Generating and Saving Example Peaks
# ===================================
#
# To demonstrate the PAM5 format, we first need some peaks. We fit a Constant
# Solid Angle (CSA) ODF model to a small patch of data and extract the peaks.
# Then, we can save the resulting ``PeaksAndMetrics`` object to a PAM5 file
# using ``save_pam``, and load it back using ``load_pam``.

hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames(
    name="stanford_hardi"
)
data, affine = load_nifti(hardi_fname)
bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
gtab = gradient_table(bvals, bvecs=bvecs)

data_small = data[20:50, 55:85, 38:39]

csamodel = CsaOdfModel(gtab, sh_order_max=4)
pam = peaks_from_model(
    model=csamodel,
    data=data_small,
    sphere=default_sphere,
    relative_peak_threshold=0.5,
    min_separation_angle=25,
    return_sh=True,
    return_odf=False,
)

print(f"peak_dirs shape: {pam.peak_dirs.shape}")

save_pam("csa_peaks.pam5", pam, affine=affine)
pam_loaded = load_pam("csa_peaks.pam5")

###############################################################################
# Creating a PAM Manually
# =======================
#
# A ``PeaksAndMetrics`` object can also be constructed directly, e.g.
# to convert peak data from FSL or MRtrix into DIPY's format.

pam_manual = PeaksAndMetrics()
pam_manual.peak_dirs = np.random.randn(10, 10, 10, 5, 3)
pam_manual.peak_values = np.zeros((10, 10, 10, 5))
pam_manual.peak_indices = np.full(
    (10, 10, 10, 5), -1, dtype=np.int32
)
pam_manual.affine = np.eye(4)
pam_manual.sphere = default_sphere
pam_manual.shm_coeff = None
pam_manual.B = None
pam_manual.gfa = np.zeros((10, 10, 10))
pam_manual.qa = np.zeros((10, 10, 10, 5))
pam_manual.odf = None
pam_manual.total_weight = 0.5
pam_manual.ang_thr = 60.0

save_pam("manual_peaks.pam5", pam_manual)

###############################################################################
# Inspecting the HDF5 Structure
# ==============================
#
# Since PAM5 is standard HDF5, any HDF5 tool can read it. Here we
# inspect the file directly with ``h5py`` to confirm the layout.

with h5py.File("csa_peaks.pam5", "r") as f:
    print(f"Version: {f.attrs['version']}")
    for name, ds in f["pam"].items():
        print(f"  {name}: shape={ds.shape}, dtype={ds.dtype}")

###############################################################################
# NIfTI Interoperability
# ======================
#
# PAM5 files can be converted to and from NIfTI using
# ``pam_to_niftis`` and ``niftis_to_pam``. This is useful when
# exchanging peak data with tools that only read NIfTI, such as
# FSL or MRtrix.
#
# .. code-block:: python
#
#     from dipy.io.peaks import pam_to_niftis, niftis_to_pam
#
#     pam_to_niftis(pam)
#
#     pam_from_nifti = niftis_to_pam(
#         affine=affine,
#         peak_dirs=peak_dirs_array,
#         peak_values=peak_values_array,
#         peak_indices=peak_indices_array,
#     )
#
