#!/usr/bin/env python
import numpy as np
import numpy.testing as npt

from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.core.gradients import GradientTable
from dipy.data import read_stanford_labels
from dipy.io.image import load_nifti_data


def num_grad(gtab):
    return (~gtab.b0s_mask).sum()


def bench_csdeconv(center=(50, 40, 40), width=12):
    img, gtab, labels_img = read_stanford_labels()
    data = load_nifti_data(img)

    labels = load_nifti_data(labels_img)
    shape = labels.shape
    mask = np.in1d(labels, [1, 2])
    mask.shape = shape

    a, b, c = center
    hw = width // 2
    idx = (slice(a - hw, a + hw), slice(b - hw, b + hw), slice(c - hw, c + hw))

    data_small = data[idx].copy()
    mask_small = mask[idx].copy()
    voxels = mask_small.sum()

    cmd = "model.fit(data_small, mask_small)"
    print("== Benchmarking CSD fit on %d voxels ==" % voxels)
    msg = "SH order - %d, gradient directons - %d :: %g sec"

    # Basic case
    sh_order = 8
    ConstrainedSphericalDeconvModel(gtab, None, sh_order=sh_order)
    time = npt.measure(cmd)
    print(msg % (sh_order, num_grad(gtab), time))

    # Smaller data set
    # data_small = data_small[..., :75].copy()
    gtab = GradientTable(gtab.gradients[:75])
    ConstrainedSphericalDeconvModel(gtab, None, sh_order=sh_order)
    time = npt.measure(cmd)
    print(msg % (sh_order, num_grad(gtab), time))

    # Super resolution
    sh_order = 12
    ConstrainedSphericalDeconvModel(gtab, None, sh_order=sh_order)
    time = npt.measure(cmd)
    print(msg % (sh_order, num_grad(gtab), time))

if __name__ == "__main__":
    bench_csdeconv()
