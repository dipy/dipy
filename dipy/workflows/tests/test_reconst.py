import os
from os.path import join
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt

from dipy.data import default_sphere, get_fnames
from dipy.direction.peaks import PeaksAndMetrics
from dipy.io.image import load_nifti, save_nifti
from dipy.io.peaks import save_pam
from dipy.testing.decorators import set_random_number_generator
from dipy.workflows.reconst import CorrectBvecsFlow


@set_random_number_generator()
def test_correct_bvecs_flow(rng):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames(name="small_25")
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0], dtype=np.uint8)
        mask_path = join(out_dir, "tmp_mask.nii.gz")
        save_nifti(mask_path, mask, affine)
        peaks = np.zeros(mask.shape + (3,), dtype=float)
        peaks[1, 1, :, :] = np.array([[0, 0, 1], [0, 0, 0]], dtype=float)
        peaks_path = join(out_dir, "peaks.nii.gz")
        save_nifti(peaks_path, peaks, affine)
        fa = rng.random(mask.shape)
        fa[1, 1, :] = [1, 0]
        fa_path = join(out_dir, "fa.nii.gz")
        save_nifti(fa_path, fa, affine)

        correct_flow = CorrectBvecsFlow()

        args = [bvec_path, fa_path, peaks_path, mask_path]
        kwargs = {"out_dir": out_dir}

        correct_flow.run(*args, **kwargs)

        assert os.path.exists(join(out_dir, "corrected_bvecs.txt"))
        bvec_corrected = np.loadtxt(join(out_dir, "corrected_bvecs.txt"))
        bvec = np.loadtxt(bvec_path)
        npt.assert_array_equal(bvec_corrected, bvec)

        correct_flow._force_overwrite = True

        peaks[1, 1, :, :] = np.array([[0, 1, 0], [0, 0, 0]], dtype=float)
        pam_path = join(out_dir, "peaks.pam5")
        pam = PeaksAndMetrics()
        pam.affine = affine
        pam.peak_dirs = peaks
        pam.peak_values = np.zeros(peaks.shape[:-1])
        pam.peak_indices = np.zeros(peaks.shape[:-1])
        pam.sphere = default_sphere
        save_pam(pam_path, pam, affine=affine)
        args = [bvec_path, fa_path, pam_path, mask_path]
        kwargs = {"out_dir": out_dir, "out_bvecs": "corrected_bvecs_2.txt"}

        correct_flow.run(*args, **kwargs)
        assert os.path.exists(join(out_dir, "corrected_bvecs_2.txt"))
        bvec_corrected = np.loadtxt(join(out_dir, "corrected_bvecs_2.txt"))
        npt.assert_array_equal(bvec_corrected, bvec)
