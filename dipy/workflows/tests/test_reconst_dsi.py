from os.path import join as pjoin
from tempfile import TemporaryDirectory
import warnings

import numpy as np
import numpy.testing as npt

from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti, load_nifti_data
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.workflows.reconst import ReconstDsiFlow


def test_reconst_dsi():
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_64D')
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0])
        mask_path = pjoin(out_dir, 'tmp_mask.nii.gz')
        save_nifti(mask_path, mask.astype(np.uint8), affine)

        dsi_flow = ReconstDsiFlow()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=descoteaux07_legacy_msg,
                                    category=PendingDeprecationWarning)
            dsi_flow.run(data_path, bval_path, bvec_path, mask_path,
                         out_dir=out_dir, extract_pam_values=True)

        peaks_dir_path = \
            dsi_flow.last_generated_outputs['out_peaks_dir']
        peaks_dir_data = load_nifti_data(peaks_dir_path)
        npt.assert_equal(peaks_dir_data.shape[-1], 15)
        npt.assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

        peaks_idx_path = \
            dsi_flow.last_generated_outputs['out_peaks_indices']
        peaks_idx_data = load_nifti_data(peaks_idx_path)
        npt.assert_equal(peaks_idx_data.shape[-1], 5)
        npt.assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

        peaks_vals_path = \
            dsi_flow.last_generated_outputs['out_peaks_values']
        peaks_vals_data = load_nifti_data(peaks_vals_path)
        npt.assert_equal(peaks_vals_data.shape[-1], 5)
        npt.assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])
