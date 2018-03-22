import logging
import numpy as np
from nose.tools import assert_equal
from os.path import join
import numpy.testing as npt

import nibabel as nib
from dipy.io.peaks import load_peaks
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.reconst import ReconstCSDFlow, ReconstCSAFlow
logging.getLogger().setLevel(logging.INFO)


def test_reconst_csa():
    reconst_flow_core(ReconstCSAFlow)


def test_reconst_csd():
    reconst_flow_core(ReconstCSDFlow)


def reconst_flow_core(flow):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_data('small_64D')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()
        mask = np.ones_like(volume[:, :, :, 0])
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), vol_img.affine)
        mask_path = join(out_dir, 'tmp_mask.nii.gz')
        nib.save(mask_img, mask_path)

        reconst_flow = flow()

        reconst_flow.run(data_path, bval_path, bvec_path, mask_path,
                         out_dir=out_dir, extract_pam_values=True)

        gfa_path = reconst_flow.last_generated_outputs['out_gfa']
        gfa_data = nib.load(gfa_path).get_data()
        assert_equal(gfa_data.shape, volume.shape[:-1])

        peaks_dir_path = reconst_flow.last_generated_outputs['out_peaks_dir']
        peaks_dir_data = nib.load(peaks_dir_path).get_data()
        assert_equal(peaks_dir_data.shape[-1], 15)
        assert_equal(peaks_dir_data.shape[:-1], volume.shape[:-1])

        peaks_idx_path = \
            reconst_flow.last_generated_outputs['out_peaks_indices']
        peaks_idx_data = nib.load(peaks_idx_path).get_data()
        assert_equal(peaks_idx_data.shape[-1], 5)
        assert_equal(peaks_idx_data.shape[:-1], volume.shape[:-1])

        peaks_vals_path = \
            reconst_flow.last_generated_outputs['out_peaks_values']
        peaks_vals_data = nib.load(peaks_vals_path).get_data()
        assert_equal(peaks_vals_data.shape[-1], 5)
        assert_equal(peaks_vals_data.shape[:-1], volume.shape[:-1])

        shm_path = reconst_flow.last_generated_outputs['out_shm']
        shm_data = nib.load(shm_path).get_data()
        assert_equal(shm_data.shape[-1], 45)
        assert_equal(shm_data.shape[:-1], volume.shape[:-1])

        pam = load_peaks(reconst_flow.last_generated_outputs['out_pam'])
        npt.assert_allclose(pam.peak_dirs.reshape(peaks_dir_data.shape),
                            peaks_dir_data)
        npt.assert_allclose(pam.peak_values, peaks_vals_data)
        npt.assert_allclose(pam.peak_indices, peaks_idx_data)
        npt.assert_allclose(pam.shm_coeff, shm_data)
        npt.assert_allclose(pam.gfa, gfa_data)

        if flow.get_short_name() == 'csd':

            reconst_flow = flow()
            reconst_flow._force_overwrite = True
            reconst_flow.run(data_path, bval_path, bvec_path, mask_path,
                             out_dir=out_dir, frf=[15, 5, 5])
            reconst_flow = flow()
            reconst_flow._force_overwrite = True
            reconst_flow.run(data_path, bval_path, bvec_path, mask_path,
                             out_dir=out_dir, frf='15, 5, 5')
            reconst_flow = flow()
            reconst_flow._force_overwrite = True
            reconst_flow.run(data_path, bval_path, bvec_path, mask_path,
                             out_dir=out_dir, frf=None)
            reconst_flow2 = flow()
            reconst_flow2._force_overwrite = True
            reconst_flow2.run(data_path, bval_path, bvec_path, mask_path,
                              out_dir=out_dir, frf=None,
                              roi_center=[10, 10, 10])


if __name__ == '__main__':
    test_reconst_csa()
    test_reconst_csd()
