import logging
import warnings
from os.path import join as pjoin

import numpy as np
import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory

from dipy.core.gradients import generate_bvecs
from dipy.data import get_fnames
from dipy.io.peaks import load_peaks
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti, save_nifti, load_nifti_data
from dipy.workflows.reconst import ReconstRUMBAFlow
from dipy.reconst.shm import descoteaux07_legacy_msg, sph_harm_ind_list
logging.getLogger().setLevel(logging.INFO)


def test_reconst_rumba():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        reconst_flow_core(ReconstRUMBAFlow)


def reconst_flow_core(flow):
    with TemporaryDirectory() as out_dir:
        data_path, bval_path, bvec_path = get_fnames('small_64D')
        volume, affine = load_nifti(data_path)
        mask = np.ones_like(volume[:, :, :, 0])
        mask_path = pjoin(out_dir, 'tmp_mask.nii.gz')
        save_nifti(mask_path, mask.astype(np.uint8), affine)

        reconst_flow = flow()
        for sh_order in [8, ]:

            reconst_flow.run(
                data_path, bval_path, bvec_path, mask_path, sh_order=sh_order,
                out_dir=out_dir, extract_pam_values=True)
