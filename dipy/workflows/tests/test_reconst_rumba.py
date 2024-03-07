import logging
from os.path import join as pjoin
from tempfile import TemporaryDirectory
import warnings

import numpy as np

from dipy.data import get_fnames
from dipy.io.image import load_nifti, save_nifti
from dipy.workflows.reconst import ReconstRUMBAFlow
from dipy.reconst.shm import descoteaux07_legacy_msg

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
        for sh_order_max in [8, ]:
            reconst_flow.run(
                data_path, bval_path, bvec_path, mask_path,
                sh_order_max=sh_order_max,
                out_dir=out_dir, extract_pam_values=True)
