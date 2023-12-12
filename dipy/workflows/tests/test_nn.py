from os.path import join as pjoin
from tempfile import TemporaryDirectory

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.io.image import load_nifti_data, save_nifti
from dipy.nn.evac import EVACPlus
from dipy.utils.optpkg import optional_package
from dipy.workflows.nn import EVACPlusFlow


tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_evac_plus_flow():
    with TemporaryDirectory() as out_dir:
        file_path = get_fnames('evac_test_data')

        volume = np.load(file_path)['input'][0]
        temp_affine = np.eye(4)
        temp_path = pjoin(out_dir, 'temp.nii.gz')
        save_nifti(temp_path, volume, temp_affine)
        save_masked = True

        evac_flow = EVACPlusFlow()
        evac_flow.run(temp_path, out_dir=out_dir, save_masked=save_masked)

        mask_name = evac_flow.last_generated_outputs['out_mask']
        masked_name = evac_flow.last_generated_outputs['out_masked']

        evac = EVACPlus()
        mask = evac.predict(volume, temp_affine)
        masked = volume * mask

        result_mask_data = load_nifti_data(pjoin(out_dir, mask_name))
        npt.assert_array_equal(result_mask_data.astype(np.uint8), mask)

        result_masked_data = load_nifti_data(pjoin(out_dir, masked_name))

        npt.assert_array_equal(result_masked_data, masked)
