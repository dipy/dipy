from tempfile import TemporaryDirectory

import numpy.testing as npt
from dipy.testing import assert_false

from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.workflows.mask import MaskFlow


def test_mask():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_fnames('small_25')
        volume, affine = load_nifti(data_path)

        mask_flow = MaskFlow()

        mask_flow.run(data_path, 10, out_dir=out_dir, ub=9)
        assert_false(mask_flow.last_generated_outputs)

        mask_flow.run(data_path, 10, out_dir=out_dir)
        mask_path = mask_flow.last_generated_outputs['out_mask']
        mask_data, mask_affine = load_nifti(mask_path)
        npt.assert_equal(mask_data.shape, volume.shape)
        npt.assert_array_almost_equal(mask_affine, affine)
