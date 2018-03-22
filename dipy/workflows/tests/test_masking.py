import numpy as np
import numpy.testing as nt
from nose.tools import assert_equal, assert_false

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.mask import MaskFlow


def test_mask():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        vol_img = nib.load(data_path)
        volume = vol_img.get_data()

        mask_flow = MaskFlow()

        mask_flow.run(data_path, 10, out_dir=out_dir, ub=9)
        assert_false(mask_flow.last_generated_outputs)

        mask_flow.run(data_path, 10, out_dir=out_dir)
        mask_path = mask_flow.last_generated_outputs['out_mask']
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_data()
        assert_equal(mask_data.shape, volume.shape)
        nt.assert_array_almost_equal(mask_img.affine, vol_img.affine)
        assert_equal(mask_data.dtype, np.uint8)


if __name__ == '__main__':
    test_mask()
