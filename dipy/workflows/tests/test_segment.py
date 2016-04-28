from os.path import join, basename, splitext
import nibabel as nib
import numpy.testing as npt

from dipy.workflows.segment import median_otsu_flow
from dipy.segment.mask import median_otsu
from dipy.data import get_data


def test_median_otsu_flow():
    with nib.tmpdirs.InTemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        volume = nib.load(data_path).get_data()
        save_masked = True
        median_radius = 3
        numpass = 3
        autocrop = False
        vol_idx = [0]
        dilate = 0

        mask_name = 'mask.nii.gz'
        masked_name = 'masked.nii.gz'
        median_otsu_flow(data_path, out_dir=out_dir, save_masked=save_masked,
                         median_radius=median_radius, numpass=numpass,
                         autocrop=autocrop, vol_idx=vol_idx, dilate=dilate,
                         mask=mask_name, masked=masked_name)

        masked, mask = median_otsu(volume, median_radius,
                                   numpass, autocrop,
                                   vol_idx, dilate)

        result_mask_data = nib.load(join(out_dir, mask_name)).get_data()
        npt.assert_array_equal(result_mask_data, mask)

        result_masked_data = nib.load(join(out_dir, masked_name)).get_data()
        npt.assert_array_equal(result_masked_data, masked)
