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

        median_otsu_flow(data_path, out_dir, save_masked, median_radius,
                         numpass, autocrop, vol_idx, dilate)

        masked, mask = median_otsu(volume, median_radius,
                                   numpass, autocrop,
                                   vol_idx, dilate)

        fname, _ = splitext(splitext(basename(data_path))[0])

        mask_fname = fname + '_mask.nii.gz'
        result_mask_data = nib.load(join(out_dir, mask_fname)).get_data()
        npt.assert_array_equal(result_mask_data, mask)

        masked_fname = fname + '_bet.nii.gz'
        result_masked_data = nib.load(join(out_dir, masked_fname)).get_data()
        npt.assert_array_equal(result_masked_data, masked)
