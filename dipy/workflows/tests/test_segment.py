import numpy.testing as npt
from os.path import join

import nibabel as nib
from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.segment.mask import median_otsu
from dipy.workflows.segment import MedianOtsuFlow


def test_median_otsu_flow():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        volume = nib.load(data_path).get_data()
        save_masked = True
        median_radius = 3
        numpass = 3
        autocrop = False
        vol_idx = [0]
        dilate = 0

        mo_flow = MedianOtsuFlow()
        mo_flow.run(data_path, out_dir=out_dir, save_masked=save_masked,
                             median_radius=median_radius, numpass=numpass,
                             autocrop=autocrop, vol_idx=vol_idx, dilate=dilate)

        mask_name = mo_flow.last_generated_outputs['out_mask']
        masked_name = mo_flow.last_generated_outputs['out_masked']

        masked, mask = median_otsu(volume, median_radius,
                                   numpass, autocrop,
                                   vol_idx, dilate)

        result_mask_data = nib.load(join(out_dir, mask_name)).get_data()
        npt.assert_array_equal(result_mask_data, mask)

        result_masked_data = nib.load(join(out_dir, masked_name)).get_data()
        npt.assert_array_equal(result_masked_data, masked)

if __name__ == '__main__':
    test_median_otsu_flow()
