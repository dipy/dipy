import numpy as np
import nibabel as nib
import numpy.testing as npt

from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.segment.rois_stats import seg_stats
from dipy.segment.energy_mrf import ising
from dipy.denoise.denspeed import add_padding_reflection
# from dipy.denoise.denspeed import remove_padding
from dipy.segment.icm_map import icm
import matplotlib.pyplot as plt


def test_ising():

    l = 1
    vox = 2
    beta = 20

    npt.assert_equal(ising(l, vox, beta), beta)

    l = 1
    vox = 1
    beta = 20

    npt.assert_equal(ising(l, vox, beta), - beta)


if __name__ == '__main__':

    test_ising()
    # npt.run_module_suite()