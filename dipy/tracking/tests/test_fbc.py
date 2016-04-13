from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.tracking.fbcmeasures import FBCMeasures

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors
from dipy.viz import window, actor

import numpy as np
import numpy.testing as npt

def test_fbc():
    """Test the FBC measures on a set of fibers"""

    # Generate two fibers of 10 points
    streamlines = []
    for i in range(2):
        fiber = np.zeros((10,3))
        for j in range(10):
            fiber[j,0] = j
            fiber[j,1] = i*0.2
            fiber[j,2] = 0
            streamlines.append(fiber)

    # Create lookup table. Set seed for deterministic results.
    D33 = 1.0
    D44 = 0.04
    t = 1
    np.random.seed(1)
    num_orientations = 5
    k = EnhancementKernel(D33, D44, t, orientations=num_orientations, force_recompute=True)

    # run FBC
    fbc = FBCMeasures(streamlines, k, verbose=True)

    # get FBC values
    fbc_sl_orig, clrs_orig, rfbc_orig = \
        fbc.get_points_rfbc_thresholded(0, emphasis=0.01)

    # check RFBC against tested value
    npt.assert_almost_equal(np.mean(rfbc_orig), 1.0549502181194517)


if __name__ == '__main__':
   npt.run_module_suite()
