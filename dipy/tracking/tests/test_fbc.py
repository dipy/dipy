from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.tracking.fbcmeasures import FBCMeasures


from dipy.core.sphere import Sphere

import numpy as np
import numpy.testing as npt


def test_fbc():
    """Test the FBC measures on a set of fibers"""

    # Generate two fibers of 10 points
    streamlines = []
    for i in range(2):
        fiber = np.zeros((10, 3))
        for j in range(10):
            fiber[j, 0] = j
            fiber[j, 1] = i*0.2
            fiber[j, 2] = 0
            streamlines.append(fiber)

    # Create lookup table.
    # A fixed set of orientations is used to guarantee deterministic results
    D33 = 1.0
    D44 = 0.04
    t = 1
    sphere = Sphere(xyz=np.array([[0.82819078, 0.51050355, 0.23127074],
                                  [-0.10761926, -0.95554309, 0.27450957],
                                  [0.4101745, -0.07154038, 0.90919682],
                                  [-0.75573448, 0.64854889, 0.09082809],
                                  [-0.56874549, 0.01377562, 0.8223982]]))
    k = EnhancementKernel(D33, D44, t, orientations=sphere,
                          force_recompute=True)

    # run FBC
    fbc = FBCMeasures(streamlines, k, verbose=True)

    # get FBC values
    fbc_sl_orig, clrs_orig, rfbc_orig = \
        fbc.get_points_rfbc_thresholded(0, emphasis=0.01)

    # check mean RFBC against tested value
    npt.assert_almost_equal(np.mean(rfbc_orig), 1.0500466494329224)

if __name__ == '__main__':
    npt.run_module_suite()
