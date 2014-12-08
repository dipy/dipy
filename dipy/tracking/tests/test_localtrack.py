import numpy as np
import numpy.testing as npt

from dipy.tracking.local.tissue_classifier import ThresholdTissueClassifier
from dipy.data import default_sphere
from dipy.direction import peaks_from_model

def test_ThresholdTissueClassifier():
    a = np.random.random((3, 5, 7))
    mid = np.sort(a.ravel())[(3 * 5 * 7) // 2]

    ttc = ThresholdTissueClassifier(a, mid)
    for i in range(3):
        for j in range(5):
            for k in range(7):
                tissue = ttc.check_point(np.array([i, j, k], dtype=float))
                if a[i, j, k] > mid:
                    npt.assert_equal(tissue, 1)
                else:
                    npt.assert_equal(tissue, 2)



