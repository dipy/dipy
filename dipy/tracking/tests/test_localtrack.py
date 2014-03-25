import numpy as np
import numpy.testing as npt

import dipy.tracking.localtrack as lt
from dipy.tracking.localtrack import _testCheckPoint, ThresholdTissueClassifier
from dipy.reconst.peaks import default_sphere, peaks_from_model

def test_ThresholdTissueClassifier():
    a = np.random.random((3, 5, 7))
    mid = np.sort(a.ravel())[(3 * 5 * 7) // 2]

    ttc = ThresholdTissueClassifier(a, mid)
    for i in range(3):
        for j in range(5):
            for k in range(7):
                print (a[i, j, k] >= mid)
                print _testCheckPoint(ttc, np.array([i, j, k], dtype=float))

def testErrorInPyDirectionGetter():

    class MyError(Exception):
        pass

    class BadDirectionGetter(lt.PythonDirectionGetter):

        def initial_direction(self, point):
            return np.eye(3)

        def _get_direction(self, point, prev_dir):
            raise MyError()

    bdg = BadDirectionGetter()
    ones = np.ones(3)
    npt.assert_raises(MyError, lt._testGetDirection, bdg, ones, ones)

