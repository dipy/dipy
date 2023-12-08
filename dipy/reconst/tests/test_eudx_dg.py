import warnings

import numpy as np
import numpy.testing as npt

from dipy.direction.peaks import default_sphere, peaks_from_model
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_EuDXDirectionGetter(rng):
    class SillyModel:
        def fit(self, data, mask=None):
            return SillyFit(self)

    class SillyFit:

        def __init__(self, model):
            self.model = model

        def odf(self, sphere):
            odf = np.zeros(sphere.theta.shape)
            r = rng.integers(0, len(odf))
            odf[r] = 1
            return odf

    def get_direction(dg, point, direction):
        newdir = direction.copy()
        state = dg.get_direction(point, newdir)
        return state, np.array(newdir)

    data = rng.random((3, 4, 5, 2))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        peaks = peaks_from_model(SillyModel(), data, default_sphere,
                                 relative_peak_threshold=.5,
                                 min_separation_angle=25)
    peaks._initialize()

    up = np.zeros(3)
    up[2] = 1.
    down = -up

    for i in range(3 - 1):
        for j in range(4 - 1):
            for k in range(5 - 1):
                point = np.array([i, j, k], dtype=float)

                # Test that the angle threshold rejects points
                peaks.ang_thr = 0.
                state, nd = get_direction(peaks, point, up)
                npt.assert_equal(state, 1)

                # Here we leverage the fact that we know Hemispheres project
                # all their vertices into the z >= 0 half of the sphere.
                peaks.ang_thr = 90.
                state, nd = get_direction(peaks, point, up)
                npt.assert_equal(state, 0)
                expected_dir = peaks.peak_dirs[i, j, k, 0]
                npt.assert_array_almost_equal(nd, expected_dir)
                state, nd = get_direction(peaks, point, down)
                npt.assert_array_almost_equal(nd, -expected_dir)

                # Check that we can get directions at non-integer points
                point += rng.random(3)
                state, nd = get_direction(peaks, point, up)
                npt.assert_equal(state, 0)

                # Check that points are rounded to get initial direction
                point -= .5
                initial_dir = peaks.initial_direction(point)
                # It should be a (1, 3) array
                npt.assert_array_almost_equal(initial_dir, [expected_dir])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        peaks1 = peaks_from_model(SillyModel(), data, default_sphere,
                                  relative_peak_threshold=.5,
                                  min_separation_angle=25,
                                  npeaks=1)
    peaks1._initialize()
    point = np.array([1, 1, 1], dtype=float)

    # it should have one direction
    npt.assert_array_almost_equal(len(peaks1.initial_direction(point)), 1)
    npt.assert_array_almost_equal(len(peaks.initial_direction(point)), 1)
