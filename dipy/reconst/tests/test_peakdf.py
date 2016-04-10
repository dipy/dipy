import numpy as np
import numpy.testing as npt

from dipy.direction.peaks import default_sphere, peaks_from_model


def test_PeaksAndMetricsDirectionGetter():

    class SillyModel(object):
        def fit(self, data, mask=None):
            return SillyFit(self)

    class SillyFit(object):

        def __init__(self, model):
            self.model = model

        def odf(self, sphere):
            odf = np.zeros(sphere.theta.shape)
            r = np.random.randint(0, len(odf))
            odf[r] = 1
            return odf

    def get_direction(dg, point, dir):
        newdir = dir.copy()
        state = dg.get_direction(point, newdir)
        return (state, np.array(newdir))

    data = np.random.random((3, 4, 5, 2))
    peaks = peaks_from_model(SillyModel(), data, default_sphere,
                             relative_peak_threshold=.5,
                             min_separation_angle=25)
    peaks._initialize()

    up = np.zeros(3)
    up[2] = 1.
    down = -up

    for i in range(3-1):
        for j in range(4-1):
            for k in range(5-1):
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
                point += np.random.random(3)
                state, nd = get_direction(peaks, point, up)
                npt.assert_equal(state, 0)

                # Check that points are rounded to get initial direction
                point -= .5
                id = peaks.initial_direction(point)
                # id should be a (1, 3) array
                npt.assert_array_almost_equal(id, [expected_dir])


if __name__ == "__main__":
    npt.run_module_suite()
