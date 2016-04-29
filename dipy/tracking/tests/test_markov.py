from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.tracking import utils
from dipy.reconst.interpolate import NearestNeighborInterpolator
from dipy.tracking.markov import (BoundaryStepper, _closest_peak,
                                  FixedSizeStepper, MarkovIntegrator,
                                  markov_streamline, OutsideImage,
                                  ClosestDirectionTracker,
                                  ProbabilisticOdfWeightedTracker)
from dipy.core.sphere import HemiSphere, unit_octahedron
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_, assert_raises)


def test_BoundaryStepper():
    os = 1
    bi = BoundaryStepper(overstep=os)
    loc = np.array([.5, .5, .5])
    step = np.array([1, 1, 1]) / np.sqrt(3)
    assert_array_almost_equal(bi(loc, step), os * step + [1, 1, 1])
    assert_array_almost_equal(bi(loc, -step), -os * step)

    os = 2
    bi = BoundaryStepper((2, 3, 4), overstep=2)
    assert_array_almost_equal(bi(loc, step), os * step + [2, 2, 2])
    assert_array_almost_equal(bi(loc, -step), -os * step)

    loc = np.array([7.5, 7.5, 7.5])
    assert_array_almost_equal(bi(loc, step), os * step + [8, 8, 8])
    assert_array_almost_equal(bi(loc, -step), [6, 6, 6] - os * step)


def test_FixedSizeStepper():
    fsi = FixedSizeStepper(step_size=2.)
    loc = np.array([2, 3, 12])
    step = np.array([3, 2, 4]) / np.sqrt(3)
    assert_array_almost_equal(fsi(loc, step), loc + 2. * step)
    assert_array_almost_equal(fsi(loc, -step), loc - 2. * step)


def test_markov_streamline():

    east = np.array([1, 0, 0])

    class MoveEastWest(object):
        def get_direction(self, location, prev_step):
            if np.any(location < 0):
                raise OutsideImage
            elif np.any(location > 10.):
                return None
            if np.dot(prev_step, east) >= 0:
                return east
            else:
                return -east

    seed = np.array([5.2, 0, 0])
    first_step = east
    dir_getter = MoveEastWest()
    stepper = FixedSizeStepper(.5)

    # The streamline terminates when it goes past (10, 0, 0). (10.2, 0, 0)
    # should be the last point in the streamline
    streamline = markov_streamline(dir_getter.get_direction, stepper,
                                   seed, first_step, 100)
    expected = np.zeros((11, 3))
    expected[:, 0] = np.linspace(5.2, 10.2, 11)
    assert_array_almost_equal(streamline, expected)

    # OutsideImage gets raised when the streamline points become negative
    # the streamline should end, and the negative points should not be part
    # of the streamline
    first_step = -east
    streamline = markov_streamline(dir_getter.get_direction, stepper,
                                   seed, first_step, 100)
    expected = np.zeros((11, 3))
    expected[:, 0] = np.linspace(5.2, 0.2, 11)
    assert_array_almost_equal(streamline, expected)


def test_MarkovIntegrator():

    class KeepGoing(MarkovIntegrator):
        def _next_step(self, location, prev_step):
            if prev_step is None:
                return np.array([[1., 0, 0],
                                 [0, 1., 0],
                                 [0, 0., 1]])
            if not self._mask[location]:
                return None
            else:
                return prev_step

    data = np.ones((10, 10, 10, 65))
    data_interp = NearestNeighborInterpolator(data, (1, 1, 1))

    seeds = [np.array([5.2, 5.2, 5.2])]
    stepper = FixedSizeStepper(.5)
    mask = np.ones((10, 10, 10), 'bool')
    gen = KeepGoing(model=None, interpolator=data_interp, mask=mask,
                    take_step=stepper, angle_limit=0., seeds=seeds)
    streamlines = list(gen)
    assert_equal(len(streamlines), 3)

    expected = np.zeros((20, 3))
    for i in range(3):
        expected[:] = 5.2
        expected[:, i] = np.arange(.2, 10, .5)
        assert_array_almost_equal(streamlines[i], expected)

    # Track only the first (largest) peak for each seed
    gen = KeepGoing(model=None, interpolator=data_interp, mask=mask,
                    take_step=stepper, angle_limit=0., seeds=seeds,
                    max_cross=1)
    streamlines = list(gen)
    assert_equal(len(streamlines), 1)

    expected = np.zeros((20, 3))
    expected[:] = 5.2
    expected[:, 0] = np.arange(.2, 10, .5)
    assert_array_almost_equal(streamlines[0], expected)

    mask = np.ones((20, 20, 20), 'bool')
    gen = KeepGoing(model=None, interpolator=data_interp, mask=mask,
                    take_step=stepper, angle_limit=0., seeds=seeds,
                    max_cross=1, mask_voxel_size=(.5, .5, .5))
    streamlines = list(gen)
    assert_equal(len(streamlines), 1)
    assert_array_almost_equal(streamlines[0], expected)

    # Test tracking with affine
    affine = np.eye(4)
    affine[:3, :] = np.random.random((3, 4)) - .5

    seeds = [np.dot(affine[:3, :3], seeds[0] - .5) + affine[:3, 3]]
    sl_affine = KeepGoing(model=None, interpolator=data_interp, mask=mask,
                    take_step=stepper, angle_limit=0., seeds=seeds,
                    max_cross=1, mask_voxel_size=(.5, .5, .5), affine=affine)

    default = np.eye(4)
    default[:3, 3] = .5
    sl_default = list(utils.move_streamlines(sl_affine, default, affine))

    assert_equal(len(sl_default), 1)
    assert_array_almost_equal(sl_default[0], expected)


def test_closest_peak():
    peak_values = np.array([1, .9, .8, .7, .6, .2, .1])
    peak_points = np.array([[1., 0., 0.],
                            [0., .9, .1],
                            [0., 1., 0.],
                            [.9, .1, 0.],
                            [0., 0., 1.],
                            [1., 1., 0.],
                            [0., 1., 1.]])
    norms = np.sqrt((peak_points * peak_points).sum(-1))
    peak_points = peak_points / norms[:, None]

    prev = np.array([1, -.9, 0])
    prev = prev / np.sqrt(np.dot(prev, prev))
    cp = _closest_peak(peak_points, prev, 0.)
    assert_array_equal(cp, peak_points[0])
    cp = _closest_peak(peak_points, -prev, 0.)
    assert_array_equal(cp, -peak_points[0])


def test_ClosestDirectionTracker():
    class MyModel(object):
        def fit(self, data):
            return MyFit()

    class MyFit(object):
        pass

    class MyDirectionFinder(object):

        directions = np.array([[1., 0, 0],
                               [0, 1., 0],
                               [0, 0., 1]])

        def __call__(self, fit):
            return self.directions

    data = np.ones((10, 10, 10, 65))
    data_interp = NearestNeighborInterpolator(data, (1, 1, 1))

    mask = np.ones((10, 10, 10), 'bool')
    mask[0, 0, 0] = False
    cdt = ClosestDirectionTracker(model=MyModel(), interpolator=data_interp,
                                  mask=mask, take_step=None,
                                  angle_limit=90., seeds=None)

    # We're going to use a silly set of directions for the test
    cdt._get_directions = MyDirectionFinder()

    prev_step = np.array([[.9, .1, .1],
                          [.1, .9, .1],
                          [.1, .1, .9]])
    prev_step /= np.sqrt((prev_step * prev_step).sum(-1))[:, None]
    a, b, c = prev_step
    assert_array_equal(cdt._next_step([1., 1., 1.], a), [1, 0, 0])
    assert_array_equal(cdt._next_step([1., 1., 1.], b), [0, 1, 0])
    assert_array_equal(cdt._next_step([1., 1., 1.], c), [0, 0, 1])
    # Assert raises outside image
    assert_raises(OutsideImage, cdt._next_step, [-1., 1., 1.], c)
    # Returns None when mask is False
    assert_equal(cdt._next_step([0, 0, 0], c), None)

    # Test Angle limit
    cdt = ClosestDirectionTracker(model=MyModel(), interpolator=data_interp,
                                  mask=mask, take_step=None,
                                  angle_limit=45, seeds=None)

    # We're going to use a silly set of directions for the test
    cdt._get_directions = MyDirectionFinder()
    sq3 = np.sqrt(3)
    a = np.array([sq3 / 2, 1. / 2, 0])
    b = np.array([1. / 2, sq3 / 2, 0])
    c = np.array([1, 1, 1]) / sq3

    assert_array_equal(cdt._next_step([1., 1., 1.], a), [1, 0, 0])
    assert_array_equal(cdt._next_step([1., 1., 1.], b), [0, 1, 0])
    assert_array_equal(cdt._next_step([1., 1., 1.], c), None)


def test_ProbabilisticOdfWeightedTracker():
    sphere = HemiSphere.from_sphere(unit_octahedron)

    # A simple image with three possible configurations, a vertical tract,
    # a horizontal tract and a crossing
    odf_list = [np.array([0., 0., 0.]),
                np.array([1., 0., 0.]),
                np.array([0., 1., 0.]),
                np.array([1., 1., 0.]),
                ]
    simple_image = np.array([[0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 3, 2, 2, 2, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0],
                             ])
    # Make the image 4d
    simple_image = simple_image[..., None, None]

    # Simple model and fit for this image
    class MyModel():
        def fit(self, data):
            return MyFit(data)

    class MyFit(object):
        def __init__(self, n):
            self.n = int(n)

        def odf(self, sphere):
            return odf_list[self.n]

    seeds = [np.array([1.5, 1.5, .5])] * 30
    model = MyModel()
    mask = np.ones([5, 6, 1], dtype="bool")
    stepper = FixedSizeStepper(1.)
    interpolator = NearestNeighborInterpolator(simple_image, (1, 1, 1))

    # These are the only two possible paths though the simple_image
    pwt = ProbabilisticOdfWeightedTracker(model, interpolator, mask,
                                          stepper, 90, seeds, sphere)
    expected = [np.array([[0.5, 1.5, 0.5],
                          [1.5, 1.5, 0.5],
                          [2.5, 1.5, 0.5],
                          [2.5, 2.5, 0.5],
                          [2.5, 3.5, 0.5],
                          [2.5, 4.5, 0.5],
                          [2.5, 5.5, 0.5]]),
                np.array([[0.5, 1.5, 0.5],
                          [1.5, 1.5, 0.5],
                          [2.5, 1.5, 0.5],
                          [3.5, 1.5, 0.5],
                          [4.5, 1.5, 0.5]])
                ]

    def allclose(x, y):
        return x.shape == y.shape and np.allclose(x, y)

    path = [False, False]
    for streamline in pwt:
        if allclose(streamline, expected[0]):
            path[0] = True
        elif allclose(streamline, expected[1]):
            path[1] = True
        else:
            raise AssertionError()
    assert_(all(path))

    # The first path is not possible if 90 degree turns are excluded
    pwt = ProbabilisticOdfWeightedTracker(model, interpolator, mask,
                                          stepper, 80, seeds, sphere)
    for streamline in pwt:
        assert_(np.allclose(streamline, expected[1]))
