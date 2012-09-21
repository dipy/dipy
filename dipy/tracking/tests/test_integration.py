import numpy as np
from dipy.tracking.integration import (BoundryIntegrator, FixedStepIntegrator,
                                       generate_streamlines, markov_streamline,
                                       OutsideImage, TrackStopper)

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_)

def test_BoundryIntegrator():
    os = 1
    bi = BoundryIntegrator(overstep=os)
    loc = np.array([.5,.5,.5])
    step = np.array([1,1,1])/np.sqrt(3)
    assert_array_almost_equal(bi.take_step(loc, step), os*step + [1,1,1])
    assert_array_almost_equal(bi.take_step(loc, -step), -os*step)

    os = 2
    bi = BoundryIntegrator((2,3,4), overstep=2)
    assert_array_almost_equal(bi.take_step(loc, step), os*step + [2,2,2])
    assert_array_almost_equal(bi.take_step(loc, -step), -os*step)

    loc = np.array([7.5,7.5,7.5])
    assert_array_almost_equal(bi.take_step(loc, step), os*step + [8,8,8])
    assert_array_almost_equal(bi.take_step(loc, -step), [6,6,6] - os*step)

def test_FixedStepIntegrator():
    fsi = FixedStepIntegrator(step_size=2.)
    loc = np.array([2,3,12])
    step = np.array([3,2,4])/np.sqrt(3)
    assert_array_almost_equal(fsi.take_step(loc, step), loc + 2.*step)
    assert_array_almost_equal(fsi.take_step(loc, -step), loc - 2.*step)


def test_trackstopper():
    class Mask(object):
        def __getitem__(self, index):
            if any(index > 10):
                return False
            else:
                return True
    stopper = TrackStopper(Mask(), 30)
    east = np.array([1, 0, 0])
    north = np.array([0, 1, 0])
    home = np.array([0, 0, 0])
    away = np.array([12, 0, 0])

    # Stop because of location
    assert_(stopper.terminate(away, east, east))
    # Stop because of angle
    assert_(stopper.terminate(home, east, north))
    # Do not stop
    assert_(not stopper.terminate(home, north, north))


def test_markov_streamline():

    east = np.array([1, 0, 0])
    class MoveEastWest(object):
        def get_direction(self, location, prev_dir):
            if np.any(location < 0):
                raise OutsideImage
            if np.dot(prev_dir, east) >= 0:
                return east
            else:
                return -east

    class StopAtTen(object):
        def terminate(self, location, prev_dir, next_dir):
            if np.any(location >= 10):
                return True
            else:
                return False

    seed = np.array([5.2, 0, 0])
    first_step = east
    dir_getter = MoveEastWest()
    stepper = FixedStepIntegrator(.5)
    stopper = StopAtTen()

    # The streamline terminates when it goes past (10, 0, 0). (10.2, 0, 0) 
    # should be the last point in the streamline
    streamline = markov_streamline(dir_getter.get_direction, stepper.take_step,
                                   stopper.terminate, seed, first_step, 100)
    expected = np.zeros((11, 3))
    expected[:, 0] = np.linspace(5.2, 10.2, 11)
    assert_array_almost_equal(streamline, expected)

    # OutsideImage gets raised when the streamline points become negative
    # the streamline should end, and the negative points should not be part
    # of the streamline
    first_step = -east
    streamline = markov_streamline(dir_getter.get_direction, stepper.take_step,
                                   stopper.terminate, seed, first_step, 100)
    expected = np.zeros((11, 3))
    expected[:, 0] = np.linspace(5.2, 0.2, 11)
    assert_array_almost_equal(streamline, expected)

def test_generate_streamlines():

    class KeepGoing(object):
        def get_direction(self, location, prev_dir):
            if prev_dir is None:
                return np.array([[1., 0, 0],
                                 [0, 1., 0],
                                 [0, 0., 1]])
            else:
                return prev_dir

    class StopAtTen(object):
        def terminate(self, location, prev_dir, next_dir):
            if np.any(location < 0):
                raise OutsideImage
            if np.any(location >= 10):
                return True
            else:
                return False

    seeds = [np.array([5.2, 5.2, 5.2])]
    stepper = FixedStepIntegrator(.5)
    nav = KeepGoing()
    stopper = StopAtTen()
    gen = generate_streamlines(nav.get_direction, stepper.take_step,
                               stopper.terminate, seeds)
    streamlines = list(gen)
    assert_equal(len(streamlines), 3)
    expected = np.zeros((21, 3))
    for i in range(3):
        expected[:] = 5.2
        expected[:, i] = np.linspace(.2, 10.2, 21)
        assert_array_almost_equal(streamlines[i], expected)

    # Track only the first (largest) peak for each seed
    gen = generate_streamlines(nav.get_direction, stepper.take_step,
                               stopper.terminate, seeds, max_cross=1)
    streamlines = list(gen)
    assert_equal(len(streamlines), 1)
    expected = np.zeros((21, 3))
    expected[:] = 5.2
    expected[:, 0] = np.linspace(.2, 10.2, 21)
    assert_array_almost_equal(streamlines[0], expected)

