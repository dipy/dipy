import numpy as np
from dipy.tracking.integration import BoundryIntegrator, FixedStepIntegrator, \
    generate_streamlines
from numpy.testing import assert_array_almost_equal, assert_array_equal

def test_BoundryIntegrator():
    os = 1
    bi = BoundryIntegrator(overstep=os)
    loc = np.array([.5,.5,.5])
    step = np.array([1,1,1])/np.sqrt(3)
    assert_array_almost_equal(bi.integrate(loc, step), os*step + [1,1,1])
    assert_array_almost_equal(bi.integrate(loc, -step), -os*step)

    os = 2
    bi = BoundryIntegrator((2,3,4), overstep=2)
    assert_array_almost_equal(bi.integrate(loc, step), os*step + [2,2,2])
    assert_array_almost_equal(bi.integrate(loc, -step), -os*step)

    loc = np.array([7.5,7.5,7.5])
    assert_array_almost_equal(bi.integrate(loc, step), os*step + [8,8,8])
    assert_array_almost_equal(bi.integrate(loc, -step), [6,6,6] - os*step)

def test_FixedStepIntegrator():
    fsi = FixedStepIntegrator(step_size=2.)
    loc = np.array([2,3,12])
    step = np.array([3,2,4])/np.sqrt(3)
    assert_array_almost_equal(fsi.integrate(loc, step), loc + 2.*step)
    assert_array_almost_equal(fsi.integrate(loc, -step), loc - 2.*step)

def test_generate_streamlines():
    steps = np.array([[1,0,0],
                      [1,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,0,1]])
    class TestStepper(object):
        state = -1
        error = StopIteration
        def next_step(self, loc, prev_step):
            self.state += 1
            if self.state >= len(steps):
                raise self.error()
            return steps[self.state]

    stepper = TestStepper()
    ss = 2.
    integrator = FixedStepIntegrator(step_size=ss)
    seeds = [[2,3,4],[5,6,7]]
    slgen = generate_streamlines(stepper, integrator, seeds, [1,0,0])
    for ii, streamline in enumerate(slgen):
        stepper.state = -1
        assert_array_equal(streamline[0], seeds[ii])
        expected_streamline = (steps*ss).cumsum(0)+seeds[ii]
        assert_array_almost_equal(streamline[1:], expected_streamline)

    stepper.error = IndexError
    slgen = generate_streamlines(stepper, integrator, seeds, [1,0,0])
    for ii, streamline in enumerate(slgen):
        stepper.state = -1
        assert_array_equal(streamline[0], seeds[ii])
        expected_streamline = (steps*ss).cumsum(0)+seeds[ii]
        assert_array_almost_equal(streamline[1:], expected_streamline[:-1])
