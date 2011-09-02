from numpy import abs, array, asarray, atleast_3d, broadcast_arrays, cos, \
                  dot, empty, floor, mgrid, pi, signbit, sqrt, vstack, \
                  concatenate

class FactIntegrator(object):
    """This is the integration method used for  for FACT fiber tracking, but it
    can also be used as a component of other fiber tracking methods
    """
    def __init__(self, voxel_size=(1,1,1), overstep=1e-1):
        """Creates a FactIntegrator instance

        Parameters:
        -----------
        voxel_size : array-like
            Size of voxels in data volume
        overstep : float
            A small number used to prevent the track from getting stuck at the
            edge of a voxel.
        """
        self.overstep = overstep
        self.voxel_size = asarray(voxel_size)

    def integrate(self, location, step):
        """takes a step just past the edge of the next voxel along step

        given a location and a step, finds the smallest step needed to move
        into the next voxel

        Parameters:
        -----------
        location : array-like, (3,)
            location to integrate from
        step : array-like, (3,)
            direction in 3 space to integrate along
        """
        space = location % self.voxel_size
        dist = self.voxel_size*(~signbit(step)) - space
        step_sizes = dist/step
        smallest_step = step_sizes.min()
        assert smallest_step >= 0
        smallest_step += self.overstep
        new_location = location + smallest_step*step
        return new_location

class FixedStepIntegrator(object):
    """An Intigrator that uses a fixed step size"""
    def __init__(self, step_size=.5):
        """Creates an Intigrator"""
        self.step_size = step_size
    def integrate(self, location, step):
        """Takes a step of step_size from location"""
        new_location = self.step_size*step + location
        return new_location

def generate_streamlines(stepper, integrator, seeds, start_steps):
    """Creates streamlines using a stepper and integrator

    Creates streamlines starting at each of the seeds, and integrates them
    using the integrator, getting the direction of each step from stepper.

    Parameters:
    -----------
    stepper : object with next_step method
        The next_step method should take a location, (x, y, z) cordinates in
        3-space, and the previous step, a unit vector in 3-space, and return a
        new step
    integrator : object with a integrate method
        The integrate method should take a location and a step, as defined
        above and return a new location.
    seeds : array-like (N, 3)
        Locations where to start tracking.
    start_steps : array-like
        start_steps should be broadcastable with seeds and is used in place of
        prev_step when determining the initial step for each seed.

    Returns:
    --------
    streamlines : generator
        A generator of streamlines, each streamline is an (M, 3) array of
        points in 3-space.
    """
    start_steps = asarray(start_steps)
    norm_start_steps = sqrt((start_steps*start_steps).sum(-1))
    norm_start_steps = norm_start_steps.reshape((-1, 1))
    start_steps = start_steps/norm_start_steps
    seeds, start_steps = broadcast_arrays(seeds, start_steps)

    next_step = stepper.next_step
    integrate = integrator.integrate
    for ii in xrange(len(seeds)):
        location = seeds[ii]
        step = start_steps[ii]
        streamline = []
        try:
            while True:
                step = next_step(location, step)
                streamline.append(location)
                location = integrate(location, step)
        except IndexError:
            pass
        except StopIteration:
            streamline.append(location)
        streamline = array(streamline)
        yield streamline

