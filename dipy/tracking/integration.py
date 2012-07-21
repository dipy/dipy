"""Implemention of various Tractography metods

these tools are ment to be paired with diffusion reconstruction methods from
dipy.reconst

This module uses the trackvis coordinate system, for more information about
this coordinate system please see dipy.tracking.utils
The following modules also use this coordinate system:
dipy.tracking.utils
dipy.tracking.integration
dipy.reconst.interpolate
"""
from __future__ import division
import numpy as np
from ..reconst.interpolate import OutsideImage


class BoundryIntegrator(object):
    """Integrates along step until the closest voxel boundry"""
    def __init__(self, voxel_size=(1, 1, 1), overstep=.1):
        """Creates a BoundryIntegrator instance

        Parameters:
        -----------
        voxel_size : array-like
            Size of voxels in data volume
        overstep : float
            A small number used to prevent the track from getting stuck at the
            edge of a voxel.
        """
        self.overstep = overstep
        self.voxel_size = np.array(voxel_size, 'float')

    def take_step(self, location, step):
        """takes a step just past the edge of the next voxel along step

        given a location and a step, finds the smallest step needed to move
        into the next voxel

        Parameters:
        -----------
        location : ndarray, (3,)
            location to integrate from
        step : ndarray, (3,)
            direction in 3 space to integrate along
        """
        step_sizes = self.voxel_size*(~np.signbit(step))
        step_sizes -= location % self.voxel_size
        step_sizes /= step
        smallest_step = min(step_sizes) + self.overstep
        return location + smallest_step*step


class FixedStepIntegrator(object):
    """An Intigrator that uses a fixed step size"""
    def __init__(self, step_size=.5):
        """Creates an Intigrator"""
        self.step_size = step_size
    def take_step(self, location, step):
        """Takes a step of step_size from location"""
        new_location = self.step_size*step + location
        return new_location


def closest_direction(directions, reference):
    """Track the peak closest to reference"""
    closest = abs(np.dot(directions, reference)).argmax()
    return directions[closest:closest+1]
def first_direction(directions, reference):
    """Track the first, ie biggest, peak at each seed"""
    return directions[0:1]
def all_directions(directions, reference):
    """Track all directions at a seed"""
    return directions
first_step_choosers = {'closest' : closest_direction,
                       'first' : first_direction,
                       'all' : all_directions}


class TrackStopper(object):
    """Stops a streamline if it leaves the mask or tries makes a turn larger
    than max_turn_angle degrees"""
    def __init__(self, mask_interpolator, max_turn_angle):
        if max_turn_angle < 0 or max_turn_angle > 90:
            raise ValueError("max_turn_angle must be between 0 and 90")
        self._cos_similarity = np.cos(np.deg2rad(max_turn_angle))
        self._mask = mask_interpolator

    def terminate(self, location, prev_dir, next_dir):
        """Check to see if streamline should end
        Returns True when streamline should end """
        if np.dot(prev_dir, next_dir) < self._cos_similarity:
            return True
        return not self._mask[location]


def markov_streamline(get_direction, take_step, terminate,
                      seed, first_step, maxlen):
    """Creates a streamline from seed

    Parameters
    ----------
    get_direction : callable
        This function should return a direction for the streamline given a 
        location and the previous direction.
    take_step : callable
        Take step should take a step from a location given a direction.
    terminate : callable
        Terminate should take three arguments a location, previous direction,
        and new direction; and should return True if the streamline has reached
        a stopping criteria. It should return False otherwise.
    seed : array (3,)
        The seed point of the streamline
    first_step : array (3,)
        A unit vector giving the direction of the first step
    maxlen : int
        The maximum number of segments allowed in the streamline. This is good
        for preventing infinite loops.

    Returns
    -------
    streamline : array (N, 3)
        A streamline.

    """
    streamline = []
    location = seed
    new_dir = first_step

    try:
        for i in xrange(maxlen):
            streamline.append(location)
            location = take_step(location, new_dir)
            prev_dir = new_dir
            new_dir = get_direction(location, prev_dir)
            if terminate(location, prev_dir, new_dir):
                streamline.append(location)
                break
    except OutsideImage:
        pass
    
    return np.vstack(streamline)


def generate_streamlines(get_direction, take_step, terminate,
                         seeds, get_first_step, reference=(0, 0, 1),
                         track_two_directions=False, maxlen=500):
    """A streamline generator
    
    Parameters
    ----------
    get_direction : callable
        This function should return a direction for the streamline given a 
        location and the previous direction.
    take_step : callable
        Take step should take a step from a location given a direction.
    terminate : callable
        Terminate should take three arguments a location, previous direction,
        and new direction; and should return True if the streamline has reached
        a stopping criteria. It should return False otherwise.
    seeds : array, (N, 3)
        points in space to track from.
    get_first_step : callable
        Method for choosing starting direction when multiple directions are
        available.
    reference : array (3,)
        Unit vector used to help pick starting direction when
        `track_to_directions` is False and possibly by `get_first_step`
    track_two_directions : bool
        If True, each seed is tracked in two antipodal directions.
    maxlen : int
        The maximum number of segments allowed in a streamline to prevent
        infinite loops.
    
    Returns
    -------
    streamlines : generator
        A streamline generator
    """
    for s in seeds:
        directions = get_direction(s, None)
        directions = get_first_step(directions, reference)
        for first_step in directions:
            # Align first_step with reference
            if np.dot(first_step, reference) < 0:
                first_step = -first_step
            # If track two_directions, track forward and backward
            if track_two_directions:
                F = markov_streamline(get_direction, take_step, terminate, s,
                                      first_step, maxlen)
                first_step = -first_step
                B = markov_streamline(get_direction, take_step, terminate, s,
                                      first_step, maxlen)
                yield np.concatenate([B[:0:-1], F], axis=0)
            else:
                yield markov_streamline(get_direction, take_step, terminate, s,
                                        first_step, maxlen)
