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
from ..reconst.interpolate import OutsideImage, NearestNeighborInterpolator


class BoundaryStepper(object):
    """Steps along a direction past the closest voxel boundary

    Parameters:
    -----------
    voxel_size : array-like
        Size of voxels in data volume
    overstep : float
        A small number used to prevent the track from getting stuck at the
        edge of a voxel.

    """
    def __init__(self, voxel_size=(1, 1, 1), overstep=.1):
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


class FixedSizeStepper(object):
    """A stepper that uses a fixed step size"""
    def __init__(self, step_size=.5):
        self.step_size = step_size

    def take_step(self, location, step):
        """Takes a step of step_size from location"""
        new_location = self.step_size*step + location
        return new_location


def markov_streamline(get_direction, take_step, seed, first_step, maxlen):
    """Creates a streamline from seed

    Parameters
    ----------
    get_direction : callable
        This function should return a direction for the streamline given a 
        location and the previous direction.
    take_step : callable
        Take step should take a step from a location given a direction.
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
    direction = first_step

    try:
        for i in xrange(maxlen):
            streamline.append(location)
            location = take_step(location, direction)
            direction = get_direction(location, direction)
            if direction is None:
                streamline.append(location)
                break
    except OutsideImage:
        pass

    return np.vstack(streamline)


class MarkovIntegrator(object):
    """An abstract class for fiber-tracking"""

    def __init__(self, model, interpolator, mask, take_step, angle_limit,
                 seeds, max_cross=None, maxlen=500, mask_voxel_size=None):
        """Creates streamlines by using a Markov approach.

        Parameters
        ----------
        model : model
            The model used to fit diffusion data.
        interpolator : interpolator
            Diffusion weighted data wrapped in an interpolator. Data should be
            normalized.
        mask : array, 3D
            Used to confine tracking, streamlines are terminated if the
            tracking leaves the mask.
        take_step : callable
            Determines the length of each step.
        angle_limit : float [0, 90]
            Maximum angle allowed between successive steps of the streamline.
        seeds : array (N, 3)
            Points to seed the tracking.
        max_cross : int or None
            The maximum number of direction to track from each seed in crossing
            voxels.  By default track all peaks of the odf, otherwise track the
            largest `max_cross` peaks.
        maxlen : int
            Maximum number of steps to track from seed. Used to prevent
            infinite loops.
        mask_voxel_size : array (3,)
            Voxel size for the mask. `mask` should cover the same FOV as data,
            but it can have a different voxel size. Same as the data by
            default.

        """
        self.model = model
        self.interpolator = interpolator
        self.seeds = seeds
        self.max_cross = max_cross
        self.maxlen = maxlen

        self._take_step = take_step
        self._cos_similarity = np.cos(np.deg2rad(angle_limit))

        if mask_voxel_size is None:
            assert mask.shape == interpolator.data.shape[:-1]
            mask_voxel_size = interpolator.voxel_size
        else:
            mask_voxel_size = np.asarray(mask_voxel_size)
            mask_FOV = mask_voxel_size * mask.shape
            data_FOV = interpolator.voxel_size * interpolator.data.shape
            assert np.allclose(mask_FOV, data_FOV)
        self._mask = NearestNeighborInterpolator(mask.copy(), mask_voxel_size)

    def __iter__(self):
        return self._generate_streamlines()

    def _generate_streamlines(self):
        """A streamline generator"""
        for s in self.seeds:
            directions = self._next_step(s, prev_step=None)
            directions = directions[:self.max_cross]
            for first_step in directions:
                F = markov_streamline(self._next_step, self._take_step, s,
                                      first_step, self.maxlen)
                first_step = -first_step
                B = markov_streamline(self._next_step, self._take_step, s,
                                      first_step, self.maxlen)
                yield np.concatenate([B[:0:-1], F], axis=0)


def _closest_peak(peak_directions, prev_step, cos_similarity):
    """Return the closest direction to prev_step from peak_directions.

    All directions should be unit vectors. Antipodal symmetry is assumed, ie
    direction x is the same as -x.

    Parameters
    ----------
    peak_directions : array (N, 3)
        N unit vectors.
    prev_step : array (3,) or None
        Previous direction.
    cos_similarity : float
        `cos(max_angle)` where `max_angle` is the maximum allowed angle between
        prev_step and the returned direction.

    Returns
    -------
    direction : array or None
        If prev_step is None, returns peak_directions. Otherwise returns the
        closest direction to prev_step. If no directions are close enough to
        prev_step, returns None
    """
    if prev_step is None:
        return peak_directions

    peak_dots = np.dot(peak_directions, prev_step)
    closest_peak = abs(peak_dots).argmax()
    dot_closest_peak = peak_dots[closest_peak]
    if dot_closest_peak >= cos_similarity:
        return peak_directions[closest_peak]
    elif dot_closest_peak <= -cos_similarity:
        return -peak_directions[closest_peak]
    else:
        return None


class ClosestDirectionTracker(MarkovIntegrator):

    def _next_step(self, location, prev_step):
        """Returns the direction closest to prev_step at location

        Fits the data from location using model and returns the tracking
        direction closest to prev_step. If prev_step is None, all the
        directions are returned.

        Parameters
        ----------
        location : point in space
            location is passed to the interpolator in order to get data
        prev_step: array_like (3,)
            the direction of the previous tracking step

        """
        if not self._mask[location]:
            return None
        vox_data = self.interpolator[location]
        fit = self.model.fit(vox_data)
        return _closest_peak(fit.directions, prev_step, self._cos_similarity)


class ProbabilisticOdfWeightedTracker(MarkovIntegrator):
    """A stochastic (probabilistic) fiber tracking method

    Stochastically tracks streamlines by randomly choosing directions from
    sphere. The likelihood of a direction being chosen is taken from
    `model.fit(data).odf(sphere)`. Negative values are set to 0. If no
    directions less than `angle_limit` degrees are from the incoming direction
    have a positive likelihood, the streamline is terminated.

    Parameters
    ----------
    model : model
        The model used to fit diffusion data.
    interpolator : interpolator
        Diffusion weighted data wrapped in an interpolator. Data should be
        normalized.
    mask : array, 3D
        Used to confine tracking, streamlines end when they leave the mask.
    take_step : callable
        Determines the length of each step.
    angle_limit : float [0, 90]
        The angle between successive steps in the streamlines cannot be more
        than `angle_limit` degrees.
    seeds : array (N, 3)
        Points to seed the tracking.
    sphere : Sphere
        sphere used to evaluate the likelihood. A Sphere is ok, a HemiSphere is
        better.
    max_cross : int or None
        Max number of directions to follow at each seed. By default follow all
        peaks of the odf.
    maxlen : int
        Maximum number of segments to follow from seed. Used to prevent
        infinite loops.
    mask_voxel_size : array (3,)
        Voxel size for the mask. `mask` should cover the same FOV as data, but
        it can have a different voxel size. Same as the data by default.

    """
    def __init__(self, model, interpolator, mask, take_step, angle_limit,
                 seeds, sphere, max_cross=None, maxlen=500,
                 mask_voxel_size=None):

        MarkovIntegrator.__init__(self, model, interpolator, take_step,
                                  angle_limit=angle_limit, mask=mask)
        self.sphere = sphere
        self._set_adjacency_matrix(sphere, self._cos_similarity)
        model.direction_finder.config(sphere=sphere)

    def _set_adjacency_matrix(self, sphere, cos_similarity):
        """A boolean array of where the angle between vertices i and j of
        sphere is less than `angle_limit` apart."""
        matrix = np.dot(sphere.vertices, sphere.vertices.T)
        matrix = abs(matrix) > cos_similarity
        keys = [tuple(v) for v in sphere.vertices]
        adj_matrix = dict(zip(keys, matrix))
        keys = [tuple(-v) for v in sphere.vertices]
        adj_matrix.update(zip(keys, matrix))
        self._adj_matrix = adj_matrix

    def _next_step(self, location, prev_step):
        """Returns the direction closest to prev_step at location

        Fits the data from location using model and returns the tracking
        direction closest to prev_step. If prev_step is None, all the
        directions are returned.

        Parameters
        ----------
        location : point in space
            location is passed to the interpolator in order to get data
        prev_step: array_like (3,)
            the direction of the previous tracking step

        """
        if not self._mask[location]:
            return None
        vox_data = self.interpolator[location]
        fit = self.model.fit(vox_data)
        if prev_step is None:
            return fit.directions
        odf = fit.odf(self.sphere)
        odf = odf.clip(0)
        cdf = (self._adj_matrix[tuple(prev_step)] * odf).cumsum()
        if cdf[-1] == 0:
            return None
        random_sample = np.random.random() * cdf[-1]
        idx = cdf.searchsorted(random_sample, 'right')
        direction = self.sphere.vertices[idx]
        if np.dot(direction, prev_step) > 0:
            return direction
        else:
            return -direction


class CDT_NNO(ClosestDirectionTracker):
    """ClosestDirectionTracker optimized for NearestNeighbor interpolator

    For use with Nearest Neighbor interpolation, directions at each voxel are
    remembered to avoid recalculating.

    Parameters:
    -----------
    model : model
        A model used to fit data. Should return a some fit object with
        directions.
    interpolator : interpolator
        A NearestNeighbor interpolator, for other interpolators do not use this
        class.
    angle_limit : float [0, 90]
        Maximum angle allowed between prev_step and next_step.

    """
    def __init__(self, model, interpolator, mask, take_step, angle_limit,
                 seeds, max_cross=None, maxlen=500, mask_voxel_size=None):
        if not isinstance(interpolator, NearestNeighborInterpolator):
            msg = ("CDT_NNO is an optimized version of "
                   "ClosestDirectionTracker that requires a "
                   "NearestNeighborInterpolator")
            raise ValueError(msg)

        ClosestDirectionTracker.__init__(self, model, interpolator, mask,
                                         take_step, angle_limit, seeds,
                                         max_cross=max_cross, maxlen=maxlen,
                                         mask_voxel_size=mask_voxel_size)
        self._data = self.interpolator.data
        self._voxel_size = self.interpolator.voxel_size
        self.reset_cache()

    def reset_cache(self):
        """Clear saved directions"""
        lookup = np.empty(self._data.shape[:-1], 'int')
        lookup.fill(-1)
        self._lookup = lookup
        self._peaks = []

    def _next_step(self, location, prev_step):
        """Returns the direction closest to prev_step at location"""
        if not self._mask[location]:
            return None

        vox_loc = tuple(location // self._voxel_size)
        hash = self._lookup[vox_loc]
        if hash >= 0:
            directions = self._peaks[hash]
        else:
            vox_data = self._data[vox_loc]
            fit = self.model.fit(vox_data)
            directions = fit.directions
            self._lookup[vox_loc] = len(self._peaks)
            self._peaks.append(directions)

        return _closest_peak(directions, prev_step, self._cos_similarity)


