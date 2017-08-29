cimport cython

cimport numpy as np
import numpy as np
from .direction_getter cimport DirectionGetter
from .tissue_classifier cimport(TissueClassifier, TissueClass, TRACKPOINT,
                                ENDPOINT, OUTSIDEIMAGE, INVALIDPOINT)


cdef extern from "dpy_math.h" nogil:
    int dpy_signbit(double x)
    double dpy_rint(double x)
    double abs(double)


@cython.cdivision(True)
cdef inline double stepsize(double point, double increment) nogil:
    """Compute the step size to the closest boundary in units of increment."""
    cdef:
        double dist
    dist = dpy_rint(point) + .5 - dpy_signbit(increment) - point
    if dist == 0:
        # Point is on an edge, return step size to next edge.  This is most
        # likely to come up if overstep is set to 0.
        return 1. / abs(increment)
    else:
        return dist / increment


cdef void step_to_boundary(double * point, double * direction,
                           double overstep) nogil:
    """Takes a step from point in along direction just past a voxel boundary.

    Parameters
    ----------
    direction : c-pointer to double[3]
        The direction along which the step should be taken.
    point : c-pointer to double[3]
        The tracking point which will be updated by this function.
    overstep : double
        It's often useful to have the points of a streamline lie inside of a
        voxel instead of having them lie on the boundary. For this reason,
        each step will overshoot the boundary by ``overstep * direction``.
        This should not be negative.

    """
    cdef:
        double step_sizes[3]
        double smallest_step

    for i in range(3):
        step_sizes[i] = stepsize(point[i], direction[i])

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        point[i] += smallest_step * direction[i]


cdef void fixed_step(double * point, double * direction, double stepsize) nogil:
    """Updates point by stepping in direction.

    Parameters
    ----------
    direction : c-pointer to double[3]
        The direction along which the step should be taken.
    point : c-pointer to double[3]
        The tracking point which will be updated by this function.
    step_size : double
        The size of step in units of direction.

    """
    for i in range(3):
        point[i] += direction[i] * stepsize


cdef inline void copypoint(double * a, double * b) nogil:
    for i in range(3):
        b[i] = a[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def pft_tracker(np.ndarray[np.float_t, ndim=1] seed,
                np.ndarray[np.float_t, ndim=1] first_step,
                np.ndarray[np.float_t, ndim=2, mode='c'] streamline,
                np.ndarray[np.float_t, ndim=2, mode='c'] directions,
                DirectionGetter dg,
                TissueClassifier tc,
                np.ndarray[np.float_t, ndim=1] voxel_size,
                double step_size,
                int pft_nbr_back_steps,
                int pft_max_steps,
                int pft_max_trial,
                int pft_nbr_particles,
                np.ndarray[np.float_t, ndim=4, mode='c'] particle_paths,
                np.ndarray[np.float_t, ndim=4, mode='c'] particle_dirs,
                np.ndarray[np.float_t, ndim=2, mode='c'] particle_weights,
                np.ndarray[np.int_t, ndim=3, mode='c'] particle_states):

    if (seed.shape[0] != 3 or first_step.shape[0] != 3 or
            voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError()
    cdef:
        int i, pft_trial, pft_streamline_i, pft_nbr_steps
        TissueClass tissue_class
        double point[3], dir[3], vs[3], voxdir[3]
        double[::1] pview = point, dview = dir
        void (*step)(double * , double*, double) nogil
    pft_trial = 0

    for j in range(3):
        streamline[0, j] = point[j] = seed[j]
        dir[j] = first_step[j]
        vs[j] = voxel_size[j]
    copypoint(dir, & directions[0, 0])

    tissue_class = TRACKPOINT
    i = 0
    while i < streamline.shape[0] - 1:
        if dg.get_direction(pview, dview):
            # no valid diffusion directions to follow
            tissue_class = INVALIDPOINT
        else:
            for j in range(3):
                voxdir[j] = dir[j] / vs[j]
            i += 1
            fixed_step(point, voxdir, step_size)
            copypoint(point, & streamline[i, 0])
            copypoint(dir, & directions[i, 0])

            tissue_class = tc.check_point(pview)

        if tissue_class == TRACKPOINT:
            continue
        elif tissue_class == ENDPOINT:
            i += 1
            break
        elif tissue_class == INVALIDPOINT:
            if pft_trial < pft_max_trial and i > 1:
                pft_streamline_i = min(i - 1, pft_nbr_back_steps)
                pft_nbr_steps = min(pft_max_steps,
                                    streamline.shape[0] - pft_streamline_i - 1)
                tissue_class, i = _pft(streamline,
                                       pft_streamline_i,
                                       directions,
                                       dg,
                                       tc,
                                       voxel_size,
                                       step_size,
                                       pft_nbr_steps,
                                       pft_nbr_particles,
                                       particle_paths,
                                       particle_dirs,
                                       particle_weights,
                                       particle_states)

                pft_trial += 1
                # update the current point
                for j in range(3):
                    point[j] = streamline[i, j]
                    dir[j] = directions[i, j]

                if not tissue_class == TRACKPOINT:
                    break
            else:
                i += 1
                break
        elif tissue_class == OUTSIDEIMAGE:
            break
    return i, tissue_class


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _pft(np.ndarray[np.float_t, ndim=2, mode='c'] streamline,
          int streamline_i,
          np.ndarray[np.float_t, ndim=2, mode='c'] directions,
          DirectionGetter dg,
          TissueClassifier tc,
          np.ndarray[np.float_t, ndim=1] voxel_size,
          double step_size,
          int pft_nbr_steps,
          int pft_nbr_particles,
          np.ndarray[np.float_t, ndim=4, mode='c'] particle_paths,
          np.ndarray[np.float_t, ndim=4, mode='c'] particle_dirs,
          np.ndarray[np.float_t, ndim=2, mode='c'] particle_weights,
          np.ndarray[np.int_t, ndim=3, mode='c'] particle_states):
    cdef:
        double sum_weights, sum_squared, N_effective
        double point[3], dir[3], vs[3], voxdir[3]
        double[::1] pview = point, dview = dir
        int s, p, j

    if pft_nbr_steps <= 0:
        return INVALIDPOINT, streamline_i

    for j in range(3):
        vs[j] = voxel_size[j]

    for p in range(pft_nbr_particles):
        for j in range(3):
            particle_paths[0, p, 0, j] = streamline[streamline_i, j]
            particle_dirs[0, p, 0, j] = directions[streamline_i, j]
        particle_weights[0, p] = 1. / pft_nbr_particles
        particle_states[0, p, 0] = TRACKPOINT
        particle_states[0, p, 1] = 0

    for s in range(pft_nbr_steps):
        for p in range(pft_nbr_particles):
            if not particle_states[0, p, 0] == TRACKPOINT:
                for j in range(3):
                    particle_paths[0, p, s, j] = 0
                    particle_dirs[0, p, s, j] = 0
                continue  # move to the next particle
            for j in range(3):
                point[j] = particle_paths[0, p, s, j]
                dir[j] = particle_dirs[0, p, s, j]
            if dg.get_direction(pview, dview):
                particle_states[0, p, 0] = INVALIDPOINT
            else:
                for j in range(3):
                    voxdir[j] = dir[j] / vs[j]
                fixed_step(point, voxdir, step_size)

                for j in range(3):
                    particle_paths[0, p, s + 1, j] = point[j]
                    particle_dirs[0, p, s + 1, j] = dir[j]

                particle_states[0, p, 0] = tc.check_point(pview)
                particle_states[0, p, 1] = s + 1
                particle_weights[0, p] *= 1 - tc.get_exclude(pview)
                if (particle_states[0, p, 0] == INVALIDPOINT and
                        particle_weights[0, p] > 0):
                    particle_states[0, p, 0] = TRACKPOINT

        sum_weights = 0
        for p in range(pft_nbr_particles):
            sum_weights += particle_weights[0, p]
        sum_squared = 0
        for p in range(pft_nbr_particles):
            particle_weights[0, p] = particle_weights[0, p] / sum_weights
            sum_squared += particle_weights[0, p] * particle_weights[0, p]

        # Resample the particles if the weights are too uneven.
        # Particles with negligable weights are replaced by duplicates of
        # those with high weigths through resamplingip
        N_effective = 1. / sum_squared
        if N_effective < pft_nbr_particles / 10.:
            # copy data in the temp arrays
            for pp in range(pft_nbr_particles):
                for ss in range(pft_nbr_steps):
                    for j in range(3):
                        particle_paths[1, pp, ss,
                                       j] = particle_paths[0, pp, ss, j]
                        particle_dirs[1, pp, ss,
                                      j] = particle_dirs[0, pp, ss, j]
                particle_weights[1, pp] = particle_weights[0, pp]
                particle_states[1, pp, 0] = particle_states[0, pp, 0]
                particle_states[1, pp, 1] = particle_states[0, pp, 1]
            # sample N new particle
            for pp in range(pft_nbr_particles):
                p_source = particle_weights[1, :].cumsum().searchsorted(
                    np.random.random(), 'right')
                for ss in range(pft_nbr_steps):
                    for j in range(3):
                        particle_paths[0, pp, ss,
                                       j] = particle_paths[1, p_source, ss, j]
                        particle_dirs[0, pp, ss,
                                      j] = particle_dirs[1, p_source, ss, j]
                particle_states[0, pp, 0] = particle_states[1, p_source, 0]
                particle_states[0, pp, 1] = particle_states[1, p_source, 1]
                particle_weights[0, pp] = 1. / pft_nbr_particles

    # update the streamline with the trajectory of one particle
    p = particle_weights[0, :].cumsum().searchsorted(
        np.random.random(), 'right')
    for s in range(particle_states[0, p, 1]):
        for j in range(3):
            streamline[streamline_i + s, j] = particle_paths[0, p, s, j]
            directions[streamline_i + s, j] = particle_dirs[0, p, s, j]
    return particle_states[0, p, 0], streamline_i + \
        particle_states[0, p, 1] - 1


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def local_tracker(DirectionGetter dg, TissueClassifier tc,
                  np.ndarray[np.float_t, ndim=1] seed,
                  np.ndarray[np.float_t, ndim=1] first_step,
                  np.ndarray[np.float_t, ndim=1] voxel_size,
                  np.ndarray[np.float_t, ndim=2, mode='c'] streamline,
                  double stepsize,
                  int fixedstep):
    """Tracks one direction from a seed.

    This function is the main workhorse of the ``LocalTracking`` class defined
    in ``dipy.tracking.local.localtracking``.

    Parameters
    ----------
    dg : DirectionGetter
        Used to choosing tracking directions.
    tc : TissueClassifier
        Used to check tissue type along path.
    seed : array, float, 1d, (3,)
        First point of the (partial) streamline.
    first_step : array, float, 1d, (3,)
        Used as ``prev_dir`` for selecting the step direction from the seed
        point.
    voxel_size : array, float, 1d, (3,)
        Size of voxels in the data set.
    streamline : array, float, 2d, (N, 3)
        Output of tracking will be put into this array. The length of this
        array, ``N``, will set the maximum allowable length of the streamline.
    stepsize : float
        Size of tracking steps in mm if ``fixed_step``.
    fixedstep : bool
        If true, a fixed stepsize is used, otherwise a variable step size is
        used.

    Returns
    -------
    end : int
        This function updates the ``streamline`` array with points as it
        tracks. Points in ``streamline[:abs(end)]`` were updated by the
        function. The sign of ``end`` and whether the last point was included
        depend on the reason that the streamline was terminated.

        End reasons:
            1) maximum length of the streamline was reached.
                ``end == N``
            2) ``direction_getter`` could not return a direction.
                ``end > 0``
                Last point is the point at which no direction could be found.
            3) Streamline encountered an ENDPOINT.
                ``end > 0``
                Last point is the ENDPOINT.
            3) Streamline encountered an OUTSIDEIMAGE.
                ``end > 0``
                Last point is the point before OUTSIDEIMAGE.
            5) Streamline encountered an INVALIDPOINT.
                ``end < 0``
                Last point is INVALIDPOINT.

    """
    if (seed.shape[0] != 3 or first_step.shape[0] != 3 or
            voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError()

    cdef:
        int i
        TissueClass tissue_class
        double point[3], dir[3], vs[3], voxdir[3]
        double[::1] pview = point, dview = dir
        void (*step)(double * , double*, double) nogil

    if fixedstep:
        step = fixed_step
    else:
        step = step_to_boundary

    for i in range(3):
        streamline[0, i] = point[i] = seed[i]
        dir[i] = first_step[i]
        vs[i] = voxel_size[i]

    tissue_class = TRACKPOINT
    for i in range(1, streamline.shape[0]):
        if dg.get_direction(pview, dview):
            break
        for j in range(3):
            voxdir[j] = dir[j] / vs[j]
        step(point, voxdir, stepsize)
        copypoint(point, & streamline[i, 0])
        tissue_class = tc.check_point(pview)
        if tissue_class == TRACKPOINT:
            continue
        elif (tissue_class == ENDPOINT or
              tissue_class == INVALIDPOINT):
            i += 1
            break
        elif tissue_class == OUTSIDEIMAGE:
            break
    else:
        # maximum length of streamline has been reached, return everything
        i = streamline.shape[0]
    return i, tissue_class
