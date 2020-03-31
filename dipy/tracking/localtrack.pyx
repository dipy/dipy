
from random import random

cimport cython
cimport numpy as np
import numpy as np
from .direction_getter cimport DirectionGetter
from .stopping_criterion cimport(
    StreamlineStatus, StoppingCriterion, AnatomicalStoppingCriterion,
    TRACKPOINT, ENDPOINT, OUTSIDEIMAGE, INVALIDPOINT, PYERROR)
from dipy.core.interpolation cimport trilinear_interpolate4d_c
from dipy.utils.fast_numpy cimport cumsum, where_to_insert, copy_point


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


cdef void fixed_step(double * point, double * direction, double step_size) nogil:
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
        point[i] += direction[i] * step_size


def local_tracker(
        DirectionGetter dg,
        StoppingCriterion sc,
        np.float_t[:] seed_pos,
        np.float_t[:] first_step,
        np.float_t[:] voxel_size,
        np.float_t[:, :] streamline,
        double step_size,
        int fixedstep):
    """Tracks one direction from a seed.

    This function is the main workhorse of the ``LocalTracking`` class defined
    in ``dipy.tracking.local_tracking``.

    Parameters
    ----------
    dg : DirectionGetter
        Used to choosing tracking directions.
    sc : StoppingCriterion
        Used to check the streamline status (e.g. endpoint) along path.
    seed_pos : array, float, 1d, (3,)
        First point of the (partial) streamline.
    first_step : array, float, 1d, (3,)
        Initial seeding direction. Used as ``prev_dir`` for selecting the step
        direction from the seed point.
    voxel_size : array, float, 1d, (3,)
        Size of voxels in the data set.
    streamline : array, float, 2d, (N, 3)
        Output of tracking will be put into this array. The length of this
        array, ``N``, will set the maximum allowable length of the streamline.
    step_size : float
        Size of tracking steps in mm if ``fixed_step``.
    fixedstep : bool
        If true, a fixed step_size is used, otherwise a variable step size is
        used.

    Returns
    -------
    end : int
        Length of the tracked streamline
    stream_status : StreamlineStatus
        Ending state of the streamlines as determined by the StoppingCriterion.
    """
    cdef:
        size_t i
        StreamlineStatus stream_status
        double dir[3]
        double vs[3]
        double seed[3]

    if (seed_pos.shape[0] != 3 or first_step.shape[0] != 3 or
            voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError('Invalid input parameter dimensions.')

    for i in range(3):
        dir[i] = first_step[i]
        vs[i] = voxel_size[i]
        seed[i] = seed_pos[i]

    i = _local_tracker(dg, sc, seed, dir, vs, streamline,
                       step_size, fixedstep, &stream_status)
    return i, stream_status


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef int _local_tracker(DirectionGetter dg,
                        StoppingCriterion sc,
                        double* seed,
                        double* dir,
                        double* voxel_size,
                        np.float_t[:, :] streamline,
                        double step_size,
                        int fixedstep,
                        StreamlineStatus* stream_status):
    cdef:
        size_t i
        double point[3]
        double voxdir[3]
        void (*step)(double*, double*, double) nogil

    if fixedstep:
        step = fixed_step
    else:
        step = step_to_boundary

    copy_point(seed, point)
    copy_point(seed, &streamline[0,0])

    stream_status[0] = TRACKPOINT
    for i in range(1, streamline.shape[0]):
        if dg.get_direction_c(point, dir):
            break
        for j in range(3):
            voxdir[j] = dir[j] / voxel_size[j]
        step(point, voxdir, step_size)
        copy_point(point, &streamline[i, 0])
        stream_status[0] = sc.check_point_c(point)
        if stream_status[0] == TRACKPOINT:
            continue
        elif (stream_status[0] == ENDPOINT or
              stream_status[0] == INVALIDPOINT or
              stream_status[0] == OUTSIDEIMAGE):
            break
    else:
        # maximum length of streamline has been reached, return everything
        i = streamline.shape[0]
    return i


def pft_tracker(
        DirectionGetter dg,
        AnatomicalStoppingCriterion sc,
        np.float_t[:] seed_pos,
        np.float_t[:] first_step,
        np.float_t[:] voxel_size,
        np.float_t[:, :] streamline,
        np.float_t[:, :] directions,
        double step_size,
        int pft_max_nbr_back_steps,
        int pft_max_nbr_front_steps,
        int pft_max_trials,
        int particle_count,
        np.float_t[:, :, :, :] particle_paths,
        np.float_t[:, :, :, :] particle_dirs,
        np.float_t[:] particle_weights,
        np.int_t[:, :]  particle_steps,
        np.int_t[:, :]  particle_stream_statuses):
    """Tracks one direction from a seed using the particle filtering algorithm.

    This function is the main workhorse of the ``ParticleFilteringTracking``
    class defined in ``dipy.tracking.local_tracking``.

    Parameters
    ----------
    dg : DirectionGetter
        Used to choosing tracking directions.
    sc : AnatomicalStoppingCriterion
        Used to check the streamline status (e.g. endpoint) along path.
    seed_pos : array, float, 1d, (3,)
        First point of the (partial) streamline.
    first_step : array, float, 1d, (3,)
        Initial seeding direction. Used as ``prev_dir`` for selecting the step
        direction from the seed point.
    voxel_size : array, float, 1d, (3,)
        Size of voxels in the data set.
    streamline : array, float, 2d, (N, 3)
        Output of tracking will be put into this array. The length of this
        array, ``N``, will set the maximum allowable length of the streamline.
    directions : array, float, 2d, (N, 3)
        Output of tracking directions will be put into this array. The length
        of this array, ``N``, will set the maximum allowable length of the
        streamline.
    step_size : float
        Size of tracking steps in mm if ``fixed_step``.
    pft_max_nbr_back_steps : int
        Number of tracking steps to back track before starting the particle
        filtering tractography.
    pft_max_nbr_front_steps : int
        Number of additional tracking steps to track.
    pft_max_trials : int
        Maximum number of trials for the particle filtering tractography
        (Prevents infinite loops).
    particle_count : int
        Number of particles to use in the particle filter.
    particle_paths : array, float, 4d, (2, particle_count, pft_max_steps, 3)
        Temporary array for paths followed by all particles.
    particle_dirs : array, float, 4d, (2, particle_count, pft_max_steps, 3)
        Temporary array for directions followed by particles.
    particle_weights : array, float, 1d (particle_count)
        Temporary array for the weights of particles.
    particle_steps : array, float, (2, particle_count)
        Temporary array for the number of steps of particles.
    particle_stream_statuses : array, float, (2, particle_count)
        Temporary array for the stream status of particles.

    Returns
    -------
    end : int
        Length of the tracked streamline
    stream_status : StreamlineStatus
        Ending state of the streamlines as determined by the StoppingCriterion.

    """
    cdef:
        size_t i
        StreamlineStatus stream_status
        double dir[3]
        double vs[3]
        double seed[3]

    if (seed_pos.shape[0] != 3 or first_step.shape[0] != 3 or
            voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError('Invalid input parameter dimensions.')

    for i in range(3):
        dir[i] = first_step[i]
        vs[i] = voxel_size[i]
        seed[i] = seed_pos[i]

    i = _pft_tracker(dg, sc, seed, dir, vs, streamline,
                     directions, step_size, &stream_status,
                     pft_max_nbr_back_steps, pft_max_nbr_front_steps,
                     pft_max_trials, particle_count, particle_paths,
                     particle_dirs, particle_weights, particle_steps,
                     particle_stream_statuses)
    return i, stream_status


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _pft_tracker(DirectionGetter dg,
                  AnatomicalStoppingCriterion sc,
                  double* seed,
                  double* dir,
                  double* voxel_size,
                  np.float_t[:, :] streamline,
                  np.float_t[:, :] directions,
                  double step_size,
                  StreamlineStatus * stream_status,
                  int pft_max_nbr_back_steps,
                  int pft_max_nbr_front_steps,
                  int pft_max_trials,
                  int particle_count,
                  np.float_t[:, :, :, :] particle_paths,
                  np.float_t[:, :, :, :] particle_dirs,
                  np.float_t[:] particle_weights,
                  np.int_t[:, :] particle_steps,
                  np.int_t[:, :] particle_stream_statuses):
    cdef:
        int i, pft_trial, pft_streamline_i, back_steps, front_steps
        int strl_array_len
        double point[3]
        double voxdir[3]
        void (*step)(double* , double*, double) nogil

    copy_point(seed, point)
    copy_point(seed, &streamline[0,0])
    copy_point(dir, &directions[0, 0])

    stream_status[0] = TRACKPOINT
    pft_trial = 0
    i = 1
    strl_array_len = streamline.shape[0]
    while i < strl_array_len:
        if dg.get_direction_c(point, dir):
            # no valid diffusion direction to follow
            stream_status[0] = INVALIDPOINT
        else:
            for j in range(3):
                voxdir[j] = dir[j] / voxel_size[j]

            fixed_step(point, voxdir, step_size)
            copy_point(point, &streamline[i, 0])
            copy_point(dir, &directions[i, 0])
            stream_status[0] = sc.check_point_c(point)
            i += 1
        if stream_status[0] == TRACKPOINT:
            # The tracking continues normally
            continue
        elif stream_status[0] == INVALIDPOINT:
            if pft_trial < pft_max_trials and i > 1:
                back_steps = min(i - 1, pft_max_nbr_back_steps)
                front_steps = min(strl_array_len - i - back_steps - 1,
                                  pft_max_nbr_front_steps)
                front_steps = max(0, front_steps)
                i = _pft(streamline, i - back_steps, directions, dg, sc,
                         voxel_size, step_size, stream_status,
                         back_steps + front_steps, particle_count,
                         particle_paths, particle_dirs, particle_weights,
                         particle_steps, particle_stream_statuses)
                pft_trial += 1
                # update the current point with the PFT results
                copy_point(&streamline[i-1, 0], point)
                copy_point(&directions[i-1, 0], dir)

                if stream_status[0] != TRACKPOINT:
                    # The tracking stops. PFT returned a valid stopping point
                    # (ENDPOINT, OUTSIDEIMAGE) or failed to find one
                    # (INVALIDPOINT, PYERROR)
                    break
            else:
                # PFT was run more times than `pft_max_trials` without finding
                # a valid stopping point. The tracking stops with INVALIDPOINT.
                break
        else:
            # The tracking stops with a valid point (ENDPOINT, OUTSIDEIMAGE)
            # or an invalid point (PYERROR)
            break

    if stream_status[0] == OUTSIDEIMAGE or stream_status[0] == PYERROR:
        i -= 1
    return i


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef _pft(np.float_t[:, :] streamline,
          int streamline_i,
          np.float_t[:, :] directions,
          DirectionGetter dg,
          AnatomicalStoppingCriterion sc,
          double* voxel_size,
          double step_size,
          StreamlineStatus * stream_status,
          int pft_nbr_steps,
          int particle_count,
          np.float_t[:, :, :, :] particle_paths,
          np.float_t[:, :, :, :] particle_dirs,
          np.float_t[:] particle_weights,
          np.int_t[:, :] particle_steps,
          np.int_t[:, :] particle_stream_statuses):
    cdef:
        double sum_weights, sum_squared, N_effective, rdm_sample
        double point[3]
        double dir[3]
        double voxdir[3]
        double eps = 1e-16
        int s, p, j

    if pft_nbr_steps <= 0:
        return streamline_i

    for p in range(particle_count):
        copy_point(&streamline[streamline_i, 0], &particle_paths[0, p, 0, 0])
        copy_point(&directions[streamline_i, 0], &particle_dirs[0, p, 0, 0])
        particle_weights[p] = 1. / particle_count
        particle_stream_statuses[0, p] = TRACKPOINT
        particle_steps[0, p] = 0

    for s in range(pft_nbr_steps):
        for p in range(particle_count):
            if particle_stream_statuses[0, p] != TRACKPOINT:
                for j in range(3):
                    particle_paths[0, p, s, j] = 0
                    particle_dirs[0, p, s, j] = 0
                continue  # move to the next particle
            copy_point(&particle_paths[0, p, s, 0], point)
            copy_point(&particle_dirs[0, p, s, 0], dir)

            if dg.get_direction_c(point, dir):
                particle_stream_statuses[0, p] = INVALIDPOINT
                particle_weights[p] = 0
            else:
                for j in range(3):
                    voxdir[j] = dir[j] / voxel_size[j]
                fixed_step(point, voxdir, step_size)
                copy_point(point, &particle_paths[0, p, s + 1, 0])
                copy_point(dir, &particle_dirs[0, p, s + 1, 0])
                particle_stream_statuses[0, p] = sc.check_point_c(point)
                particle_steps[0, p] = s + 1
                particle_weights[p] *= 1 - sc.get_exclude_c(point)
                if particle_weights[p] < eps:
                    particle_weights[p] = 0
                if (particle_stream_statuses[0, p] == INVALIDPOINT and
                        particle_weights[p] > 0):
                    particle_stream_statuses[0, p] = TRACKPOINT

        sum_weights = 0
        for p in range(particle_count):
            sum_weights += particle_weights[p]

        if sum_weights > 0:
            sum_squared = 0
            for p in range(particle_count):
                particle_weights[p] = particle_weights[p] / sum_weights
                sum_squared += particle_weights[p] * particle_weights[p]

            # Resample the particles if the weights are too uneven.
            # Particles with negligible weights are replaced by duplicates of
            # those with high weights through resampling
            N_effective = 1. / sum_squared
            if N_effective < particle_count / 10.:
                # copy data in the temp arrays
                for pp in range(particle_count):
                    for ss in range(pft_nbr_steps):
                        copy_point(&particle_paths[0, pp, ss, 0],
                                  &particle_paths[1, pp, ss, 0])
                        copy_point(&particle_dirs[0, pp, ss, 0],
                                  &particle_dirs[1, pp, ss, 0])
                    particle_stream_statuses[1, pp] = \
                            particle_stream_statuses[0, pp]
                    particle_steps[1, pp] = particle_steps[0, pp]

                # sample N new particle
                cumsum(&particle_weights[0],
                       &particle_weights[0],
                       particle_count)
                for pp in range(particle_count):
                    rdm_sample = random() * particle_weights[particle_count - 1]
                    p_source = where_to_insert(&particle_weights[0],
                                               rdm_sample,
                                               particle_count)
                    for ss in range(pft_nbr_steps):
                        copy_point(&particle_paths[1, p_source, ss, 0],
                                  &particle_paths[0, pp, ss, 0])
                        copy_point(&particle_dirs[1, p_source, ss, 0],
                                  &particle_dirs[0, pp, ss, 0])
                    particle_stream_statuses[0, pp] = \
                            particle_stream_statuses[1, p_source]
                    particle_steps[0, pp] = particle_steps[1, p_source]
                for pp in range(particle_count):
                    particle_weights[pp] = 1. / particle_count

    # update the streamline with the trajectory of one particle
    cumsum(&particle_weights[0],
           &particle_weights[0],
           particle_count)
    if particle_weights[particle_count - 1] > 0:
        rdm_sample = random() * particle_weights[particle_count - 1]
        p = where_to_insert(&particle_weights[0], rdm_sample, particle_count)
    else:
        p = 0

    for s in range(1, particle_steps[0, p]):
        copy_point(&particle_paths[0, p, s, 0],
                   &streamline[streamline_i + s, 0])
        copy_point(&particle_dirs[0, p, s, 0], &directions[streamline_i + s, 0])
    stream_status[0] = <StreamlineStatus> particle_stream_statuses[0, p]
    return streamline_i + particle_steps[0, p]
