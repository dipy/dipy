cimport cython
cimport numpy as np
from .direction_getter cimport DirectionGetter
from .tissue_classifier cimport (TissueClassifier, TissueClass, TRACKPOINT,
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


cdef void step_to_boundary(double *point, double *direction,
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


cdef void fixed_step(double *point, double *direction, double stepsize) nogil:
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


cdef inline void copypoint(double *a, double *b) nogil:
    for i in range(3):
        b[i] = a[i]


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
        void (*step)(double*, double*, double) nogil

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
        copypoint(point, &streamline[i, 0])
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
