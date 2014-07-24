cimport cython
cimport numpy as np
from .direction_getter cimport DirectionGetter
from .tissue_classifier cimport (TissueClassifier, TissueClass, TRACKPOINT,
                                 ENDPOINT, OUTSIDEIMAGE, INVALIDPOINT)


cdef extern from "dpy_math.h" nogil:
    int signbit(double x)
    double round(double x)
    double abs(double)


@cython.cdivision(True)
cdef inline double stepsize(double point, double increment) nogil:
    """Compute the step size to the closest boundary in units of increment."""
    cdef:
        double dist
    dist = round(point) + .5 - signbit(increment) - point
    if dist == 0:
        # Point is on an edge, return step size to next edge.  This is most
        # likely to come up if overstep is set to 0.
        return 1. / abs(increment)
    else:
        return dist / increment


cdef void step_to_boundry(double *point, double *direction,
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
        double step_sizes[3], smallest_step

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

    if (seed.shape[0] != 3 or first_step.shape[0] != 3 or
        voxel_size.shape[0] != 3 or streamline.shape[1] != 3):
        raise ValueError()

    cdef:
        int i
        TissueClass tssuclass
        double point[3], dir[3], vs[3], voxdir[3]
        double[::1] pview = point, dview = dir
        void (*step)(double*, double*, double) nogil

    if fixedstep:
        step = fixed_step
    else:
        step = step_to_boundry

    for i in range(3):
        point[i] = seed[i]
        dir[i] = first_step[i]
        vs[i] = voxel_size[i]

    for i in range(streamline.shape[0]):
        copypoint(point, &streamline[i, 0])
        if dg.get_direction(pview, dview):
            i += 1
            break
        for j in range(3):
            voxdir[j] = dir[j] / vs[j]
        step(point, voxdir, stepsize)
        tssuclass = tc.check_point(pview)
        if tssuclass == TRACKPOINT:
            continue
        elif tssuclass == ENDPOINT:
            i += 1
            break
        elif tssuclass == OUTSIDEIMAGE:
            break
        elif tssuclass == INVALIDPOINT:
            i = - (i + 1)
            break
    return i

