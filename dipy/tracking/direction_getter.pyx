cimport cython

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,)
from dipy.utils.fast_numpy cimport copy_point


cimport numpy as cnp

cdef extern from "dpy_math.h" nogil:
    int dpy_signbit(double x)
    double dpy_rint(double x)
    double fabs(double)

@cython.cdivision(True)
cdef inline double _stepsize(double point, double increment) noexcept nogil:
    """Compute the step size to the closest boundary in units of increment."""
    cdef:
        double dist
    dist = dpy_rint(point) + .5 - dpy_signbit(increment) - point
    if dist == 0:
        # Point is on an edge, return step size to next edge.  This is most
        # likely to come up if overstep is set to 0.
        return 1. / fabs(increment)
    else:
        return dist / increment

cdef void _step_to_boundary(double * point, double * direction,
                           double overstep) noexcept nogil:
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
        step_sizes[i] = _stepsize(point[i], direction[i])

    smallest_step = step_sizes[0]
    for i in range(1, 3):
        if step_sizes[i] < smallest_step:
            smallest_step = step_sizes[i]

    smallest_step += overstep
    for i in range(3):
        point[i] += smallest_step * direction[i]

cdef void _fixed_step(double * point, double * direction, double step_size) noexcept nogil:
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



cdef class DirectionGetter:

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(
            self, double[::1] point):
        pass


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] direction,
                                    double[::1] voxel_size,
                                    double step_size,
                                    StoppingCriterion stopping_criterion,
                                    cnp.float_t[:, :] streamline,
                                    StreamlineStatus stream_status,
                                    int fixedstep
                                    ):
       cdef:
           cnp.npy_intp i
           cnp.npy_intp len_streamlines = streamline.shape[0]
           double point[3]
           double voxdir[3]
           void (*step)(double*, double*, double) noexcept nogil

       if fixedstep > 0:
           step = _fixed_step
       else:
           step = _step_to_boundary

       copy_point(&seed[0], point)
       copy_point(&seed[0], &streamline[0,0])

       stream_status = TRACKPOINT
       for i in range(1, len_streamlines):
           if self.get_direction_c(point, direction):
               break
           for j in range(3):
               voxdir[j] = direction[j] / voxel_size[j]
           step(point, voxdir, step_size)
           copy_point(point, &streamline[i, 0])
           stream_status = stopping_criterion.check_point_c(point)
           if stream_status == TRACKPOINT:
               continue
           elif (stream_status == ENDPOINT or
                 stream_status == INVALIDPOINT or
                 stream_status == OUTSIDEIMAGE):
               break
       else:
           # maximum length of streamline has been reached, return everything
           i = streamline.shape[0]
       return i, stream_status

    def get_direction(self, double[::1] point, double[::1] direction):
        return self.get_direction_c(point, direction)

    cdef int get_direction_c(self, double[::1] point, double[::1] direction):
        pass
