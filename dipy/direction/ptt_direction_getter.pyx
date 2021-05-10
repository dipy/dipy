# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of a probabilistic direction getter based on sampling from
discrete distribution (pmf) at each step of the tracking.
"""

from random import random

import numpy as np
cimport numpy as cnp

from dipy.direction.closest_peak_direction_getter cimport ProbabilisticDirectionGetter
from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport PmfGen, SimplePmfGen, SHCoeffPmfGen
from dipy.utils.fast_numpy cimport cumsum, where_to_insert


cdef class PTTDirectionGetter(ProbabilisticDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """

    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1, **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : float, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : float [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        relative_peak_threshold : float in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : float in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See also
        --------
        dipy.direction.peaks.peak_directions

        """
        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold, **kwargs)


    cdef int propagate(self):
        <OVERWRITE>
        
    cdef int initialize(self):
        <OVERWRITE>
        
    cdef int flip(self):
        <OVERWRITE>

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] dir,
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
           void (*step)(double*, double*, double) nogil

       if fixedstep > 0:
           step = _fixed_step
       else:
           step = _step_to_boundary

       copy_point(&seed[0], point)
       copy_point(&seed[0], &streamline[0,0])

       stream_status = TRACKPOINT
      
       if (self.isInitialized()):
          self.flip()
       else:
          self.initialize()
        
       for i in range(1, len_streamlines):
           if self.propagate():
               break
           for j in range(3):
               voxdir[j] = dir[j] / voxel_size[j]
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
