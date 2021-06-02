# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of parallel transport tractography (PTT)
"""

from random import random

import numpy as np
cimport numpy as cnp

from dipy.direction.probabilistic_direction_getter cimport ProbabilisticDirectionGetter
from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport PmfGen, SimplePmfGen, SHCoeffPmfGen
from dipy.utils.fast_numpy cimport cumsum, where_to_insert, copy_point
from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,
                                               PYERROR)
from dipy.tracking.direction_getter cimport _fixed_step, _step_to_boundary


from libc.stdlib cimport rand
from libc.math cimport sqrt, fabs, M_PI, pow, sin, cos

cdef extern from "limits.h":
    int INT_MAX

# Pick a random number between -1 and 1
cdef float unidis_m1_p1():
    return 2.0*rand()/float(INT_MAX) - 1.0

cdef float norm(float[:] v):
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

cdef void normalize(float[:] v):
    cdef float scale = 1.0/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    v[0] = v[0]*scale
    v[1] = v[1]*scale
    v[2] = v[2]*scale

cdef float dot(float[:] v1, float[:] v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

cdef void cross(float[:] out, float[:] v1, float[:] v2):
    out[0] = v1[1]*v2[2] - v1[2]*v2[1]
    out[1] = v1[2]*v2[0] - v1[0]*v2[2]
    out[2] = v1[0]*v2[1] - v1[1]*v2[0]

cdef void getAUnitRandomVector(float[:] out):
    out[0] = unidis_m1_p1()
    out[1] = unidis_m1_p1()
    out[2] = unidis_m1_p1()
    normalize(out)

cdef void getAUnitRandomPerpVector(float[:] out,float[:] inp):
    cdef float[3] tmp
    getAUnitRandomVector(tmp)
    cross(out,inp,tmp)
    normalize(out)

cdef (float,float) getARandomPointWithinDisk(float r):
    cdef float x = 1
    cdef float y = 1
    while ((x*x+y*y)>1):
        x = unidis_m1_p1()
        y = unidis_m1_p1()
    return (r*x,r*y)


# Tracking Parameters
# (This might not be necessary but I am still putting it here for completeness. We can remove it later if we find it redundant.)
cdef struct TP:
    float step_size
    float max_curvature
    float probe_length
    float probe_radius
    float probe_quality
    float probe_count
    float data_support_exponent

# Parallel Trasport Frame
cdef class PTF():

    cdef TP          params                # Tracking parameters for this frame.
    cdef float[3]    p                     # Last position
    cdef float[3][3] F                     # Frame
    cdef float       k1                    # k1 value of the current frame
    cdef float       k2                    # k2 value of the current frame
    cdef float       k1_cand               # Candidate k1 value for the next frame
    cdef float       k2_cand               # Candidate k2 value for the next frame
    cdef float       likelihood            # Likelihood of the next candidate frame constructed with k1_cand and k2_cand

    # The following variables are mainly used for code optimization
    cdef float[9]    PP                    # Propagator
    cdef float       angular_separation
    cdef float       probe_step_size
    cdef float       probe_normalizer
    cdef float       last_val
    cdef float       last_val_cand
    cdef float       init_first_val
    cdef float       init_first_val_cand


    # For each streamline, create a new PTF object with tracking parameters
    def __init__(self, TP _params):

        # Set this PTF's parameters
        self.params = _params

        # Initialize this PTF's internal tracking parameters
        self.angular_separation = 2.0*M_PI/float(self.params.probe_count)
        self.probe_step_size    = self.params.probe_length/(self.params.probe_quality-1)
        self.probe_normalizer   = 1.0/float(self.params.probe_quality*self.params.probe_count)


    # First set the (initial) position of the parallel transport frame (PTF), i.e. set the seed point
    cdef void set_position(self, float[:] pos):
        self.p[0] = pos[0]
        self.p[1] = pos[1]
        self.p[2] = pos[2]

    # After initial position is set, a random PTF (a walking frame, i.e., 3 orthonormal vectors (F), plus 2 scalars, i.e., k1 and k2) is set with this function.
    # Optionally, the tangential component of PTF can be user provided with the input initDir parameter.
    # Use initdir={0,0,0} if initDir is not available.
    # A point + PTF parametrizes a curve that is named the "probe". Using probe parameters (probe_length, probe_radius, probe_quality, probe_count),
    # a short fiber bundle segment is modelled.
    # This function does NOT pick the initial curve. It only returns the datasupport (likelihood) value for a randomly picked candidate.
    cdef float getInitCandidate(self, float[:] initDir):

        cdef float    fodAmp = 0.0
        cdef float[3] pp

        self.getARandomFrame(initDir)
        (self.k1_cand,self.k2_cand) = getARandomPointWithinDisk(self.params.max_curvature)
        self.k1 = self.k1_cand
        self.k2 = self.k2_cand

        # First part of the probe
        self.likelihood = 0.0

        if (self.params.probe_count==1):
            # fodAmp = getFODamp(p,F[0])
            self.likelihood = fodAmp
        else:
            for c in range(self.params.probe_count):
                for i in range(3):
                    pp[i] = self.p[i] + self.F[1][i]*self.params.probe_radius*cos(c*self.params.angular_separation) + self.F[2][i]*self.params.probe_radius*sin(c*self.params.angular_separation)

                # fodAmp = getFODamp(pp,F[0])
                self.likelihood += fodAmp

        self.init_first_val_cand = self.likelihood
        self.last_val_cand      = self.likelihood

        return self.calcDataSupport()


    # Propagates the last position (p) by step_size amount, using the parameters of the last candidate.
    cdef void walk(self):

        self.prepPropagator(self.params.step_size)
        self.k1 = self.k1_cand
        self.k2 = self.k2_cand

        cdef float[3] T

        for i in range(3):
            self.p[i]    = self.PP[0]*self.F[0][i] +  self.PP[1]*self.F[1][i]  +  self.PP[2]*self.F[2][i] + self.p[i]
            T[i]         = self.PP[3]*self.F[0][i] +  self.PP[4]*self.F[1][i]  +  self.PP[5]*self.F[2][i]
            self.F[2][i] = self.PP[6]*self.F[0][i] +  self.PP[7]*self.F[1][i]  +  self.PP[8]*self.F[2][i]

        normalize(T)
        cross(self.F[1],self.F[2],T)
        normalize(self.F[1])
        cross(self.F[2],T,self.F[1])

        self.F[0][0]    = T[0]
        self.F[0][1]    = T[1]
        self.F[0][2]    = T[2]

        self.likelihood = 0.0

    # Using the current position, pick a random curve parametrization. The walking frame (F) is same, only the k1 and k2 are randomly picked. This was a smooth curve is sampled.
    # This function does NOT pick the next curve. It only returns the datasupport (likelihood) value for the randomly picked candidate.
    cdef float getCandidate(self):
        (self.k1_cand,self.k2_cand) = getARandomPointWithinDisk(self.params.max_curvature)
        return self.calcDataSupport()

    # Copies PTF parameters then flips the curve. This function can be used after the initial curve is picked in order to save a copy of the curve for tracking towards the other side.
    cdef void getFlippedCopy(self,PTF ptf):

        self.k1       =  ptf.k1
        self.k2       =  ptf.k2
        self.k1_cand  =  ptf.k1_cand
        self.k2_cand  =  ptf.k2_cand

        for i in range(3):
            self.p[i] = ptf.p[i]
            for j in range(3):
                self.F[i][j] = ptf.F[i][j]

        for i in range(9):
            self.PP[i]  = ptf.PP[i]

        self.likelihood 	   = ptf.likelihood
        self.init_first_val      = ptf.init_first_val
        self.last_val           = ptf.last_val
        self.init_first_val_cand = ptf.init_first_val_cand
        self.last_val_cand      = ptf.last_val_cand

        for i in range(3):
            self.F[0][i] *= -1.0
            self.F[1][i] *= -1.0

        self.k1         *= -1.0
        self.k1_cand    *= -1.0

        self.likelihood  = 0.0
        self.last_val     = self.init_first_val

    # Randomly generate 3 unit vectors that are orthogonal to each other.
    # This is used for initializing the moving frame of the tracker
    # Optionally, the initial direction, i.e., tangent, can also be provided
    # if norm(_dir.size)==0, then the tangent will also be a random vector
    cdef void getARandomFrame(self,float[:] _dir):

        if (norm(_dir)==0):
            getAUnitRandomVector(self.F[0])
        else:
            self.F[0][0] = _dir[0]
            self.F[0][1] = _dir[1]
            self.F[0][2] = _dir[2]

        getAUnitRandomPerpVector(self.F[2],self.F[0])
        cross(self.F[1],self.F[2],self.F[0])


    # Prepares the propagator, PP, that is used for transporting the moving frame forward
    cdef void prepPropagator(self, float t):

        cdef float tto2

        if ( (abs(self.k1_cand)<0.0001) & (abs(self.k2_cand)<0.0001) ) :

            self.PP[0] = t
            self.PP[1] = 0
            self.PP[2] = 0
            self.PP[3] = 1
            self.PP[4] = 0
            self.PP[5] = 0
            self.PP[6] = 0
            self.PP[7] = 0
            self.PP[8] = 1

        else :

            if (abs(self.k1_cand)<0.0001): self.k1_cand = 0.0001
            if (abs(self.k2_cand)<0.0001): self.k2_cand = 0.0001

            tto2  = t*t/2.0

            self.PP[0] = t
            self.PP[1] = self.k1_cand*tto2
            self.PP[2] = self.k2_cand*tto2
            self.PP[3] = 1-self.k2_cand*self.k2_cand*tto2-self.k1_cand*self.k1_cand*tto2
            self.PP[4] = self.k1_cand*t
            self.PP[5] = self.k2_cand*t
            self.PP[6] = -self.k2_cand*t
            self.PP[7] = -self.k1_cand*self.k2_cand*tto2
            self.PP[8] = 1-self.k2_cand*self.k2_cand*tto2

    # Calculates data support for the candidate probe
    cdef float calcDataSupport(self):

        cdef float        fodAmp = 0.0
        cdef float[3]    _p
        cdef float[3][3] _F
        cdef float[3]    _T      = {0,0,0}
        cdef float[3]    _N1     = {0,0,0}
        cdef float[3]    _N2     = {0,0,0}
        cdef float[3]     pp

        self.prepPropagator(self.params.probe_step_size)

        for i in range(3):
            _p[i] = self.p[i]
            for j in range(3):
                _F[i][j] = self.F[i][j]

        self.likelihood = self.last_val

        for q in range(self.params.probe_quality-1):

            for i in range(3):
                _p[i] = self.PP[0]*_F[0][i] +  self.PP[1]*_F[1][i]  +  self.PP[2]*_F[2][i] + _p[i]
                _T[i] = self.PP[3]*_F[0][i] +  self.PP[4]*_F[1][i]  +  self.PP[5]*_F[2][i]

            normalize(_T)

            if (q < self.params.probe_quality-1):

                for i in range(3):
                    _N2[i]  = self.PP[6]*_F[0][i] +  self.PP[7]*_F[1][i]  +  self.PP[8]*_F[2][i]

                cross(_N1,_N2,_T)

                for i in range(3):
                    _F[0][i] =  _T[i]
                    _F[1][i] = _N1[i]
                    _F[2][i] = _N2[i]


            if (self.params.probe_count==1):

                # fodAmp       = getFODamp(_p,_T)
                self.last_val_cand = fodAmp
                self.likelihood  += self.last_val_cand

            else :

                self.last_val_cand = 0

                if (q == self.params.probe_quality-1):
                    for i in range(3):
                        _N2[i]  = self.PP[6]*_F[0][i] +  self.PP[7]*_F[1][i]  +  self.PP[8]*_F[2][i]
                    cross(_N1,_N2,_T)

                for c in range(self.params.probe_count):

                    for i in range(3):
                        pp[i] = _p[i] + _N1[i]*self.params.probe_radius*cos(c*self.params.angular_separation) + _N2[i]*self.params.probe_radius*sin(c*self.params.angular_separation)

                    # fodAmp = getFODamp(pp,_T)
                    self.last_val_cand += fodAmp

                self.likelihood += self.last_val_cand

            self.prepPropagator(self.params.probe_step_size)

        self.likelihood *= self.params.probe_normalizer
        if (self.params.data_support_exponent != 1):
            self.likelihood  = pow(self.likelihood,self.params.data_support_exponent)

        return self.likelihood



cdef class PTTDirectionGetter(ProbabilisticDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """

    # cdef int propagate(self):
    #     return 0

    # cdef int initialize(self):
    #     return 0

    # cdef int flip(self):
    #     return 0

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

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)

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
