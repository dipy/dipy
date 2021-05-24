# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of parallel transport tractography (PTT)
"""

from random import random

import numpy as np
cimport numpy as cnp

from dipy.direction.closest_peak_direction_getter cimport ProbabilisticDirectionGetter
from dipy.direction.peaks import peak_directions, default_sphere
from dipy.direction.pmf cimport PmfGen, SimplePmfGen, SHCoeffPmfGen
from dipy.utils.fast_numpy cimport cumsum, where_to_insert

# These need to be defined (they are probably already available in dipy)
cdef float[3]               getAUnitRandomVector()   
cdef (float[3],float[3])    getARandomPointWithinDisk(float radius)
cdef float[3]               getAUnitRandomPerpVector(float* inp)

# Tracking Parameters
# (This might not be necessary but I am still putting it here for completeness. We can remove it later if we find it redundant.)
cdef struct TP:
    cdef float stepSize
    cdef float maxCurvature
    cdef float probeLength
    cdef float probeRadius
    cdef float probeQuality
    cdef float probeCount
    cdef float dataSupportExponent

# Parallel Trasport Frame
cdef class PTF():

    # For each streamline, create a new PTF object with tracking parameters
    def __init__(self, TP* _params):
        
        # Set this PTF's parameters
        self.params = _params

        # Initialize this PTF's internal tracking parameters
        self.angularSeparation = 2.0*np.pi/float(self.params->paramsprobeCount)
        self.probeStepSize     = self.params->probeLength/(self.params->probeQuality-1)
        self.probeNormalizer   = 1.0/float(self.params->probeQuality*self.params->probeCount)

    
    TP* params              # Tracking parameters for this frame.
    cdef float p[3]         # Last position
    cdef float F[3][3]      # Frame    
    cdef float k1           # k1 value of the current frame
    cdef float k2           # k2 value of the current frame
    cdef float k1_cand      # Candidate k1 value for the next frame
    cdef float k2_cand      # Candidate k2 value for the next frame
    cdef float likelihood   # Likelihood of the next candidate frame constructed with k1_cand and k2_cand

    # The following variables are mainly used for code optimization
    cdef float PP[9]                # Propagator
    cdef float angularSeparation
    cdef float probeStepSize
    cdef float probeNormalizer 
    cdef float lastVal
    cdef float lastVal_cand
    cdef float initFirstVal
    cdef float initFirstVal_cand
    
    # First set the (initial) position of the parallel transport frame (PTF), i.e. set the seed point
    cdef void setPosition(self, float* pos):
        self.p[0] = pos[0]
        self.p[1] = pos[1]
        self.p[2] = pos[2]

    cdef void getARandomFrame(self,float* _dir):
        if (_dir==NULL):
            F[0] = getAUnitRandomVector();
            F[2] = getAUnitRandomPerpVector(F[0]);
            cross(F[1],F[2],F[0]);
            return;
        
        F[0][0] = _dir[0];
        F[0][1] = _dir[1];
        F[0][2] = _dir[2];
        getAUnitRandomPerpVector(F[2],F[0]);
        cross(F[1],F[2],F[0]);

    
    # After initial position is set, a random PTF (a walking frame, i.e., 3 orthonormal vectors (F), plus 2 scalars, i.e., k1 and k2) is set with this function.
    # Optionally, the tangential component of PTF can be user provided with the input initDir parameter.
    # Use initDir=NULL if initDir is not available.
    # A point + PTF parametrizes a curve that is named the "probe". Using probe parameters (probeLength, probeRadius, probeQuality, probeCount),
    # a short fiber bundle segment is modelled. 
    # This function does NOT pick the initial curve. It only returns the datasupport (likelihood) value for a randomly picked candidate.
    cdef float getInitCandidate(self, float *initDir)
        getARandomFrame(initDir)
        getARandomPointWithinDisk()
        k1 = k1_cand
        k2 = k2_cand

    # Propagates the last position (p) by stepSize amount, using the parameters of the last candidate.
    cdef void walk()
    
    # Using the current position, pick a random curve parametrization. The walking frame (F) is same, only the k1 and k2 are randomly picked. This was a smooth curve is sampled.
    # This function does NOT pick the next curve. It only returns the datasupport (likelihood) value for the randomly picked candidate.
    cdef float getCandidate()

    # Copies PTF parameters then flips the curve. This function can be used after the initial curve is picked in order to save a copy of the curve for tracking towards the other side.  
    cdef void getFlippedCopy(PTF *ptf)
    
    cdef void print()


    cdef float calcDataSupport()
    cdef void  prepPropagator(float t)    



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
