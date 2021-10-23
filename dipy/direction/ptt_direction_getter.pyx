# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False

"""
Implementation of parallel transport tractography (PTT)
"""

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
                                               PYERROR,
                                               NODATASUPPORT)
from dipy.tracking.direction_getter cimport _fixed_step, _step_to_boundary


from libc.stdlib cimport rand
from libc.math cimport sqrt, fabs, M_PI, pow, sin, cos

cdef extern from "limits.h":
    int INT_MAX

# Pick a random number between 0 and 1
cdef double uniform_01():
    return rand()/float(INT_MAX)

# Pick a random number between -1 and 1
cdef double unidis_m1_p1():
    return 2.0 * uniform_01() - 1.0

cdef double norm(double[:] v):
    return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])

cdef void normalize(double[:] v):
    cdef double scale = 1.0/sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    v[0] = v[0]*scale
    v[1] = v[1]*scale
    v[2] = v[2]*scale

cdef double dot(double[:] v1, double[:] v2):
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

cdef void cross(double[:] out, double[:] v1, double[:] v2):
    out[0] = v1[1]*v2[2] - v1[2]*v2[1]
    out[1] = v1[2]*v2[0] - v1[0]*v2[2]
    out[2] = v1[0]*v2[1] - v1[1]*v2[0]

cdef void getAUnitRandomVector(double[:] out):
    out[0] = unidis_m1_p1()
    out[1] = unidis_m1_p1()
    out[2] = unidis_m1_p1()
    normalize(out)

cdef void getAUnitRandomPerpVector(double[:] out,double[:] inp):
    cdef double[3] tmp
    getAUnitRandomVector(tmp)
    cross(out,inp,tmp)
    normalize(out)

cdef (double,double) getARandomPointWithinDisk(double r):
    cdef double x = 1
    cdef double y = 1
    while ((x*x+y*y)>1):
        x = unidis_m1_p1()
        y = unidis_m1_p1()
    return (r*x,r*y)


# Tracking Parameters
# (This might not be necessary but I am still putting it here for completeness. We can remove it later if we find it redundant.)
cdef struct TP:
    double step_size
    double max_curvature
    double probe_length
    double probe_radius
    double probe_quality
    double probe_count
    double data_support_exponent

cdef class PTTDirectionGetter(ProbabilisticDirectionGetter):
    """Randomly samples direction of a sphere based on probability mass
    function (pmf).

    The main constructors for this class are current from_pmf and from_shcoeff.
    The pmf gives the probability that each direction on the sphere should be
    chosen as the next direction. To get the true pmf from the "raw pmf"
    directions more than ``max_angle`` degrees from the incoming direction are
    set to 0 and the result is normalized.
    """

    cdef TP         params                # Tracking parameters for this frame.
    cdef double[3]    p                     # Last position
    cdef double[3][3] F                     # Frame
    cdef double       k1                    # k1 value of the current frame
    cdef double       k2                    # k2 value of the current frame
    cdef double       k1_cand               # Candidate k1 value for the next frame
    cdef double       k2_cand               # Candidate k2 value for the next frame
    cdef double       likelihood            # Likelihood of the next candidate frame constructed with k1_cand and k2_cand

    cdef bint       initialized           # True is initialization was done. This is used for flipping.
    cdef double[3]    init_p                # Initial position
    cdef double[3][3] init_F                # Initial frame
    cdef double       init_k1               # Initial k1 value of the current frame
    cdef double       init_k2               # Initial k2 value of the current frame


    # The following variables are mainly used for code optimization
    cdef double[9]    PP                    # Propagator
    cdef double       angular_separation
    cdef double       probe_step_size
    cdef double       probe_normalizer
    cdef double       last_val
    cdef double       last_val_cand
    cdef double       init_last_val


    # For each streamline, create a new PTF object with tracking parameters
    def __init__(self, pmf_gen, max_angle, sphere, pmf_threshold=.1,
                 max_curvature=1/2, probe_length=1/2, probe_radius=0,
                 probe_quality=3, probe_count=1, data_support_exponent=1,
                 **kwargs):
        """Direction getter from a pmf generator.

        Parameters
        ----------
        pmf_gen : PmfGen
            Used to get probability mass function for selecting tracking
            directions.
        max_angle : double, [0, 90]
            The maximum allowed angle between incoming direction and new
            direction.
        sphere : Sphere
            The set of directions to be used for tracking.
        pmf_threshold : double [0., 1.]
            Used to remove direction from the probability mass function for
            selecting the tracking direction.
        relative_peak_threshold : double in [0., 1.]
            Used for extracting initial tracking directions. Passed to
            peak_directions.
        min_separation_angle : double in [0, 90]
            Used for extracting initial tracking directions. Passed to
            peak_directions.

        See also
        --------
        dipy.direction.peaks.peak_directions

        """
        # Set this PTF's parameters
        self.params = TP()

        # TODO: review max_angle vs max_curvature.
        self.params.max_curvature = max_curvature
        self.params.probe_length = probe_length
        self.params.probe_radius = probe_radius
        self.params.probe_quality = probe_quality
        self.params.probe_count = probe_count
        self.params.data_support_exponent = data_support_exponent
        ProbabilisticDirectionGetter.__init__(self, pmf_gen, max_angle, sphere,
                                       pmf_threshold, **kwargs)

        # Initialize this PTF's internal tracking parameters
        self.angular_separation = 2.0*M_PI/float(self.params.probe_count)
        self.probe_step_size    = self.params.probe_length/(self.params.probe_quality-1)
        self.probe_normalizer   = 1.0/float(self.params.probe_quality*self.params.probe_count)


    # First set the (initial) position of the parallel transport frame (PTF), i.e. set the seed point
    cdef void set_position(self, double[:] pos):
        self.p[0] = pos[0]
        self.p[1] = pos[1]
        self.p[2] = pos[2]


    cdef double get_initial_candidate(self, double[:] init_dir):
        """"Return the likelihood value for a randomly picked candidate.

        After initial position is set, a random PTF (a walking frame, i.e.,
        3 orthonormal vectors (F), plus 2 scalars, i.e., k1 and k2) is set
        with this function. Optionally, the tangential component of PTF can be
        user provided with the input initDir parameter.
        A point + PTF parametrizes a curve that is named the "probe". Using
        probe parameters (probe_length, probe_radius, probe_quality,
        probe_count), a short fiber bundle segment is modelled.

        Parameters
        ----------
        init_dir : np.array
            Use initdir=[0,0,0] if initDir is not available.

        Notes
        -----
        This function does NOT pick the initial curve. It only returns the
        datasupport (likelihood) value for a randomly picked candidate.

        """
        cdef double fod_amp
        cdef double[3] pp

        self.getARandomFrame(init_dir)
        (self.k1_cand,self.k2_cand) = getARandomPointWithinDisk(self.params.max_curvature)
        self.k1 = self.k1_cand
        self.k2 = self.k2_cand

        self.last_val = 0

        if self.params.probe_count==1:
            fod_amp = self.pmf_gen.get_pmf_value(self.p, self.F[0])
            self.last_val = fod_amp
        else:
            for c in range(self.params.probe_count):
                for i in range(3):
                    pp[i] = self.p[i] + self.F[1][i]*self.params.probe_radius*cos(c*self.params.angular_separation) + self.F[2][i]*self.params.probe_radius*sin(c*self.params.angular_separation)

                fod_amp = self.pmf_gen.get_pmf_value(pp, self.F[0])
                self.last_val += fod_amp

        self.init_last_val = self.last_val

        return self.calcDataSupport()

    cdef void walk(self):
        """Propagates the last position (p) by step_size amount.

        The progation is using the parameters of the last candidate.

        """
        self.k1 = self.k1_cand
        self.k2 = self.k2_cand

        cdef double[3] T

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

    cdef double get_candidate(self):
        """Pick a random curve parametrization by using current position.

        The walking frame (F) is same, only the k1 and k2 are randomly picked.
        This was a smooth curve is sampled. This function does NOT pick the
        next curve. It only returns the datasupport (likelihood) value for the
        randomly picked candidate.

        """
        self.k1_cand, self.k2_cand = getARandomPointWithinDisk(self.params.max_curvature)
        return self.calcDataSupport()

    cdef void flip(self):
        """"Copy PTF parameters then flips the curve.

        This function can be used after the initial curve is picked in order
        to save a copy of the curve for tracking towards the other side.
        """

        for i in range(3):
            self.p[i] = self.init_p[i]
            for j in range(3):
                self.F[i][j] = self.init_F[i][j]

        self.k1_cand  = self.init_k1
        self.k2_cand  = self.init_k2
        self.last_val = self.init_last_val

        return

    cdef void getARandomFrame(self,double[:] _dir):
        """Randomly generate 3 unit vectors that are orthogonal to each other.

        This is used for initializing the moving frame of the tracker
        Optionally, the initial direction, i.e., tangent, can also be provided
        if norm(_dir.size)==0, then the tangent will also be a random vector

        """
        if norm(_dir) == 0:
            getAUnitRandomVector(self.F[0])
        else:
            self.F[0][0] = _dir[0]
            self.F[0][1] = _dir[1]
            self.F[0][2] = _dir[2]

        getAUnitRandomPerpVector(self.F[2],self.F[0])
        cross(self.F[1],self.F[2],self.F[0])


    cdef void prepare_propagator(self, double t):
        """Prepare the propagator.

        The propagator PP, that is used for transporting the moving frame
        forward.

        """
        cdef double tto2

        if (abs(self.k1_cand)<0.0001) & (abs(self.k2_cand)<0.0001):
            self.PP[0] = t
            self.PP[1] = 0
            self.PP[2] = 0
            self.PP[3] = 1
            self.PP[4] = 0
            self.PP[5] = 0
            self.PP[6] = 0
            self.PP[7] = 0
            self.PP[8] = 1
        else:
            if abs(self.k1_cand)<0.0001: self.k1_cand = 0.0001
            if abs(self.k2_cand)<0.0001: self.k2_cand = 0.0001

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

    cdef double calcDataSupport(self):
        """Calculate data support for the candidate probe."""

        cdef double        fod_amp
        cdef double[3]    _p
        cdef double[3][3] _F
        cdef double[3]    _T    = [0,0,0]
        cdef double[3]    _N1     = [0,0,0]
        cdef double[3]    _N2     = [0,0,0]
        cdef double[3]     pp

        self.prepare_propagator(self.probe_step_size)

        for i in range(3):
            _p[i] = self.p[i]
            for j in range(3):
                _F[i][j] = self.F[i][j]

        self.likelihood = self.last_val

        for q in range(1, int(self.params.probe_quality)):

            for i in range(3):
                _p[i] = self.PP[0]*_F[0][i] +  self.PP[1]*_F[1][i]  +  self.PP[2]*_F[2][i] + _p[i]
                _T[i] = self.PP[3]*_F[0][i] +  self.PP[4]*_F[1][i]  +  self.PP[5]*_F[2][i]

            normalize(_T)

            if q < (self.params.probe_quality-1):

                for i in range(3):
                    _N2[i]  = self.PP[6]*_F[0][i] +  self.PP[7]*_F[1][i]  +  self.PP[8]*_F[2][i]

                cross(_N1,_N2,_T)

                for i in range(3):
                    _F[0][i] =  _T[i]
                    _F[1][i] = _N1[i]
                    _F[2][i] = _N2[i]


            if self.params.probe_count == 1:
                fod_amp = self.pmf_gen.get_pmf_value(_p, _T)
                self.last_val_cand = fod_amp
                self.likelihood += self.last_val_cand
            else:
                self.last_val_cand = 0

                if q == self.params.probe_quality-1:
                    for i in range(3):
                        _N2[i]  = self.PP[6]*_F[0][i] +  self.PP[7]*_F[1][i]  +  self.PP[8]*_F[2][i]
                    cross(_N1,_N2,_T)

                for c in range(int(self.params.probe_count)):

                    for i in range(3):
                        pp[i] = _p[i] + _N1[i]*self.params.probe_radius*cos(c*self.params.angular_separation) + _N2[i]*self.params.probe_radius*sin(c*self.params.angular_separation)

                    fod_amp = self.pmf_gen.get_pmf_value(pp, _T)
                    self.last_val_cand += fod_amp

                self.likelihood += self.last_val_cand

        self.likelihood *= self.probe_normalizer
        if self.params.data_support_exponent != 1:
            self.likelihood  = pow(self.likelihood, self.params.data_support_exponent)

        return self.likelihood


    cdef StreamlineStatus reinitialize(self, double[:] _seed_point, double[:] _seed_direction):
        """Sample an initial curve by rejection sampling."""

        # Reset initialization
        self.initialized = False

        # Set initial position
        self.set_position(_seed_point)

        # Initial max estimate
        cdef double dataSupport  = 0
        cdef double posteriorMax = 0

        cdef int tries
        for tries in range(1000):
            dataSupport = self.get_initial_candidate(_seed_direction)
            if dataSupport > posteriorMax:
                posteriorMax = dataSupport

        # Compensation for underestimation of max posterior estimate
        posteriorMax = pow(2.0*posteriorMax,self.params.data_support_exponent)

        # Initialization is successful if a suitable candidate can be sampled within the trial limit
        for tries in range(1000):
            if uniform_01()*posteriorMax < self.get_initial_candidate(_seed_direction):

                self.last_val = self.last_val_cand

                for i in range(3):
                    self.init_p[i]    =  self.p[i]
                    self.init_F[0][i] = -self.F[0][i]
                    self.init_F[1][i] = -self.F[1][i]
                    self.init_F[2][i] =  self.F[2][i]

                self.init_k1     = -self.k1
                self.init_k2     =  self.k2
                self.initialized = True

                return TRACKPOINT


        return NODATASUPPORT

    cdef StreamlineStatus propagate(self):
        self.prepare_propagator(self.params.step_size)

        self.walk()

        # Initial max estimate
        cdef double dataSupport  = 0
        cdef double posteriorMax = 0

        cdef int tries
        for tries in range(20): # This is adaptively set in Trekker. But let's ignore it for now since implementation of that is challenging.
            dataSupport = self.get_candidate()
            if dataSupport > posteriorMax:
                posteriorMax = dataSupport

        # Compensation for underestimation of max posterior estimate
        posteriorMax = pow(2.0*posteriorMax,self.params.data_support_exponent)

        # Propagation is successful if a suitable candidate can be sampled within the trial limit
        for tries in range(1000):
            if uniform_01()*posteriorMax < self.get_candidate():
                self.last_val = self.last_val_cand
                return TRACKPOINT

        return NODATASUPPORT


    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] dir,
                                    #TODO: Move step_size, voxel_size, fixed_Step variable
                                    # should be outside of generate_streamline
                                    # use it inside the init
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

        self.params.step_size = step_size

        copy_point(&seed[0], point)
        copy_point(&seed[0], &streamline[0,0])

        stream_status = TRACKPOINT

        # An initial direction can be provided to the tracker for initialization
        # But for now let's assume no initialiation is done, i.e., [0,0,0]
        cdef double[3] seed_direction = [0,0,0]
        cdef double[3] seed_point     = [seed[0],seed[1],seed[2]]

        # This step only initializes or flips the frame at the seed point.
        # It does not do any propagation, i.e., tracker is at the seed point and it is ready to propagate
        # Initialization basically checks whether the tracker has data support to move or propagate forward (but it does not move it)
        if self.initialized:
            self.flip()
        else:
            stream_status = self.reinitialize(seed_point,seed_direction)

        # If initialization is successful than the tracker propagates.
        # Propagation first pushes the moving frame forward.
        # Then the propagator sets the frame for the next propagation step (here data support for the next frame is checked)
        # i.e. self.p is always supported by data (it should be added to the streamline unless there is another reason such a termination ROI etc.)
        # If propagator returns NODATASUPPORT, further propagation is not possible
        # If propagator returns TRACKPOINT, self.p can be pushed forward
        # Propagator works like this because it does not check termination conditions.
        # If self.p reaches a termination ROI, then it should not be appended to the streamline.
        for i in range(1, len_streamlines):
            stream_status = self.propagate()
            if stream_status == NODATASUPPORT:
                break
            copy_point(<double *>&self.p[0], &streamline[i, 0])
            stream_status = stopping_criterion.check_point_c(<double *>&self.p[0])
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
